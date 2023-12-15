import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import os, gc, re, warnings
warnings.filterwarnings("ignore")
import random
import time
import category_encoders as ce
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import json
import pickle

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def add_columns(data, id_map):
    sm_name_to_smiles = data.set_index('sm_name')['SMILES'].to_dict()
    id_map['SMILES'] = id_map['sm_name'].map(sm_name_to_smiles)
    return id_map
    
def read_data(data_path):
    df_de_train = pd.read_parquet(os.path.join(data_path, 'de_train.parquet'))
    # target_cols = df_de_train.columns.to_list()[5:]
    id_map = pd.read_csv(os.path.join(data_path, 'id_map.csv'))
    id_map = add_columns(df_de_train, id_map)
    # smiles_list = df_de_train['SMILES'].unique()
    sample_submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    return df_de_train, id_map, sample_submission
    
class ResNetRegression(nn.Module):
    def __init__(self, input_size, output_size, pretrained=False, reshape_size=32, num_channels=16, dropout_rate=0.2):
        super(ResNetRegression, self).__init__()
        self.reshape_size = reshape_size

        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(1, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(num_channels, num_channels*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(num_channels*2, num_channels*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(num_channels * 4 * input_size, self.reshape_size * self.reshape_size),
        )

        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Reshape x for Conv1d
        x = self.conv1d_layers(x)
        x = x.view(x.size(0), -1)  # Flatten for the linear layer
        x = self.fc_layers(x)
        x = x.view(x.size(0), 1, self.reshape_size, self.reshape_size)
        x = self.resnet(x)
        return x
    

def custom_mean_rowwise_rmse(y_pred, y_true):
    """
    Custom metric to calculate the Mean Rowwise Root Mean Squared Error (RMSE) in PyTorch.

    Parameters:
    - y_true: The true target values (NumPy array).
    - y_pred: The predicted values (NumPy array).

    Returns:
    - Mean Rowwise RMSE as a scalar tensor.
    """
    y_true = torch.from_numpy(y_true)
    y_pred = torch.from_numpy(y_pred)

    rmse_per_row = torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=1))
    mean_rmse = torch.mean(rmse_per_row)
    
    return mean_rmse


def prepare_test_dataloader(data, batch_size=32):
    tensor_x = torch.Tensor(data)

    dataset = TensorDataset(tensor_x) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def prepare_dataloader(data, labels, batch_size=32):
    tensor_x = torch.Tensor(data)
    tensor_y = torch.Tensor(labels)

    dataset = TensorDataset(tensor_x, tensor_y) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_model(path, model, criterion, optimizer, train_loader, val_loader, reducer, num_epochs, patience, fold=-1):
    if not os.path.exists(path):
        os.makedirs(path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_loss = float('inf')
    best_score = float('inf')
    epochs_no_improve = 0
    early_stop = False
    train_losses, val_losses, train_metrics, val_metrics = [], [], [], []

    for epoch in range(num_epochs):
        start_time = time.time()

        if early_stop:
            print("Early stopping triggered")
            break

        model.train()
        total_loss, total_metric = 0.0, 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            outputs = reducer.inverse_transform(outputs.cpu().detach().numpy())
            labels = reducer.inverse_transform(labels.cpu().detach().numpy())
            metric = custom_mean_rowwise_rmse(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_metric += metric.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_metric = total_metric / len(train_loader)
        train_losses.append(avg_train_loss)
        train_metrics.append(avg_train_metric)

        model.eval()
        total_loss, total_metric = 0.0, 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                outputs = reducer.inverse_transform(outputs.cpu().detach().numpy())
                labels = reducer.inverse_transform(labels.cpu().detach().numpy())

                metric = custom_mean_rowwise_rmse(outputs, labels)

                total_loss += loss.item()
                total_metric += metric.item()

        avg_val_loss = total_loss / len(val_loader)
        avg_val_metric = total_metric / len(val_loader)
        val_losses.append(avg_val_loss)
        val_metrics.append(avg_val_metric)

        end_time = time.time()
        epoch_duration = end_time - start_time

        if avg_val_metric < best_score:
            best_score = avg_val_metric
            best_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'{path}/{fold}_best_model_checkpoint.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered due to no improvement in validation loss")
                early_stop = True
                break

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Metric: {avg_train_metric:.4f}, Val Loss: {avg_val_loss:.4f}, Val Metric: {avg_val_metric:.4f}, Duration: {epoch_duration:.2f} seconds')

    if fold != -1:
        return best_loss, best_score
    else:
        return train_losses, val_losses, train_metrics, val_metrics
    
    
def get_kfold_from_MT(de_train):
    folds_index_data_MT = []
    fold_to_compounds = {   
        0: ['Alvocidib', 'Belinostat', 'Foretinib', 'LDN 193189',  'Linagliptin', 'O-Demethylated Adapalene'],
        1: ['Dabrafenib', 'Dactolisib', 'Idelalisib', 'MLN 2238', 'Palbociclib', 'Porcn Inhibitor III'],
        2: ['CHIR-99021', 'Crizotinib', 'Oprozomib (ONX 0912)', 'Penfluridol',  'R428']
        }
    for fold_id in [0,1,2]:
        mask_va = de_train['cell_type'].isin(['Myeloid cells', 'B cells']) & de_train['sm_name'].isin(fold_to_compounds[fold_id])
        mask_tr = ~mask_va    
        IX_train = np.where( mask_tr > 0 )[0]
        IX_test = np.where( mask_va > 0 )[0]
        folds_index_data_MT.append( [IX_train, IX_test])
    return folds_index_data_MT


def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def inference(model, tsvd, test_loader, device, original_shape):
    with torch.no_grad():
        model_preds = []
        for feature in test_loader:
            feature = feature[0].to(device)
            preds = model(feature)
            model_preds.append(preds.cpu())

        predictions = torch.cat(model_preds, dim=0)
        predictions = predictions.view(original_shape)
        predictions = tsvd.inverse_transform(predictions.numpy())
        return predictions
    

def MT_cross_validate_model(NN_MODEL_DIR, de_train, id_map, batch_size=8, num_epochs=25, patience=20):
    labels = de_train.drop(columns=["cell_type","sm_name","sm_lincs_id","SMILES","control"]).values
    features_columns = ["cell_type", "sm_name"]
    features = pd.DataFrame(de_train, columns=features_columns).values
    test_data = pd.DataFrame(id_map, columns=features_columns).values
    
    mae_scores = []
    mrrmse_scores = []
    test_preds = []
    model_list = []
    sigma = 0.27
    print("***********************For MT Kfolds***********************")
    folds_index_data_MT = get_kfold_from_MT(de_train)
    for fold, (train_ids, test_ids) in enumerate(folds_index_data_MT):
        print('Fold', fold)
        
        tsvd = TruncatedSVD(n_components=45, n_iter=3, random_state=42)
        y_train = tsvd.fit_transform(labels[train_ids])
        y_val = tsvd.transform(labels[test_ids])

        X_train, X_val = features[train_ids], features[test_ids]

        if sigma == 'default':
            enc = ce.LeaveOneOutEncoder()
        else:
            enc = ce.LeaveOneOutEncoder(sigma=sigma)
        
        for i_target in range(45):
            if i_target == 0:
                X_train_encoded = enc.fit_transform(X_train, y_train[:,i_target])
                X_val_encoded = enc.transform(X_val)
                X_test_encoded = enc.transform(test_data)
            else:
                X_train_encoded_tmp = enc.fit_transform(X_train, y_train[:,i_target])
                X_train_encoded = np.concatenate([X_train_encoded, X_train_encoded_tmp], axis = 1)
                X_val_encoded_tmp = enc.transform(X_val)
                X_val_encoded = np.concatenate([X_val_encoded, X_val_encoded_tmp], axis = 1)
                X_test_encoded_tmp = enc.transform(test_data)
                X_test_encoded = np.concatenate([X_test_encoded, X_test_encoded_tmp], axis = 1)
        
        train_loader = prepare_dataloader(X_train_encoded, y_train, batch_size=batch_size)
        valid_loader = prepare_dataloader(X_val_encoded, y_val, batch_size=batch_size)
        test_loader = prepare_test_dataloader(X_test_encoded, batch_size=32)
        
        model = ResNetRegression(input_size=X_train_encoded.shape[1], output_size=45, pretrained=False, reshape_size=64, num_channels=16)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        criterion = nn.L1Loss()

        save_path = NN_MODEL_DIR
        val_losses, val_metrics = train_model(
            save_path, model, criterion, optimizer, train_loader, valid_loader, tsvd, num_epochs=num_epochs, patience=patience, fold=fold
        )
        mae_scores.append(val_losses)
        mrrmse_scores.append(val_metrics)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_model(model, os.path.join(save_path, f'{fold}_best_model_checkpoint.pth'), device)
        original_shape = (len(test_loader.dataset), 45)
        predictions = inference(model, tsvd, test_loader, device, original_shape)
        test_preds.append(predictions)

    
    mean_mae = np.mean(mae_scores)
    mean_mrrmse = np.mean(mrrmse_scores)
    average_predictions = np.mean(test_preds, axis=0)

    # Print the results
    print('################')
    print(f'Sigma = {sigma}')
    print(mrrmse_scores)
    print(f'Average MAE across {fold} folds: {mean_mae:.4f}')
    print(f'Average MRRMSE across {fold} folds: {mean_mrrmse:.4f}')
    print('################')
    return average_predictions


def save_submission(sample_submission, preds, submission_path):
    sample_columns = sample_submission.columns
    sample_columns= sample_columns[1:]
    submission_df = pd.DataFrame(preds, columns=sample_columns)
    submission_df.insert(0, 'id', range(255))
    submission_df.to_csv(os.path.join(submission_path, 'nn.csv'), index=False)
        

if __name__ == '__main__':
    seed_torch(42)
    
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    data_path = settings.get("RAW_DATA_DIR")
    NN_MODEL_DIR = settings.get("NN_MODEL_DIR")
    SUBMISSION_DIR = settings.get("SUBMISSION_DIR")
    
    df_de_train, id_map, sample_submission = read_data(data_path)

    average_predictions = MT_cross_validate_model(NN_MODEL_DIR, df_de_train, id_map, batch_size=8, num_epochs=1000, patience=100)
    
    save_submission(sample_submission, average_predictions, SUBMISSION_DIR)