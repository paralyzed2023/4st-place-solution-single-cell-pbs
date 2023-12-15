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

def seed_torch(seed=2023):
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


def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def single_model_inference(model, tsvd, test_loader, device, original_shape):
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

    

def predict(NN_MODEL_DIR, de_train, id_map):
    labels = de_train.drop(columns=["cell_type","sm_name","sm_lincs_id","SMILES","control"]).values
    features_columns = ["cell_type", "sm_name"]
    features = pd.DataFrame(de_train, columns=features_columns).values
    test_data = pd.DataFrame(id_map, columns=features_columns).values
    
    # tsvds, encs = load_tools(NN_INFER_TOOLS_DIR)
    test_preds = []
    folds_index_data_MT = get_kfold_from_MT(de_train)
    sigma = 'default'
    for fold, (train_ids, test_ids) in enumerate(folds_index_data_MT):
        print('Fold', fold)
        
        tsvd = TruncatedSVD(n_components=35, n_iter=7, random_state=42)
        y_train = tsvd.fit_transform(labels[train_ids])
        y_val = tsvd.transform(labels[test_ids])
        
        X_train, X_val = features[train_ids], features[test_ids]
 
        if sigma == 'default':
            enc = ce.LeaveOneOutEncoder()
        else:
            enc = ce.LeaveOneOutEncoder(sigma=sigma)

        for i_target in range(35):
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

        test_loader = prepare_test_dataloader(X_test_encoded, batch_size=32)
        
        model = ResNetRegression(input_size=X_test_encoded.shape[1], output_size=35, pretrained=False, reshape_size=64, num_channels=16)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        criterion = nn.L1Loss()

        save_path = NN_MODEL_DIR

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_model(model, os.path.join(save_path, f'{fold}_best_model_checkpoint.pth'), device)
        original_shape = (len(test_loader.dataset), 35)
        predictions = single_model_inference(model, tsvd, test_loader, device, original_shape)
        test_preds.append(predictions)

    average_predictions = np.mean(test_preds, axis=0)

    return average_predictions


def save_submission(sample_submission, preds, submission_path):
    sample_columns = sample_submission.columns
    sample_columns= sample_columns[1:]
    submission_df = pd.DataFrame(preds, columns=sample_columns)
    submission_df.insert(0, 'id', range(255))
    submission_df.to_csv(os.path.join(submission_path, 'nn.csv'), index=False)
        

if __name__ == '__main__':
    seed_torch(2023)

    with open('settings.json', 'r') as file:
        settings = json.load(file)
    data_path = settings.get("RAW_DATA_DIR")
    NN_MODEL_DIR = settings.get("NN_MODEL_DIR")
    SUBMISSION_DIR = settings.get("SUBMISSION_DIR")
    
    df_de_train, id_map, sample_submission = read_data(data_path)

    average_predictions = predict(NN_MODEL_DIR, df_de_train, id_map)

    save_submission(sample_submission, average_predictions, SUBMISSION_DIR)