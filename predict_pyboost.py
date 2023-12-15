import joblib
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
# simple case - just one class is used
from py_boost import GradientBoosting, TLPredictor, TLCompiledPredictor
from py_boost.cv import CrossValidation
import os
import random
import category_encoders as ce
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from transformers import AutoModel,AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import json


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state.detach().cpu()
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        text = self.df.loc[idx,"SMILES"]
        tokens = tokenizer(
                text,
                None,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=150,
                return_tensors="pt")
        tokens = {k:v.squeeze(0) for k,v in tokens.items()}
        return tokens
    

def get_embeddings(de_train, MODEL_NM='DeepChem/ChemBERTa-10M-MTR', MAX_LEN=150, BATCH_SIZE=32, verbose=True):
    global tokenizer
    # Extract unique texts
    unique_texts = de_train["SMILES"].unique()

    # Create a dataset for unique texts
    ds_unique = EmbedDataset(pd.DataFrame(unique_texts, columns=["SMILES"]))
    embed_dataloader_unique = torch.utils.data.DataLoader(ds_unique, batch_size=BATCH_SIZE, shuffle=False)

    DEVICE="cuda"
    model = AutoModel.from_pretrained( MODEL_NM )
    tokenizer = AutoTokenizer.from_pretrained( MODEL_NM )
    
    model = model.to(DEVICE)
    model.eval()
    unique_emb = []
    for batch in tqdm(embed_dataloader_unique,total=len(embed_dataloader_unique)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
        unique_emb.extend(sentence_embeddings)
    unique_emb = np.array(unique_emb)
    if verbose:
        print('unique embeddings shape',unique_emb.shape)
        
    text_to_embedding = {text: emb for text, emb in zip(unique_texts, unique_emb)}

    train_emb = np.array([text_to_embedding[text] for text in de_train['SMILES']])
    test_emb = np.array([text_to_embedding[text] for text in id_map["SMILES"]])
        
    return train_emb, test_emb


def load_models(model_path):
    loaded_models = []
    for i in range(1, 4): 
        model = joblib.load(os.path.join(model_path, f'model_{i}.joblib'))
        loaded_models.append(model)
    return loaded_models

def compute_grouped_stats(all_df, target_df, group_col, agg_func):
    if 'quantile' in agg_func:
        percent = float(agg_func.split('_')[1])
        agg_ = all_df.groupby(group_col).quantile(percent).reset_index()
    else:
        agg_ = all_df.groupby(group_col).agg(agg_func).reset_index()
    rows = []
    for name in target_df[group_col]:
        rows.append(agg_[agg_[group_col] == name].copy())
        
    res = pd.concat(rows)
    res = res.reset_index(drop=True)
    res = res.drop(group_col, axis=1)
    # display(res)
    return res.values

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

def predict(PYBOOST_MODEL_DIR, de_train, id_map, test_preds_svr_with_emb_v1, oof_preds_svr_with_emb_v1, test_preds_svr_with_emb_v2, oof_preds_svr_with_emb_v2, te_chem_emb):
    loaded_models = load_models(PYBOOST_MODEL_DIR)
    print(len(loaded_models))
        
    de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
    de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]
    sigma = 0.21
    agg_feas = {}
    agg_funcs = ['mean', 'max']
    
    for func in agg_funcs:
        agg_feas[func] = {
            'tr_cell_type': compute_grouped_stats(de_cell_type, de_cell_type, 'cell_type', func),
            'tr_sm_name': compute_grouped_stats(de_sm_name, de_sm_name, 'sm_name', func),
            'te_cell_type': compute_grouped_stats(de_cell_type, id_map, 'cell_type', func),
            'te_sm_name': compute_grouped_stats(de_sm_name, id_map, 'sm_name', func)
        }
    
    labels = de_train.drop(columns=["cell_type","sm_name","sm_lincs_id","SMILES","control"]).values
    features_columns = ["cell_type", "sm_name"]
    features = pd.DataFrame(de_train, columns=features_columns).values
    test_data = pd.DataFrame(id_map, columns=features_columns).values
    
    test_preds = []
    folds_index_data_MT = get_kfold_from_MT(de_train)
    for fold, (train_ids, test_ids) in enumerate(folds_index_data_MT):
        print('Fold', fold)
 
        tsvd = TruncatedSVD(n_components=52, n_iter=19, random_state=42)
        y_train = tsvd.fit_transform(labels[train_ids])
        y_val = tsvd.transform(labels[test_ids])

        X_train, X_val = features[train_ids], features[test_ids]

        if sigma == 'default':
            enc = ce.LeaveOneOutEncoder()
        else:
            enc = ce.LeaveOneOutEncoder(sigma=sigma)

        for i_target in range(52):
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
        
        tr_svr_v1 = oof_preds_svr_with_emb_v1[train_ids]
        va_svr_v1 = oof_preds_svr_with_emb_v1[test_ids]
        tr_svr_v2 = oof_preds_svr_with_emb_v2[train_ids]
        va_svr_v2 = oof_preds_svr_with_emb_v2[test_ids]
        tr_chem_emb = all_chem_emb[train_ids]
        va_chem_emb = all_chem_emb[test_ids]
        
        tr_svr = np.concatenate([tr_svr_v1, tr_svr_v2, tr_chem_emb],axis=1)
        te_svr = np.concatenate([test_preds_svr_with_emb_v1, test_preds_svr_with_emb_v2, te_chem_emb],axis=1)
        tsvd_svr = TruncatedSVD(n_components=35, n_iter=7, random_state=42)
        tr_svr_tsvd = tsvd_svr.fit_transform(tr_svr)
        te_svr_tsvd = tsvd_svr.transform(te_svr)
        
        te_data = X_test_encoded
        te_data = np.concatenate([te_data, te_svr_tsvd],axis=1)
        
        for func in agg_funcs:
            tr_cell_type = agg_feas[func]['tr_cell_type'][train_ids]
            te_cell_type = agg_feas[func]['te_cell_type']
            tsvd_cell_type = TruncatedSVD(n_components=35, n_iter=7, random_state=42)
            tr_cell_type_tsvd = tsvd_cell_type.fit_transform(tr_cell_type)
            te_cell_type_tsvd = tsvd_cell_type.transform(te_cell_type)

            tr_sm_name = agg_feas[func]['tr_sm_name'][train_ids]
            te_sm_name = agg_feas[func]['te_sm_name']
            tsvd_sm_name = TruncatedSVD(n_components=35, n_iter=7, random_state=42)
            tr_sm_name_tsvd = tsvd_sm_name.fit_transform(tr_sm_name)
            te_sm_name_tsvd = tsvd_sm_name.transform(te_sm_name)
            
        
        for func in agg_funcs:
            te_data = np.concatenate([te_data, te_cell_type_tsvd],axis=1)
            te_data = np.concatenate([te_data, te_sm_name_tsvd],axis=1)
        
        print("Start predicting...")
        model = loaded_models[fold]
        te_data_pred = model.predict(te_data)
        print("Predict finished")
        te_data_pred = tsvd.inverse_transform(te_data_pred)
        
        test_preds.append(te_data_pred)

    average_predictions = np.mean(test_preds, axis=0)

    return average_predictions


def save_submission(sample_submission, preds, submission_path):
    sample_columns = sample_submission.columns
    sample_columns= sample_columns[1:]
    submission_df = pd.DataFrame(preds, columns=sample_columns)
    submission_df.insert(0, 'id', range(255))
    submission_df.to_csv(os.path.join(submission_path, 'pyboost.csv'), index=False)
        
if __name__ == '__main__':
    set_seed(42)
    
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    data_path = settings.get("RAW_DATA_DIR")
    TRAIN_SVR_FEAS_DIR = settings.get("TRAIN_SVR_FEAS_DIR")
    TEST_SVR_FEAS_DIR = settings.get("TEST_SVR_FEAS_DIR")
    PYBOOST_MODEL_DIR = settings.get("PYBOOST_MODEL_DIR")
    SUBMISSION_DIR = settings.get("SUBMISSION_DIR")
    
    df_de_train, id_map, sample_submission = read_data(data_path)
    all_chem_emb, te_chem_emb = get_embeddings(df_de_train)
    test_preds_svr_with_emb_v1 = np.load(os.path.join(TEST_SVR_FEAS_DIR, 'test_preds_svr_v1.npy'))
    oof_preds_svr_with_emb_v1 = np.load(os.path.join(TRAIN_SVR_FEAS_DIR, 'oof_preds_svr_v1.npy'))
    test_preds_svr_with_emb_v2 = np.load(os.path.join(TEST_SVR_FEAS_DIR, 'test_preds_svr_v2.npy'))
    oof_preds_svr_with_emb_v2 = np.load(os.path.join(TRAIN_SVR_FEAS_DIR, 'oof_preds_svr_v2.npy'))
    
    average_predictions = predict(PYBOOST_MODEL_DIR, df_de_train, id_map, test_preds_svr_with_emb_v1, oof_preds_svr_with_emb_v1, test_preds_svr_with_emb_v2, oof_preds_svr_with_emb_v2, te_chem_emb)
    save_submission(sample_submission, average_predictions, SUBMISSION_DIR)

    