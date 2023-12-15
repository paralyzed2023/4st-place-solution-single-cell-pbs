import numpy as np 
import pandas as pd 
import os, gc, re, warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import random
from transformers import AutoModel,AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from cuml.svm import SVR
import cuml
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
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
    return df_de_train, id_map

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


def generate_feas(df_de_train, id_map):
    all_train_text_feats, te_text_feats = get_embeddings(df_de_train)
    
    de_cell_type = df_de_train.iloc[:, [0] + list(range(5, df_de_train.shape[1]))]
    de_sm_name = df_de_train.iloc[:, [1] + list(range(5, df_de_train.shape[1]))]
    
    fearures_columns = ['cell_type', 'sm_name']
    features = pd.DataFrame(df_de_train, columns=fearures_columns)
    test_data = pd.DataFrame(id_map, columns=fearures_columns)
    
    encoder = OneHotEncoder()
    encoder.fit(features)
    one_hot_encode_features = encoder.transform(features)
    one_hot_test = encoder.transform(test_data)
    
    all_train_text_feats = np.concatenate([all_train_text_feats, one_hot_encode_features.toarray()],axis=1)
    te_text_feats = np.concatenate([te_text_feats, one_hot_test.toarray()],axis=1)
    
    print('Our train concatenated embeddings have shape', all_train_text_feats.shape )
    print('Our test concatenated embeddings have shape', te_text_feats.shape )
    
    return all_train_text_feats, te_text_feats


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
    return res

# 将添加特征的函数进行修改以适用于新结构
def add_agg_feas(base_feats, agg_feats_dict, target_col, selected_idx=None):
    base_feats = pd.DataFrame(base_feats)
    
    if selected_idx is not None:
        for func, data in agg_feats_dict.items():
            cell_type_key = f'{func}_cell_type'
            sm_name_key = f'{func}_sm_name'
            base_feats = base_feats.join(data['tr_cell_type'][target_col].iloc[selected_idx].reset_index(drop=True), rsuffix=f'_{cell_type_key}')
            base_feats = base_feats.join(data['tr_sm_name'][target_col].iloc[selected_idx].reset_index(drop=True), rsuffix=f'_{sm_name_key}')
    else:
        for func, data in agg_feats_dict.items():
            cell_type_key = f'{func}_cell_type'
            sm_name_key = f'{func}_sm_name'
            base_feats = base_feats.join(data['te_cell_type'][target_col], rsuffix=f'_{cell_type_key}')
            base_feats = base_feats.join(data['te_sm_name'][target_col], rsuffix=f'_{sm_name_key}')
    return base_feats.values

# 一个函数来执行交叉验证过程
def cross_validate(df_de_train, id_map, all_train_text_feats, te_text_feats, agg_funcs = ['mean']):
    target_cols = df_de_train.columns.to_list()[5:]
    de_cell_type = df_de_train.iloc[:, [0] + list(range(5, df_de_train.shape[1]))]
    de_sm_name = df_de_train.iloc[:, [1] + list(range(5, df_de_train.shape[1]))]
    agg_feas = {}
    
    # 计算所有聚合特征
    for func in agg_funcs:
        agg_feas[func] = {
            'tr_cell_type': compute_grouped_stats(de_cell_type, de_cell_type, 'cell_type', func),
            'tr_sm_name': compute_grouped_stats(de_sm_name, de_sm_name, 'sm_name', func),
            'te_cell_type': compute_grouped_stats(de_cell_type, id_map, 'cell_type', func),
            'te_sm_name': compute_grouped_stats(de_sm_name, id_map, 'sm_name', func)
        }

    te_preds = []
    ev_preds = []
    scores = []
    
    FOLDS = 5
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    for i,(train_index, val_index) in enumerate(kf.split(df_de_train)):
        df_de_train.loc[val_index,'FOLD'] = i
    print('Train samples per fold:')
    print(df_de_train.FOLD.value_counts())
    
    oof_preds = np.zeros((df_de_train.shape[0], len(target_cols)))  # 初始化oof预测矩阵
    test_preds = np.zeros((te_text_feats.shape[0], len(target_cols)))  # 初始化测试集预测矩阵
    
    for fold in range(FOLDS):
        
        tr_idx = df_de_train[df_de_train["FOLD"] != fold].index
        ev_idx = df_de_train[df_de_train["FOLD"] == fold].index
        
        tr_text_feats = all_train_text_feats[list(tr_idx), :]
        ev_text_feats = all_train_text_feats[list(ev_idx), :]
        
        # test_preds_old = np.zeros((len(te_text_feats),len(target_cols)))
        for i, t in enumerate(tqdm(target_cols)):
        
            tr_features = tr_text_feats
            ev_features = ev_text_feats
            te_features = te_text_feats
            
            tr_features = add_agg_feas(tr_text_feats, agg_feas, t, tr_idx)
            ev_features = add_agg_feas(ev_text_feats, agg_feas, t, ev_idx)
            te_features = add_agg_feas(te_text_feats, agg_feas, t)
            
            # 训练模型
            clf = SVR(C=1)
            clf.fit(tr_features, df_de_train.loc[tr_idx, t])
            # clf.fit(tr_features, tr_idx[t].values)
            
            # 做预测
            fold_ev_preds = clf.predict(ev_features)
            # print(fold_ev_preds)
            fold_te_preds = clf.predict(te_features)
            oof_preds[ev_idx, target_cols.index(t)] = fold_ev_preds
            test_preds[:, target_cols.index(t)] += fold_te_preds / FOLDS
            
            # 在这里累积验证集预测结果
            if t == target_cols[0]:
                fold_ev_preds_tmp = fold_ev_preds
            else:
                fold_ev_preds_tmp = np.column_stack((fold_ev_preds_tmp, fold_ev_preds))
            
            # test_preds_old[:,i] = clf.predict(te_features)
            
        # 计算并打印这一折的分数
        score = mean_absolute_error(df_de_train.loc[ev_idx, target_cols].values, fold_ev_preds_tmp)
        scores.append(score)
        print("Fold : {} MAE score: {}".format(fold, score))
        te_preds.append(fold_te_preds)
        ev_preds.append(fold_ev_preds)

    print('Overall CV MAE =', np.mean(scores))
    overall_oof_score = mean_absolute_error(df_de_train[target_cols].values, oof_preds)
    print('Overall OOF MAE =', overall_oof_score)
    return oof_preds, test_preds

def save_svr_feas(test_preds_name, oof_preds_name, test_preds, oof_preds):
    np.save(test_preds_name, test_preds)
    np.save(oof_preds_name, oof_preds)
    print("Successful saved svr feas")


if __name__ == '__main__':
    set_seed(42)
    
    with open('settings.json', 'r') as file:
        settings = json.load(file)
    data_path = settings.get("RAW_DATA_DIR")
    TRAIN_SVR_FEAS_DIR = settings.get("TRAIN_SVR_FEAS_DIR")
    TEST_SVR_FEAS_DIR = settings.get("TEST_SVR_FEAS_DIR")
    
    df_de_train, id_map = read_data(data_path)
    all_train_text_feats, te_text_feats = generate_feas(df_de_train, id_map)
    agg_funcs1 = ['mean', 'min', 'max', 'median', 'first', 'quantile_0.4']
    agg_funcs2 = []
    oof_preds1, test_preds1 = cross_validate(df_de_train, id_map, all_train_text_feats, te_text_feats, agg_funcs1)
    save_svr_feas(os.path.join(TRAIN_SVR_FEAS_DIR, 'test_preds_svr_v1.npy'), os.path.join(TEST_SVR_FEAS_DIR, 'oof_preds_svr_v1.npy'), test_preds1, oof_preds1)
    oof_preds2, test_preds2 = cross_validate(df_de_train, id_map, all_train_text_feats, te_text_feats, agg_funcs2)
    save_svr_feas(os.path.join(TRAIN_SVR_FEAS_DIR, 'test_preds_svr_v2.npy'), os.path.join(TEST_SVR_FEAS_DIR, 'oof_preds_svr_v2.npy'), test_preds2, oof_preds2)
    