import os
import pandas as pd
import numpy as np
import json

with open('settings.json', 'r') as file:
    settings = json.load(file)
SUBMISSION_DIR = settings.get("SUBMISSION_DIR")


df_pyboost = pd.read_csv(os.path.join(SUBMISSION_DIR, 'pyboost.csv'), index_col='id')
df_nn = pd.read_csv(os.path.join(SUBMISSION_DIR, 'nn.csv'), index_col='id')
open_sourced_0531 = pd.read_csv(os.path.join(SUBMISSION_DIR, 'open-sourced-0531.csv'), index_col='id')
open_sourced_0720 = pd.read_csv(os.path.join(SUBMISSION_DIR, 'open-sourced-0720.csv'), index_col='id')

       
submission = ((df_pyboost*0.3 + df_nn*0.7)*0.9 + open_sourced_0720*0.1)*0.95 + open_sourced_0531*0.05
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'))