Hello!

Below you can find a outline of how to reproduce my solution for the <[Open Problems – Single-Cell Perturbations](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations)> competition.
We have noticed that even when the seed is fixed, there are variations in the results when running on different devices. Since the servers we rented during the competition have expired, re-running the same code on a new device produces results that differ from the final submission in the competition. The difference now is in around 0.002 in Private LB score.
If you run into any trouble with the setup/code or have any questions please contact me at <paralyzed2023@163.com>

#ARCHIVE CONTENTS
data                     : competition raw data download from kaggle

models                     : trained pyboost model and nn model

submissions                   : predict results on the test dataset by models and open-sourced submissions

prepare_data.py                 : code to train RAPIDS SVR model and get OOF/TESTS features from scratch

train_pyboost.py                  : code to rebuild pyboost model from scratch and get predictions on the test dataset

train_nn.py                    : code to rebuild nn model from scratch and get predictions on the test dataset

predict_pyboost.py                : code to generate predictions from trained pyboost model which should be the same as the predictions generated from train_pyboost.py

predict_nn.py                : code to generate predictions from trained nn model which should be the same as the predictions generated from train_nn.py

ensemble.py                : code to generate submission.csv from above predictions and open-sourced predictions

#HARDWARE: (The following specs were used to create the original solution)
Ubuntu 18.04 (90 GB boot disk)
12 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
1 x RTX 4090

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.8.10
CUDA Version: 12.2

#DATA PROCESSING

Data processing part contains pretrain on the RAPIDS SVR models and generate two versions of SVR OOF/TESTS feas for the Pyboost model training.
Run below will overwrite ./data/processed/train_svr_feas and ./data/processed/test_svr_feas. This precess will in 6 hours to finish.

`python prepare_data.py`

#MODEL BUILD: There are three options to produce the solution.
1) fast prediction on pre-trained models
    a) runs in a few minutes
    b) uses pre-trained Pyboost model and NN model
2) retrain models
    a) expect this to run about 2-3 hours
    b) trains Pyboost model and NN model from scratch

shell command to run each build is below
#1) fast prediction on pre-trained models (overwrites ./submissions/nn.csv, ./submissions/pyboost.csv and ./submissions/submission.csv)
`python predict_pyboost.py`

`python predict_nn.py`

`python ensemble.py`

#2) retrain models (overwrites models and submissions)
`python train_pyboost.py`

`python nn_pyboost.py`

`python ensemble.py`
