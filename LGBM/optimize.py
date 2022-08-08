import warnings

warnings.filterwarnings("ignore")
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold

cv_id = 0

text = pd.read_csv("../text.csv")
data = np.load("last_hidden_state.npy")


def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'is_unbalance': True,
        'verbose': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['dart', 'gbdt']),
        'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.2, 1.0),
        'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.2, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.2, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'max_depth': trial.suggest_int("max_depth", 2, 10),
        'n_estimators': trial.suggest_int("n_estimators", 100, 400),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.0001, 1),
        "cat_smooth": trial.suggest_int('cat_smooth', 1, 100),
        "max_bin": trial.suggest_int('max_bin', 10, 300),
        'random_state': 0,
        'n_jobs': 16
    }

    kf = StratifiedGroupKFold(n_splits=2, shuffle=True)
    log_loss = []
    for train_index, test_index in kf.split(data, y=text['discourse_effectiveness'], groups=text['essay_id']):
        X_train_K, X_test_K = data[train_index], data[test_index]
        y_train_K, y_test_K = text["discourse_effectiveness"].values[train_index], \
                              text["discourse_effectiveness"].values[test_index]
        train_set = lgb.Dataset(X_train_K, label=y_train_K)
        model = lgb.train(params, train_set)
        log_loss.append(metrics.log_loss(y_test_K, model.predict(X_test_K)))
        print(metrics.log_loss(y_train_K, model.predict(X_train_K)))
    return np.mean(log_loss)


study = optuna.create_study(direction='minimize', sampler=TPESampler())
study.optimize(objective, n_trials=200)
study.trials_dataframe().to_csv("result.csv")
