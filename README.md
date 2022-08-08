# Feedback Prize - Predicting Effective Arguments (Kaggle)

- Initialize [Kaggle API](https://github.com/Kaggle/kaggle-api#api-credentials)
- Run `sh download.sh` to download dataset and preprocess

## Transformer based models

- Demonstration of Adversarial Training (use Fast Gradient Method)
- Demonstration
  of [Masked Pseudo Labeling](https://towardsdatascience.com/pseudo-labeling-to-deal-with-small-datasets-what-why-how-fd6f903213af)

## [GBDT](https://github.com/microsoft/LightGBM) based models

- Demonstration of feature extraction using pretrained transformer based models
- Demonstration of parameter searching using [Optuna](https://optuna.org/)

## Useful attempts

- Unbalanced labels
    - StratifiedGroupKFold
    - Focal Loss
- Overfitting
    - Larger models
    - Ensembling
    - Dropout
    - CosineAnnealingWarmRestarts
    - Regularization
    - Adversarial Training
    - Masked Pseudo Labeling
    - Different lr for different layers