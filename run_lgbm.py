import pandas as pd
from nyaggle.experiment import run_experiment
from nyaggle.feature_store import load_features, load_feature
from sklearn.metrics import average_precision_score

from src.utils import prauc, get_folds

submission = pd.read_csv("input/atmaCup5__sample_submission.csv")
all_df = load_feature("all", "working")

data = load_features(
    all_df,
    feature_names=[
        "fitting",
        "peak_around",
        "intensity_stats",
        "savgol_peak",
        "spec_percentile",
        "fitting_combination",
    ],
    ignore_columns=["spectrum_id", "spectrum_filename", "chip_id"],
)

train = data[data.target.notnull()].copy()
test = data[data.target.isnull()].copy()

target_col = "target"
drop_cols = ["spectrum_id", "spectrum_filename", "chip_id"]
X_train = train.drop(drop_cols + [target_col], axis=1)
y_train = train[target_col]
X_test = test.drop(drop_cols + [target_col], axis=1)

params = {
    "n_estimators": 10000,
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "custom",
    "num_leaves": 300,
    "subsample": 1,
    "colsample_bytree": 0.4,
    "learning_rate": 0.05,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 0,
    "min_data_in_leaf": 40,
}

fit_params = {"early_stopping_rounds": 500, "verbose": 100, "eval_metric": prauc}

result = run_experiment(
    params,
    X_train,
    y_train,
    X_test,
    eval_func=average_precision_score,
    fit_params=fit_params,
    cv=get_folds(train, mode="stratified", random_state=0),
    sample_submission=submission,
    with_mlflow=True,
)
