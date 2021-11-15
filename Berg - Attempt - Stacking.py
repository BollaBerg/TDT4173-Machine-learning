# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %%
# !pip install catboost


import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.ensemble import StackingRegressor, RandomForestRegressor

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

pd.set_option('display.max_rows', None)


LOG_TARGET = True
LOG_AREA = True
REMOVE_OUTLIERS = True
USE_POLAR_COORDINATES = True

print(f"Running with variables: LOG_TARGET={LOG_TARGET}, LOG_AREA={LOG_AREA}, REMOVE_OUTLIERS={REMOVE_OUTLIERS},",
      f"USE_POLAR_COORDINATES={USE_POLAR_COORDINATES}"    
)

def read_file(url):
  url = url + "?raw=true"
  df = pd.read_csv(url, index_col="id")
  return df

# url = "https://github.com/andbren/TDT-4173/blob/main/premade/X_train.csv"
# X_train = read_file(url)
# url = "https://github.com/andbren/TDT-4173/blob/main/premade/y_train.csv"
# y_train = read_file(url)

# url = "https://github.com/andbren/TDT-4173/blob/main/premade/X_valid.csv"
# X_valid = read_file(url)
# url = "https://github.com/andbren/TDT-4173/blob/main/premade/y_valid.csv"
# y_valid = read_file(url)

data_train = pd.read_csv("data/preprocessed/data_train.csv", index_col="id")
data_valid = pd.read_csv("data/preprocessed/data_valid.csv", index_col="id")
X_train = data_train.drop("price", axis=1)
X_valid = data_valid.drop("price", axis=1)
y_train = data_train["price"]
y_valid = data_valid["price"]

needed_dtypes = {
    "seller": CategoricalDtype(categories=[0, 1, 2, 3, 4]),
    "floor": "uint8",
    "rooms": "uint8",
    "bathrooms_shared": "uint8",
    "bathrooms_private": "uint8",
    "windows_court": CategoricalDtype(categories=[0, 1, 2]),
    "windows_street": CategoricalDtype(categories=[0, 1, 2]),
    "balconies": "uint8",
    "loggias": "uint8",
    "condition": CategoricalDtype(categories=[0, 1, 2, 3, 4]),
    "phones": "uint8",
    "new": CategoricalDtype(categories=[0, 1, 2]),
    "district": CategoricalDtype(categories=list(range(13))),
    "constructed": "uint16",
    "material": CategoricalDtype(categories=list(range(8))),
    "stories": "uint8",
    "elevator_without": "bool",
    "elevator_passenger": "bool",
    "elevator_service": "bool",
    "parking": CategoricalDtype(categories=[0, 1, 2, 3]),
    "garbage_chute": "bool",
    "heating": CategoricalDtype(categories=[0, 1, 2, 3, 4]),
}
X_train = X_train.astype(needed_dtypes)
X_valid = X_valid.astype(needed_dtypes)

def root_mean_squared_log_error(y_true, y_pred):
    # Alternatively: sklearn.metrics.mean_squared_log_error(y_true, y_pred) ** 0.5
    # assert (y_true >= 0).all() 
    # assert (y_pred >= 0).all()
    y_true = np.where(y_true < 0, 0, y_true)
    y_pred = np.where(y_pred < 0, 0, y_pred)
    log_error = np.log1p(y_pred) - np.log1p(y_true)  # Note: log1p(x) = log(1 + x)
    return np.mean(log_error ** 2) ** 0.5

def evaluate_predictions(predictions: pd.DataFrame, y_true: pd.DataFrame):
    """Evaluate predictions, the same way as done when uploading to Kaggle.

    Args:
      predictions: pandas DataFrame with predictions. Should be in the same
        order as the True data.
    
    Example:
      >>> # model = a previously trained model
      >>> results = model.predict(X_valid)
      >>> score = evaluate_predictions(results, y_valid)
    """
    return root_mean_squared_log_error(y_true, predictions)

def evaluate_logged_predictions(predictions: pd.DataFrame, y_true: pd.DataFrame):
    """Evaluate predictions, if the results are logged. This ensures comparitive
    scores when LOG_TARGET = True and LOG_TARGET = False.

    Should only be used when LOG_TARGET = True

    Args:
      predictions: pandas DataFrame with predictions. Should be in the same
        order as the True data.
    
    Example:
      >>> # model = a previously trained model
      >>> results = model.predict(X_valid)
      >>> score = evaluate_predictions(results, y_valid)
    """
    return root_mean_squared_log_error(10 ** y_true, 10 ** predictions)

if LOG_TARGET:
    RMSLE_scorer = make_scorer(evaluate_logged_predictions, greater_is_better=False)
else:
    RMSLE_scorer = make_scorer(evaluate_predictions, greater_is_better=False)


if LOG_TARGET:
    y_train = np.log10(y_train)
    y_valid = np.log10(y_valid)
if LOG_AREA:
    X_train["area_total"] = np.log10(X_train["area_total"] + 1)
    X_train["area_kitchen"] = np.log10(X_train["area_kitchen"] + 1)
    X_train["area_living"] = np.log10(X_train["area_living"] + 1)
    X_valid["area_total"] = np.log10(X_valid["area_total"] + 1)
    X_valid["area_kitchen"] = np.log10(X_valid["area_kitchen"] + 1)
    X_valid["area_living"] = np.log10(X_valid["area_living"] + 1)


if USE_POLAR_COORDINATES:
    X_train.drop(["latitude", "longitude"], axis=1, inplace=True)
    X_valid.drop(["latitude", "longitude"], axis=1, inplace=True)
else:
    X_train.drop(["distance_from_center", "angle"], axis=1, inplace=True)
    X_valid.drop(["distance_from_center", "angle"], axis=1, inplace=True)


categorical_columns = list(X_train.select_dtypes(include=["category", "bool"]).columns)
categorical_columns_indices = [X_train.columns.get_loc(c) for c in categorical_columns if c in X_train]

X_combined = X_train.append(X_valid)
y_combined = y_train.append(y_valid)
train_indeces = X_train.index
valid_indeces = X_valid.index

### CATBOOST ###
valid_set = Pool(
    X_valid,
    label = y_valid,
    cat_features = categorical_columns
)


catboost = CatBoostRegressor(
    cat_features = categorical_columns,
    logging_level="Silent",
    custom_metric="MSLE",
    eval_metric="MSLE",
    # task_type="GPU",
    border_count=254,
    ctr_target_border_count=2,
    depth=10,
    iterations=1500,
    l2_leaf_reg=1,
    one_hot_max_size=8,
)

### LIGHTGBM ###
lightgbm_hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbosity': -1,
    "max_depth": 8,
    "max_leaves": 256,  
    "max_bin": 512,
    "eval_set": [(X_valid, y_valid)],
    "num_iterations": 100000,
    "eval_metric": 'l1',
    # "early_stopping_rounds": 100,
    "categorical_column": categorical_columns_indices
}
lightgbm = lgb.LGBMRegressor(**lightgbm_hyper_params)

stack = StackingRegressor(
    estimators = [
        ("CatBoost", catboost),
        ("LightGBM", lightgbm)
    ],
    final_estimator=RandomForestRegressor(),
    passthrough=True
)

stack.fit(X_train, y_train)
predictions = stack.predict(X_valid)

if LOG_TARGET:
    predictions = 10 ** predictions
    y_valid = 10 ** y_valid

print(f"Number of negative predictions: {sum((1 for pred in predictions if pred < 0))}")

zeroed_predictions = np.where(predictions < 0, 0, predictions)
print(f"Score: {evaluate_predictions(zeroed_predictions, y_valid)}")


# url = "https://github.com/andbren/TDT-4173/blob/main/premade/test.csv"
# data_test = read_file(url)
data_test = pd.read_csv("data/preprocessed/test.csv", index_col="id")

data_test = data_test.astype(needed_dtypes)

if USE_POLAR_COORDINATES:
    data_test.drop(["latitude", "longitude"], axis=1, inplace=True)
    data_test.fillna(
        value = {
            "distance_from_center": data_test["distance_from_center"].median(skipna=True),
            "angle": data_test["angle"].median(skipna=True)
        },
        inplace=True
    )
else:
    data_test.drop(["distance_from_center", "angle"], axis=1, inplace=True)
    data_test.fillna(
        value = {
            "latitude": np.median(data_test["latitude"]),
            "longitude": np.median(data_test["longitude"])
        },
        inplace=True
    )

submission = pd.DataFrame()
submission['id'] = data_test.index

if LOG_AREA:
    data_test["area_total"] = np.log10(data_test["area_total"] + 1)
    data_test["area_kitchen"] = np.log10(data_test["area_kitchen"] + 1)
    data_test["area_living"] = np.log10(data_test["area_living"] + 1)

predictions_test = stack.predict(data_test)

if LOG_TARGET:
    predictions_test = 10 ** predictions_test

submission["price_prediction"] = predictions_test

savepath = 'stacked_submission.csv'
submission.to_csv(savepath, index=False)
print(f"Training done! Submission saved to '{savepath}'")
