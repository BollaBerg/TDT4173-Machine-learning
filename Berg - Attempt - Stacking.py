# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %%
# !pip install catboost


import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor

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
    "rooms": "category", # "uint8",
    "bathrooms_shared": "category", # "uint8",
    "bathrooms_private": "category", # "uint8",
    "windows_court": CategoricalDtype(categories=[0, 1, 2]),
    "windows_street": CategoricalDtype(categories=[0, 1, 2]),
    "balconies": "category", # "uint8",
    "loggias": "category", # "uint8",
    "condition": CategoricalDtype(categories=[0, 1, 2, 3, 4]),
    "phones": "category", # "uint8",
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
    "bathroom_amount": "category",
    "cluster": "category",
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
    return root_mean_squared_log_error(np.expm1(y_true), np.expm1(predictions))

if LOG_TARGET:
    evaluate_predictions = evaluate_logged_predictions

RMSLE_scorer = make_scorer(evaluate_predictions, greater_is_better=False)

def lightgbm_feval(y_true, y_pred):
    return "RMSLE", evaluate_predictions(y_pred, y_true), False

if LOG_TARGET:
    y_train = np.log1p(y_train)
    y_valid = np.log1p(y_valid)
if LOG_AREA:
    X_train["area_total"] = np.log1p(X_train["area_total"])
    X_train["area_kitchen"] = np.log1p(X_train["area_kitchen"])
    X_train["area_living"] = np.log1p(X_train["area_living"])
    X_valid["area_total"] = np.log1p(X_valid["area_total"])
    X_valid["area_kitchen"] = np.log1p(X_valid["area_kitchen"])
    X_valid["area_living"] = np.log1p(X_valid["area_living"])


if USE_POLAR_COORDINATES:
    X_train.drop(["latitude", "longitude"], axis=1, inplace=True)
    X_valid.drop(["latitude", "longitude"], axis=1, inplace=True)
else:
    X_train.drop(["distance_from_center", "angle"], axis=1, inplace=True)
    X_valid.drop(["distance_from_center", "angle"], axis=1, inplace=True)

categorical_columns = [
    "seller",
    "rooms",
    "bathrooms_shared",
    "bathrooms_private",
    "windows_court",
    "windows_street",
    "balconies",
    "loggias",
    "condition",
    "phones",
    "new",
    "district",
    "material",
    "elevator_without",
    "elevator_passenger",
    "elevator_service",
    "parking",
    "garbage_chute",
    "heating",
    "bathroom_amount",
    "cluster",
]
categorical_columns_indices = [X_train.columns.get_loc(c) for c in categorical_columns if c in X_train]

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
    'metric': ["l1", 'l2'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbosity': 0,
    "max_depth": 6,
    "num_leaves": 60,  
    "max_bin": 256,
    "num_iterations": 100000,
    "categorical_column": categorical_columns_indices
}
lightgbm = lgb.LGBMRegressor(**lightgbm_hyper_params)

# Fit all first-level models
print("Fitting CatBoost")
catboost.fit(X_train, y_train,
    eval_set=valid_set,
    use_best_model=True
)
print("Fitting LightGBM")
callbacks = [lgb.early_stopping(10, verbose=0), lgb.log_evaluation(period=0)]
lightgbm.fit(X_train, y_train, 
    early_stopping_rounds=1000,
    eval_set=[(X_valid, y_valid)],
    eval_metric=lightgbm_feval,
    callbacks=callbacks             # Mute LightGBM
)

train_pred = pd.DataFrame({
    "catboost": catboost.predict(X_train),
    "lightgbm": lightgbm.predict(X_train),
    "area_total": X_train["area_total"],
    "rooms": X_train["rooms"],
    "distance_from_center": X_train["distance_from_center"],
    "angle": X_train["angle"]
})

# Create stack
print("Fitting stack")
stack = RandomForestRegressor()
stack.fit(train_pred, y_train)

# Predict validation stuff
print("Predicting validation stuff")
catboost_pred = catboost.predict(X_valid)
lightgbm_pred = lightgbm.predict(X_valid)
valid_pred = pd.DataFrame({
    "catboost": catboost_pred,
    "lightgbm": lightgbm_pred,
    "area_total": X_valid["area_total"],
    "rooms": X_valid["rooms"],
    "distance_from_center": X_valid["distance_from_center"],
    "angle": X_valid["angle"]
})
stack_pred = stack.predict(valid_pred)


print(f"CatBoost score on valid set: {evaluate_predictions(catboost_pred, y_valid)}")
print(f"LightGBM score on valid set: {evaluate_predictions(lightgbm_pred, y_valid)}")
print(f"Stack score on valid set: {evaluate_predictions(stack_pred, y_valid)}")


print("Starting test stuff")
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
submission_catboost = pd.DataFrame()
submission_lightgbm = pd.DataFrame()
submission['id'] = data_test.index
submission_catboost['id'] = data_test.index
submission_lightgbm['id'] = data_test.index

if LOG_AREA:
    data_test["area_total"] = np.log1p(data_test["area_total"])
    data_test["area_kitchen"] = np.log1p(data_test["area_kitchen"])
    data_test["area_living"] = np.log1p(data_test["area_living"])


# Predict test stuff
print("Predict test stuff")
catboost_test = catboost.predict(data_test)
lightgbm_test = lightgbm.predict(data_test)
test_pred = pd.DataFrame({
    "catboost": catboost_test,
    "lightgbm": lightgbm_test,
    "area_total": data_test["area_total"],
    "rooms": data_test["rooms"],
    "distance_from_center": data_test["distance_from_center"],
    "angle": data_test["angle"]
})
stack_test = stack.predict(test_pred)

if LOG_TARGET:
    stack_test = np.expm1(stack_test)
    catboost_test = np.expm1(catboost_test)
    lightgbm_test = np.expm1(lightgbm_test)

submission["price_prediction"] = stack_test
submission_catboost["price_prediction"] = catboost_test
submission_lightgbm["price_prediction"] = lightgbm_test

print("Saving CSVs")
savepath = 'submissions/stacked_submission.csv'
savepath_catboost = "submissions/stacked_catboost_submission.csv"
savepath_lightgmb = "submissions/stacked_lightgmb_submission.csv"
submission.to_csv(savepath, index=False)
submission_catboost.to_csv(savepath_catboost, index=False)
submission_lightgbm.to_csv(savepath_lightgmb, index=False)
print(f"Training done! Submission saved to '{savepath}', '{savepath_catboost}' and '{savepath_lightgmb}'")
