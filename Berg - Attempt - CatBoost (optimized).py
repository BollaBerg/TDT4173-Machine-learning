# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %%
# !pip install catboost
# !pip install scikit-optimize


# %%
from datetime import datetime
from lightgbm.engine import train
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from sklearn.metrics import make_scorer

from catboost import CatBoostRegressor, Pool

from skopt import BayesSearchCV

pd.set_option('display.max_rows', None)


LOG_TARGET = True
SEARCH_PARAMETERS = True
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


valid_set = Pool(
    X_valid,
    label = y_valid,
    cat_features = categorical_columns
)

if SEARCH_PARAMETERS:
    model = CatBoostRegressor(
        cat_features = categorical_columns,
        logging_level="Silent",
        custom_metric="MSLE",
        eval_metric="MSLE",
        # task_type="GPU",
    )
else:
    model = CatBoostRegressor(
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
    model.fit(X_train, y_train, eval_set=valid_set, use_best_model=True)


X_gridsearch = X_train.append(X_valid)
y_gridsearch = y_train.append(y_valid)
train_indeces = X_train.index
valid_indeces = X_valid.index


if SEARCH_PARAMETERS:
    search_grid = {
        "iterations" : [500, 1000, 2000],
        "depth" : [4, 6, 8, 10, 16],
        "l2_leaf_reg" : [1, 2, 8, 16, 32],
        "border_count" : [16, 64, 128, 254],
        "ctr_target_border_count" : [1, 2, 8, 32, 128, 255],
        "one_hot_max_size" : [2, 8, 16],
    }
    grid_combinations = 1
    for grid_row in search_grid.values():
        grid_combinations *= len(grid_row)
    print(f"Size of search grid: {grid_combinations}")
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Starting search at: {current_time}")

    parameter_search = BayesSearchCV(
        model,
        search_grid,
        n_iter=20,
        scoring=RMSLE_scorer,
        cv=[(train_indeces, valid_indeces)],
        verbose=3
    )
    parameter_search.fit(
        X_gridsearch,
        y_gridsearch
    )


if SEARCH_PARAMETERS:
    print(f"Best parameters: {parameter_search.best_params_}")
    print(f"Feature importance: {parameter_search.best_estimator_.feature_importances_}")
    predictions = parameter_search.predict(X_valid)
else:
    print(f"Best parameters: {model.get_all_params()}")
    print(f"Feature importance: {model.feature_importances_}")
    predictions = model.predict(X_valid)

print(f"Number of negative predictions: {sum((1 for pred in predictions if pred < 0))}")

zeroed_predictions = np.where(predictions < 0, 0, predictions)
print(f"Score: {evaluate_predictions(zeroed_predictions, y_valid)}")


# url = "https://github.com/andbren/TDT-4173/blob/main/premade/test.csv"
# data_test = read_file(url)
data_test = pd.read_csv("data/preprocessed/test.csv", index_col="id")
if USE_POLAR_COORDINATES:
    data_test.drop(["latitude", "longitude"], axis=1, inplace=True)
else:
    data_test.drop(["distance_from_center", "angle"], axis=1, inplace=True)
    
data_test = data_test.astype(needed_dtypes)

submission = pd.DataFrame()
submission['id'] = data_test.index

if LOG_AREA:
    data_test["area_total"] = np.log1p(data_test["area_total"])
    data_test["area_kitchen"] = np.log1p(data_test["area_kitchen"])
    data_test["area_living"] = np.log1p(data_test["area_living"])


if SEARCH_PARAMETERS:
    predictions_test = parameter_search.predict(data_test)
else:
    predictions_test = model.predict(data_test)

if LOG_TARGET:
    predictions_test = np.expm1(predictions_test)

submission["price_prediction"] = predictions_test

savepath = 'submissions/CatBoost_submission.csv'
submission.to_csv(savepath, index=False)
print(f"Training done! Submission saved to '{savepath}'")



if SEARCH_PARAMETERS:
    # Save best estimator
    parameter_search.best_estimator_.save_model("catboost.model")