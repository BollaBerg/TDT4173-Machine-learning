# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %%
# !pip install catboost
# !pip install scikit-optimize


# %%
from datetime import datetime
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from catboost import CatBoostRegressor, Pool
from sklearn_pandas import DataFrameMapper

from skopt import BayesSearchCV

pd.set_option('display.max_rows', None)


LOG_TARGET = True
SEARCH_PARAMETERS = True
LOG_AREA = True
REMOVE_OUTLIERS = True

print(f"Running with variables: LOG_TARGET={LOG_TARGET}, LOG_AREA={LOG_AREA}, REMOVE_OUTLIERS={REMOVE_OUTLIERS}")

# %%
def read_file(url):
  url = url + "?raw=true"
  df = pd.read_csv(url)
  return df

url = "https://github.com/andbren/TDT-4173/blob/main/apartments_train.csv"
apartments = read_file(url)

url = "https://github.com/andbren/TDT-4173/blob/main/buildings_train.csv"
buildings = read_file(url)

print(f'All apartments have an associated building: {apartments.building_id.isin(buildings.id).all()}')
data = pd.merge(apartments, buildings.set_index('id'), how='left', left_on='building_id', right_index=True)


# %%
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

# %%
# Add price bins, to sort and later split data
NUM_BUCKETS = 10
log_price = np.log10(data['price'])

price_bin_max = log_price.max()
price_bin_min = log_price.min()
price_bin_size = (price_bin_max - price_bin_min) / NUM_BUCKETS

price_bins = [
    i*price_bin_size + price_bin_min for i in range(NUM_BUCKETS)
]
labels = [i for i in range(len(price_bins) - 1)]

data['price_bin'] = pd.cut(log_price, bins=price_bins, labels=labels)
data["price_bin"].fillna(8, inplace=True)

# %% [markdown]
# # Preprocessing

# Remove outliers
if REMOVE_OUTLIERS:
    before_removing = data.shape[0]
    data.drop(data.index[data["price"] > 1.5e9], inplace=True)
    data.drop(data.index[(data["price"] > 0.5e9) & (data["seller"] == 1)], inplace=True)
    data.drop(data.index[data["area_living"] > 600], inplace=True)
    data.drop(data.index[(data["price"] > 0.5e9) & (data["constructed"] > 1900) & (data["constructed"] < 1925)], inplace=True,)
    print(f"Removed {before_removing - data.shape[0]} rows of data. New length: {data.shape[0]}")
    data.reset_index(inplace=True, drop=True)

# %%
# Fill in missing values
default_values = {   
    "seller": 4,          # Add a new category to seller - UNKNOWN = 4
    "area_kitchen": np.median(data["area_kitchen"].dropna()),
    "area_living": np.median(data["area_living"].dropna()),
    "layout": 3,          # Add a new category to layout - UNKNOWN = 3
    "ceiling": np.median(data["ceiling"].dropna()),
    "bathrooms_shared": 0,
    "bathrooms_private": 0,
    "windows_court": 2,   # Change "windows_court" to categorical. New category - UNKNOWN = 2
    "windows_street": 2,  # Change "windows_street" to categorical. New category - UNKNOWN = 2
    "balconies": 0,
    "loggias": 0,
    "condition": 4,       # Add a new category to condition - UNKNOWN = 4
    "phones": 0,
    "new": 2,             # Change "new" to be categorical. New category - UNKNOWN = 2
    "district": 12,       # Add new category to district - UNKNOWN = 12
    "constructed": np.median(data["constructed"].dropna()),
    "material": 7,        # Add new category to material - UNKNOWN = 7
    "elevator_without": 0,
    "elevator_passenger": 0,
    "elevator_service": 0,
    "parking": 3,         # Add new category to parking - UNKNOWN = 3
    "garbage_chute": 0,
    "heating": 4,         # Add new category to heating - UNKNOWN = 4
}

data.fillna(value=default_values, inplace = True)

# %% [markdown]
# # Feature engineering

# %%
# Convert latitude, longitude to polar coordinates
def cartesian_to_polar_coordinates(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
geographical_weighted_center_latitude = np.average(data["latitude"], weights=data["price_bin"])
geographical_weighted_center_longitude = np.average(data["longitude"], weights=data["price_bin"])

delta_latitude = data["latitude"] - geographical_weighted_center_latitude
delta_longitude = data["longitude"] - geographical_weighted_center_longitude

data["distance_from_center"], data["angle"] = cartesian_to_polar_coordinates(delta_latitude, delta_longitude)

# %%
# Convert boolean and categorical values to their right types
needed_dtypes = {
    "seller": CategoricalDtype(categories=[0, 1, 2, 3, 4]),
    "floor": "uint8",
    "rooms": "uint8",
    "layout": CategoricalDtype(categories=[0, 1, 2, 3]),
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
data = data.astype(needed_dtypes)

# %% [markdown]
# # Train/test split

# %%
if LOG_TARGET:
    data["price"] = np.log10(data["price"])
if LOG_AREA:
    data["area_total"] = np.log10(data["area_total"])
    data["area_kitchen"] = np.log10(data["area_kitchen"])
    data["area_living"] = np.log10(data["area_living"])

X_train, X_valid, y_train, y_valid = train_test_split(
    data.drop(["price", "price_bin"], axis=1),
    data["price"],
    test_size = 0.3,
    stratify = data["price_bin"],
)


# %%
fig, (ax1, ax2) = plt.subplots(figsize=(16, 4), ncols=2, dpi=100)
ax1.set_title('Distribution of train set prices after log transform')
if LOG_TARGET:
    sns.histplot(y_train.rename('log10(price)'), ax=ax1)
    sns.histplot(y_valid.rename('log10(price)'), ax=ax2)
else:
    sns.histplot(np.log10(y_train).rename('log10(price)'), ax=ax1)
    sns.histplot(np.log10(y_valid).rename('log10(price)'), ax=ax2)
ax2.set_title('Distribution of validation set prices after log transform')

# %% [markdown]
# # Drop some columns

# %%
# Drop string columns
string_columns = list(X_train.select_dtypes(include=["object"]).columns)

X_train.drop(string_columns, axis=1, inplace=True)
X_valid.drop(string_columns, axis=1, inplace=True)

# Drop either (latitude, longitude) or (distance_from_center, angle)
X_train.drop(["latitude", "longitude"], axis=1, inplace=True)
X_valid.drop(["latitude", "longitude"], axis=1, inplace=True)
# X_train.drop(["distance_from_center", "angle"], axis=1, inplace=True)
# X_valid.drop(["distance_from_center", "angle"], axis=1, inplace=True)


# %%
categorical_columns = list(X_train.select_dtypes(include=["category", "bool"]).columns)
categorical_columns_indices = [X_train.columns.get_loc(c) for c in categorical_columns if c in X_train]

# %% [markdown]
# # Implementation

# %%
valid_set = Pool(
    X_valid,
    label = y_valid,
    cat_features = categorical_columns
)


# %%
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
        ctr_target_border_count=8,
        depth=10,
        iterations=1500,
        l2_leaf_reg=1,
        one_hot_max_size=16
    )
    model.fit(X_train, y_train, eval_set=valid_set, use_best_model=True)


# %%
X_gridsearch = pd.concat((X_train, X_valid))
y_gridsearch = pd.concat((y_train, y_valid))
train_indeces = X_train.index
valid_indeces = X_valid.index


# %%
if SEARCH_PARAMETERS:
    search_grid = {
    "iterations" : [500, 1000, 1500],
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
        n_iter=100,
        scoring=RMSLE_scorer,
        cv=[(train_indeces, valid_indeces)],
        verbose=3
    )
    parameter_search.fit(
        X_gridsearch,
        y_gridsearch
    )

# with open("log.txt", 'w') as log_file, open("log_error.txt", 'w') as err_file:
#     model.grid_search(
#         search_grid,
#         X=X_gridsearch,
#         y=y_gridsearch,
#         cv=[(train_indeces, valid_indeces)],
#         plot=True,
#         log_cout=log_file,
#         log_cerr=err_file
#     )


# %%
# print(f"Best parameters: {model.get_all_params()}")
# print(f"Feature importance: {model.feature_importances_}")
if SEARCH_PARAMETERS:
    print(f"Best parameters: {parameter_search.best_params_}")
    predictions = parameter_search.predict(X_valid)
else:
    predictions = model.predict(X_valid)

if LOG_TARGET:
    predictions = 10 ** predictions
    y_valid = 10 ** y_valid

print(f"Number of negative predictions: {sum((1 for pred in predictions if pred < 0))}")

zeroed_predictions = np.where(predictions < 0, 0, predictions)
print(f"Score: {evaluate_predictions(zeroed_predictions, y_valid)}")

# %% [markdown]
# # Submittable results

# %%
url = "https://github.com/andbren/TDT-4173/blob/main/apartments_test.csv"
apartments_test = read_file(url)

url = "https://github.com/andbren/TDT-4173/blob/main/buildings_test.csv"
buildings_test = read_file(url)

print(f'All apartments have an associated building: {apartments.building_id.isin(buildings.id).all()}')
data_test = pd.merge(apartments_test, buildings_test.set_index('id'), how='left', left_on='building_id', right_index=True)

# %%
# Train model on whole train-set
# model.fit(X_gridsearch, y_gridsearch)

# %%
submission = pd.DataFrame()
submission['id'] = data_test.id

data_test.fillna(value=default_values, inplace = True)
data_test = data_test.astype(needed_dtypes)

delta_latitude = data_test["latitude"] - geographical_weighted_center_latitude
delta_longitude = data_test["longitude"] - geographical_weighted_center_longitude

data_test["distance_from_center"], data_test["angle"] = cartesian_to_polar_coordinates(delta_latitude, delta_longitude)
data_test.drop(["latitude", "longitude"], axis=1, inplace=True)

if LOG_AREA:
    data_test["area_total"] = np.log10(data_test["area_total"])
    data_test["area_kitchen"] = np.log10(data_test["area_kitchen"])
    data_test["area_living"] = np.log10(data_test["area_living"])

X_test = data_test.drop(string_columns, axis=1)

if SEARCH_PARAMETERS:
    predictions_test = parameter_search.predict(X_test)
else:
    predictions_test = model.predict(X_test)

if LOG_TARGET:
    predictions_test = 10 ** predictions_test

submission["price_prediction"] = predictions_test

savepath = 'CatBoost_submission.csv'
submission.to_csv(savepath, index=False)
print(f"Training done! Submission saved to '{savepath}'")



if SEARCH_PARAMETERS:
    # Save best estimator
    parameter_search.best_estimator_.save_model("catboost.model")