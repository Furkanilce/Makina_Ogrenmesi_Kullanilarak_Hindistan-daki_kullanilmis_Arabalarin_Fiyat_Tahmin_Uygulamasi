import pandas as pd
import numpy as np
from pandas.core.common import random_state
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib
import streamlit as st

df = pd.read_csv("C:/Users/Furkan/Desktop/Proje/Dataset/cars_data_clean.csv")

df["car_age"] = 2025 - df["myear"]
df["km_year"] = df["km"] / df["car_age"].replace(0, 1)
df["marka_model"] = df["oem"] + " " + df["model"]

numerical_property = ["car_age", "km", "km_year", "Length", "Width", "Height", "Wheel Base", "Seats",
                   "Max Power Delivered", "Max Torque Delivered"]
categorical_feature = ["marka_model", "fuel", "transmission", "body", "City", "Color", "Drive Type", "carType", "owner_type"]


y = df["listed_price"]
x = df[numerical_property + categorical_feature]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)


on_isleme = ColumnTransformer(
    transformers = [
        ("sayisal", Pipeline([
            ("inputer", SimpleImputer(strategy = "mean")),
            ("scaler", StandardScaler())
        ]), numerical_property),

        ("kategorik", Pipeline([
            ("inputer1", SimpleImputer(strategy = "most_frequent")),
            ("encoder", TargetEncoder()),
        ]), categorical_feature)
    ])

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

#param_grid_rf = {
#    "mak_model__n_estimators" : [100, 300, 500],
#    "mak_model__max_depth" : [None, 10, 20],
#}

param_grid_xgb = {
    "mak_model__n_estimators": [300, 500, 800],
    "mak_model__max_depth": [4, 6, 8],
    "mak_model__learning_rate": [0.01, 0.05, 0.1],
}

#param_grid_knn = {
#    "mak_model__n_neighbors": [3, 5, 7, 9, 11],
#    "mak_model__weights": ["uniform", "distance"],
#    "mak_model__metric": ["euclidean", "manhattan"]
#}



#rf_model = Pipeline(steps = [
#    ("on_isleme", on_isleme),
#    ("mak_model", RandomForestRegressor(random_state = 42)),
#])
#grid_rf = GridSearchCV(rf_model, param_grid = param_grid_rf, cv = 3, n_jobs = -1, scoring = "r2")
#grid_rf.fit(x_train, y_train_log)

xgb_model = Pipeline(steps = [
    ("on_isleme", on_isleme),
    ("mak_model", XGBRegressor(random_state = 42)),
])
grid_xgb = GridSearchCV(xgb_model, param_grid = param_grid_xgb, cv = 3, n_jobs = -1, scoring = "r2")
grid_xgb.fit(x_train, y_train_log)

#knn_model = Pipeline(steps = [
#    ("on_isleme", on_isleme),
#    ("mak_model", KNeighborsRegressor())
#])
#grid_knn = GridSearchCV(knn_model, param_grid = param_grid_knn, cv = 3, n_jobs = -1, scoring = "r2")
#grid_knn.fit(x_train, y_train_log)





#y_pred_rf = np.expm1(grid_rf.best_estimator_.predict(x_test))
#print("RandomForest R²:", r2_score(y_test, y_pred_rf))
#print("RandomForest mean squared error:", mean_squared_error(y_test, y_pred_rf))
#print("RandomForest mean absolute error:", mean_absolute_error(y_test, y_pred_rf))
#print("-----------------------------------------------------------------------------------------")

#y_pred_xgb = np.expm1(grid_xgb.best_estimator_.predict(x_test))
#print("XGBoost R²:", r2_score(y_test, y_pred_xgb))
#print("XGBoost mean squared error :", mean_squared_error(y_test, y_pred_xgb))
#print("XGBoost mean absolute error :", mean_absolute_error(y_test, y_pred_xgb))
#print("-----------------------------------------------------------------------------------------")

#Makalede Kullanılan Algoritma
#y_pred_knn = np.expm1(grid_knn.best_estimator_.predict(x_test))
#print("KNN R²:", r2_score(y_test, y_pred_knn))
#print("KNN mean squared error: ", mean_squared_error(y_test, y_pred_knn))
#print("KNN mean absolute error: ", mean_absolute_error(y_test, y_pred_knn))


#sayisal_ozellik = ["araba_yasi", "km", "km_year", "Length", "Width", "Height", "Wheel Base", "Seats",
#                   "Max Power Delivered", "Max Torque Delivered"]
#kategorik_ozellikler = ["marka_model", "fuel", "transmission", "body", "City", "Color", "Drive Type", "carType", "owner_type"]

joblib.dump(grid_xgb.best_estimator_, "xgb_model.pkl")
print("Model başarıyla eğitildi ve 'xgb_model.pkl' olarak kaydedildi.")