# organic_recommendation_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ----------------------------------
# LOAD DATA
# ----------------------------------
df = pd.read_csv("Rice_Organic_Yield_Recommendation_Dataset.csv")
df.columns = df.columns.str.strip()

# ----------------------------------
# RENAME COLUMNS
# ----------------------------------
df = df.rename(columns={
    "Nitrogen_N": "Nitrogen_kg_ha",
    "Phosphorus_P": "Phosphorus_kg_ha",
    "Potassium_K": "Potassium_kg_ha",
    "Total_Yield_tons_per_hectare": "Expected_Yield_Tons_per_HA"
})

# ----------------------------------
# TARGET & FEATURES
# ----------------------------------
TARGET = "Expected_Yield_Tons_per_HA"

categorical_cols = [
    "Soil_Type",
    "Rice_Variety",
    "Organic_Fertilizer_Used"
]

numerical_cols = [
    "Rainfall_mm",
    "Temperature_C",
    "Nitrogen_kg_ha",
    "Phosphorus_kg_ha",
    "Potassium_kg_ha",
    "Soil_pH",
    "Organic_Content_Ratio_%"
]

df = df[categorical_cols + numerical_cols + [TARGET]].dropna()

# ----------------------------------
# 🔥 FEATURE ENGINEERING
# ----------------------------------
df["Total_NPK"] = (
    df["Nitrogen_kg_ha"] +
    df["Phosphorus_kg_ha"] +
    df["Potassium_kg_ha"]
)

df["Temp_Rainfall_Index"] = df["Temperature_C"] * df["Rainfall_mm"]

numerical_cols += ["Total_NPK", "Temp_Rainfall_Index"]

X = df[categorical_cols + numerical_cols]
y = df[TARGET]

# ----------------------------------
# TRAIN–TEST SPLIT
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# PREPROCESSING
# ----------------------------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numerical_cols)
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestRegressor(random_state=42))
])

# ----------------------------------
# 🔥 HYPERPARAMETER TUNING (OPTIMIZED)
# ----------------------------------
param_grid = {
    "rf__n_estimators": [500, 800],
    "rf__max_depth": [14, 20],
    "rf__min_samples_split": [4, 8],
    "rf__min_samples_leaf": [2, 4]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ----------------------------------
# EVALUATION
# ----------------------------------
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\n🌱 Improved Organic Yield Model")
print("----------------------------------")
print("Best Parameters:", grid.best_params_)
print("R² Accuracy Score:", round(r2, 3))

# ----------------------------------
# SAVE MODEL
# ----------------------------------
joblib.dump(best_model, "organic_yield_recommendation_model.pkl")
print("💾 Model saved as organic_yield_recommendation_model.pkl")