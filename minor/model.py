# model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ----------------------------------
# LOAD DATA (YOUR NEW PATH)
# ----------------------------------
df = pd.read_csv(
    r"C:\Users\gotlu\OneDrive\Desktop\academic_traditional_dataset_v2.csv"
)

# ----------------------------------
# CLEAN COLUMN NAMES AUTOMATICALLY
# ----------------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
    .str.replace("/", "_")
)

print("\nAvailable Columns:")
print(df.columns.tolist())

# ----------------------------------
# DEFINE TARGET (EDIT IF NEEDED)
# ----------------------------------
TARGET = "Expected_Yield_Tons_per_HA"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset!")

# ----------------------------------
# FEATURE SELECTION (FIXED)
# ----------------------------------

categorical_cols = [
    "Rice_Variety",
    "Soil_Texture",
    "Irrigation_Type"
]

numerical_cols = [
    col for col in df.columns
    if col not in categorical_cols + [TARGET]
]

print("\nCategorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)


# ----------------------------------
# PREPARE FEATURES & TARGET
# ----------------------------------

X = df[categorical_cols + numerical_cols]
y = df[TARGET]

# ----------------------------------
# TRAIN–TEST SPLIT
# ----------------------------------

from sklearn.model_selection import train_test_split

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
# HYPERPARAMETER TUNING
# ----------------------------------
param_grid = {
    "rf__n_estimators": [400, 700],
    "rf__max_depth": [12, 18],
    "rf__min_samples_split": [5, 10],
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

print("\n🌾 Improved Traditional Yield Model")
print("----------------------------------")
print("Best Parameters:", grid.best_params_)
print("R² Accuracy Score:", round(r2, 3))

# ----------------------------------
# SAVE MODEL
# ----------------------------------
joblib.dump(best_model, "traditional_yield_model.pkl")
print("💾 Model saved as traditional_yield_model.pkl")
