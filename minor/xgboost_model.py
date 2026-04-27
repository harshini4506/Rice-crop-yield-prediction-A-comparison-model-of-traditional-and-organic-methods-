import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(
    r"C:\Users\gotlu\OneDrive\Desktop\academic_traditional_dataset_v2.csv"
)

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)

TARGET = "Expected_Yield_Tons_per_HA"

categorical_cols = [
    "Rice_Variety",
    "Soil_Texture",
    "Irrigation_Type"
]

numerical_cols = [
    col for col in df.columns
    if col not in categorical_cols + [TARGET]
]

df = df.dropna()

X = df[categorical_cols + numerical_cols]
y = df[TARGET]

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# PREPROCESSING
# ----------------------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numerical_cols)
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("xgb", XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    ))
])

# ----------------------------
# HYPERPARAMETER TUNING
# ----------------------------
param_grid = {
    "xgb__n_estimators": [300, 500],
    "xgb__max_depth": [4, 6],
    "xgb__learning_rate": [0.05, 0.1],
    "xgb__subsample": [0.8, 1]
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

# ----------------------------
# EVALUATION
# ----------------------------
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\n🌾 XGBoost Yield Model")
print("-----------------------")
print("Best Parameters:", grid.best_params_)
print("R² Accuracy Score:", round(r2, 3))

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(best_model, "xgboost_yield_model.pkl")
print("💾 Model saved as xgboost_yield_model.pkl")
