import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("bmw_cars_market_dataset_synthetic.csv")

# Target value
target = "price_usd"

# Features
X = df.drop(columns=[target])
y = df[target]

# Identify Feature types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# preprocessing

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ML Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train Test Split
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))

# Save model
joblib.dump(pipeline, "bmw_price_model.pkl")
print("Model saved successfully")