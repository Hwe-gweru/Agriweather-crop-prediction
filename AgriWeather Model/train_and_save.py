import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# List of target crops
CROPS_LIST = [
    'Maize', 'Wheat', 'Soybean', 'Sunflower', 
    'Groundnuts', 'Beans', 'Cotton', 'Sorghum'
]

print("Generating dataset...")
np.random.seed(42)
n_samples = 2000

df = pd.DataFrame({
    'crop': np.random.choice(CROPS_LIST, size=n_samples),
    'rainfall': np.random.uniform(200, 1400, size=n_samples),
    'frost_level': np.random.uniform(0, 15, size=n_samples),
    'temperature': np.random.uniform(12, 38, size=n_samples),
    'solar_radiation': np.random.uniform(10, 30, size=n_samples),
    'wind_speed': np.random.uniform(1, 25, size=n_samples),
    'yield_t_ha': np.random.uniform(0.5, 9.0, size=n_samples)
})

X = df[['crop', 'rainfall', 'frost_level', 'temperature', 'solar_radiation', 'wind_speed']]
y = df['yield_t_ha']

# Build Column Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['crop']),
        ('num', 'passthrough', ['rainfall', 'frost_level', 'temperature', 'solar_radiation', 'wind_speed'])
    ]
)

# Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=150, 
        learning_rate=0.07, 
        max_depth=5, 
        random_state=42
    ))
])

# Fit Pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model_pipeline.fit(X_train, y_train)

# Save Pipeline to disk
joblib.dump(model_pipeline, 'crop_yield_model.joblib')
print("✅ Pipeline saved successfully to 'crop_yield_model.joblib'!")