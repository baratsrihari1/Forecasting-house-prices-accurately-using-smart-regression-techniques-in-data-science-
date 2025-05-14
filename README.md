# Forecasting-house-prices-accurately-using-smart-regression-techniques-in-data-science-

# House Price Prediction - Complete Implementation
# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. Data Loading
# ======================
print("Loading dataset...")
try:
    df = pd.read_csv('data/house_data.csv')  # Update path to your dataset
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    print("Error: File not found. Please check the path.")
    exit()

# ======================
# 2. Data Preprocessing
# ======================
print("\nPreprocessing data...")

def preprocess_data(df):
    # Drop columns with >30% missing values
    df = df.dropna(axis=1, thresh=0.7*len(df))
    
    # Fill numerical missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical missing values
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

df = preprocess_data(df)
print("Missing values after preprocessing:")
print(df.isnull().sum().sort_values(ascending=False))

# ======================
# 3. Exploratory Data Analysis
# ======================
print("\nPerforming EDA...")
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True, bins=30)
plt.title('Distribution of House Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.savefig('plots/price_distribution.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.savefig('plots/correlation_matrix.png')
plt.close()

# ======================
# 4. Feature Engineering
# ======================
print("\nEngineering new features...")

def create_features(df):
    # Price per square foot
    df['price_per_sqft'] = df['price'] / df['sqft_living']
    
    # Age of the house
    df['house_age'] = pd.Timestamp.now().year - df['yr_built']
    
    # Total rooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Has basement
    df['has_basement'] = df['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)
    
    return df

df = create_features(df)

# ======================
# 5. Model Preparation
# ======================
print("\nPreparing for model training...")

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# ======================
# 6. Model Training
# ======================
print("\nTraining models...")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    # Create and train pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)])
    pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    print(f"\n{name} Performance:")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")

# ======================
# 7. Model Evaluation
# ======================
print("\nEvaluating models...")

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)

# Visual comparison
plt.figure(figsize=(12, 6))
results_df['RMSE'].sort_values().plot(kind='bar', color='skyblue')
plt.title('Model Comparison by RMSE (Lower is Better)')
plt.ylabel('Root Mean Squared Error ($)')
plt.xticks(rotation=45)
plt.savefig('plots/model_comparison.png')
plt.close()

# Feature importance
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
rf_pipeline.fit(X_train, y_train)

# Get feature names
feature_names = numerical_cols.tolist()
if len(categorical_cols) > 0:
    ohe_features = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    feature_names.extend(ohe_features)

# Plot top 10 features
importances = rf_pipeline.named_steps['model'].feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(10, 6))
plt.title('Top 10 Important Features (Random Forest)')
plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig('plots/feature_importance.png')
plt.close()

# ======================
# 8. Save Best Model
# ======================
print("\nSaving the best model...")
best_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, random_state=42))])
best_model.fit(X_train, y_train)

# Save model
joblib.dump(best_model, 'models/house_price_predictor.pkl')
print("Best model (XGBoost) saved to 'models/house_price_predictor.pkl'")

# ======================
# 9. Generate Predictions
# ======================
print("\nGenerating sample predictions...")
sample_data = X_test.iloc[:5].copy()
predictions = best_model.predict(sample_data)

print("\nSample Predictions:")
for i, (idx, row) in enumerate(sample_data.iterrows()):
    print(f"House {i+1}:")
    print(f"Actual Price: ${y_test.loc[idx]:,.2f}")
    print(f"Predicted Price: ${predictions[i]:,.2f}")
    print(f"Difference: ${abs(y_test.loc[idx] - predictions[i]):,.2f}\n")

print("\nHouse Price Prediction Project Completed Successfully!")
