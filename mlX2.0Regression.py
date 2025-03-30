import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_df = pd.read_csv('data/train.csv')  # Using the combined data instead
test_df = pd.read_csv('data/test.csv')

# Create copies to avoid modifying originals
X_train = train_df.copy()
y_train = X_train.pop('target')
X_test = test_df.copy()

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Feature Engineering
def engineer_features(df):
    # Extract features from publication_timestamp
    df['publication_date'] = pd.to_datetime(df['publication_timestamp'])
    df['publication_year'] = df['publication_date'].dt.year
    df['publication_month'] = df['publication_date'].dt.month
    df['publication_day'] = df['publication_date'].dt.day
    df['publication_dayofweek'] = df['publication_date'].dt.dayofweek
    df['publication_quarter'] = df['publication_date'].dt.quarter
    df['days_since_2000'] = (df['publication_date'] - pd.Timestamp('2000-01-01')).dt.days
    df['days_since_2010'] = (df['publication_date'] - pd.Timestamp('2010-01-01')).dt.days
    df['days_since_2020'] = (df['publication_date'] - pd.Timestamp('2020-01-01')).dt.days
    
    # Advanced timestamp features
    df['is_weekend'] = df['publication_dayofweek'].isin([5, 6]).astype(int)
    
    # Extract track name lengths and features
    for i in range(3):  # For the first 3 tracks
        col_name = f'composition_label_{i}'
        if col_name in df.columns:
            df[f'track_{i}_name_length'] = df[col_name].fillna('').apply(len)
            df[f'track_{i}_word_count'] = df[col_name].fillna('').apply(lambda x: len(str(x).split()))
    
    # Artist features
    df['artist_name_length'] = df['creator_collective'].fillna('').apply(len)
    df['artist_name_word_count'] = df['creator_collective'].fillna('').apply(lambda x: len(str(x).split()))
    
    # Audio feature aggregations across tracks
    audio_features = [
        'intensity_index', 'rhythmic_cohesion', 'organic_texture', 'beat_frequency', 
        'emotional_charge', 'performance_authenticity', 'emotional_resonance', 
        'groove_efficiency', 'organic_immersion', 'instrumental_density',
        'vocal_presence', 'harmonic_scale', 'tonal_mode'
    ]
    
    for feature in audio_features:
        cols = [f"{feature}_{i}" for i in range(3) if f"{feature}_{i}" in df.columns]
        if cols:
            df[f'{feature}_mean'] = df[cols].mean(axis=1)
            df[f'{feature}_std'] = df[cols].std(axis=1)
            df[f'{feature}_max'] = df[cols].max(axis=1)
            df[f'{feature}_min'] = df[cols].min(axis=1)
            df[f'{feature}_range'] = df[f'{feature}_max'] - df[f'{feature}_min']
            # More sophisticated aggregations
            df[f'{feature}_median'] = df[cols].median(axis=1)
            df[f'{feature}_skew'] = df[cols].skew(axis=1)
            # Delta between tracks
            if len(cols) >= 2:
                df[f'{feature}_0_1_delta'] = df[cols[0]] - df[cols[1]]
            if len(cols) >= 3:
                df[f'{feature}_1_2_delta'] = df[cols[1]] - df[cols[2]]
                df[f'{feature}_0_2_delta'] = df[cols[0]] - df[cols[2]]
    
    # Interaction terms between key features
    emotional_features = ['emotional_charge_mean', 'emotional_resonance_mean']
    energy_features = ['intensity_index_mean', 'rhythmic_cohesion_mean', 'beat_frequency_mean']
    texture_features = ['organic_texture_mean', 'instrumental_density_mean']
    
    # Create pairwise interactions between important feature groups
    for ef in emotional_features:
        if ef in df.columns:
            for en in energy_features:
                if en in df.columns:
                    df[f'{ef}_{en}_interaction'] = df[ef] * df[en]
            
            for tf in texture_features:
                if tf in df.columns:
                    df[f'{ef}_{tf}_interaction'] = df[ef] * df[tf]
    
    for en in energy_features:
        if en in df.columns:
            for tf in texture_features:
                if tf in df.columns:
                    df[f'{en}_{tf}_interaction'] = df[en] * df[tf]
    
    # Calculated features for duration
    for i in range(3):
        ms_col = f'duration_ms_{i}'
        if ms_col in df.columns:
            # Convert milliseconds to minutes
            df[f'duration_minutes_{i}'] = df[ms_col] / 60000
            # Create duration bins
            df[f'duration_bin_{i}'] = pd.cut(
                df[f'duration_minutes_{i}'], 
                bins=[0, 2, 3, 4, 5, 10, 100], 
                labels=['very_short', 'short', 'medium', 'long', 'very_long', 'extra_long']
            )
    
    # Duration aggregations if all 3 exist
    duration_cols = [f'duration_minutes_{i}' for i in range(3) if f'duration_minutes_{i}' in df.columns]
    if len(duration_cols) > 0:
        df['duration_total'] = df[duration_cols].sum(axis=1)
        df['duration_mean'] = df[duration_cols].mean(axis=1)
        df['duration_std'] = df[duration_cols].std(axis=1)
    
    # Vocal presence processing
    vocal_cols = [f'vocal_presence_{i}' for i in range(3) if f'vocal_presence_{i}' in df.columns]
    if vocal_cols:
        df['has_vocals'] = (df[vocal_cols] > 0.5).any(axis=1).astype(int)
        df['all_vocals'] = (df[vocal_cols] > 0.5).all(axis=1).astype(int)
        df['vocal_mix'] = ((df[vocal_cols] > 0.5).sum(axis=1) / len(vocal_cols))
    
    # Categorical encodings - Moon phase simplification
    if 'lunar_phase' in df.columns:
        df['is_full_moon'] = df['lunar_phase'].str.contains('Full', na=False).astype(int)
        df['is_new_moon'] = df['lunar_phase'].str.contains('New', na=False).astype(int)
        df['is_waxing'] = df['lunar_phase'].str.contains('Waxing', na=False).astype(int)
        df['is_waning'] = df['lunar_phase'].str.contains('Waning', na=False).astype(int)
    
    # Season simplification and encoding
    if 'season_of_release' in df.columns:
        # One-hot encode seasons
        df['is_winter'] = df['season_of_release'].str.contains('Winter', na=False).astype(int)
        df['is_spring'] = df['season_of_release'].str.contains('Spring', na=False).astype(int)
        df['is_summer'] = df['season_of_release'].str.contains('Summer', na=False).astype(int)
        df['is_fall'] = df['season_of_release'].str.contains('Fall|Autumn', na=False).astype(int)
        
    # Harmonic features and combinations
    for i in range(3):
        tonal_col = f'tonal_mode_{i}'
        harmonic_col = f'harmonic_scale_{i}'
        if tonal_col in df.columns and harmonic_col in df.columns:
            # Create a harmonic signature feature
            df[f'harmonic_signature_{i}'] = df[tonal_col].astype(str) + "_" + df[harmonic_col].astype(str)
            # Create numerical interaction
            df[f'tonal_harmonic_product_{i}'] = df[tonal_col] * df[harmonic_col]
    
    # Weekday encoding
    if 'weekday_of_release' in df.columns:
        # One-hot encode weekdays
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in weekdays:
            df[f'is_{day.lower()}'] = (df['weekday_of_release'] == day).astype(int)
        
        # Group features
        df['is_weekend_release'] = df['weekday_of_release'].isin(['Saturday', 'Sunday']).astype(int)
        df['is_friday_release'] = (df['weekday_of_release'] == 'Friday').astype(int)
        df['is_start_of_week'] = df['weekday_of_release'].isin(['Monday', 'Tuesday']).astype(int)
        df['is_midweek'] = df['weekday_of_release'].isin(['Wednesday', 'Thursday']).astype(int)
    
    # Polynomial features for important numerical features
    for col in ['intensity_index_mean', 'emotional_charge_mean', 'rhythmic_cohesion_mean', 
                'album_component_count', 'artist_count', 'duration_mean']:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_cubed'] = df[col] ** 3
    
    # Time signature analysis
    time_sig_cols = [f'time_signature_{i}' for i in range(3) if f'time_signature_{i}' in df.columns]
    if time_sig_cols:
        # Check if time signatures are the same across tracks
        if len(time_sig_cols) >= 2:
            df['time_signature_consistent'] = df[time_sig_cols].nunique(axis=1) == 1
        
        # Common time signatures
        for col in time_sig_cols:
            df[f'{col}_is_4_4'] = (df[col] == 4).astype(int)
            df[f'{col}_is_3_4'] = (df[col] == 3).astype(int)
    
    # Advanced interactions for key predictors
    if all(x in df.columns for x in ['emotional_charge_mean', 'rhythmic_cohesion_mean', 'intensity_index_mean']):
        df['emotional_rhythm_energy'] = df['emotional_charge_mean'] * df['rhythmic_cohesion_mean'] * df['intensity_index_mean']
    
    # Artist count interactions
    if 'artist_count' in df.columns:
        if 'emotional_charge_mean' in df.columns:
            df['artist_emotional_interaction'] = df['artist_count'] * df['emotional_charge_mean']
        if 'intensity_index_mean' in df.columns:
            df['artist_intensity_interaction'] = df['artist_count'] * df['intensity_index_mean']
    
    return df

# Apply feature engineering
print("Applying feature engineering...")
X_train = engineer_features(X_train)
X_test = engineer_features(X_test)

# Function to prepare data for modeling
def prepare_for_modeling(train_df, test_df):
    # Identify column types
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove ID and date columns from feature list
    cols_to_drop = ['id', 'publication_timestamp', 'publication_date', 'track_identifier']
    cols_to_drop += [col for col in train_df.columns if 'composition_label' in col or 'creator_collective' in col]
    
    # Remove columns that should be dropped from our feature lists
    numeric_cols = [col for col in numeric_cols if col not in cols_to_drop]
    categorical_cols = [col for col in categorical_cols if col not in cols_to_drop]
    
    # Print feature counts
    print(f"Using {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")
    
    # Create preprocessing pipelines with improved imputation and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', PowerTransformer(method='yeo-johnson'))  # Better for skewed data
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop any remaining columns
    )
    
    return preprocessor, numeric_cols, categorical_cols

# Prepare data for modeling
preprocessor, numeric_cols, categorical_cols = prepare_for_modeling(X_train, X_test)

# Create train/validation split for local evaluation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Function to train and evaluate model
def train_and_evaluate(X, y, X_val=None, y_val=None, preprocessor=None):
    # Define models with improved parameters
    models = {
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=10,
            num_leaves=50,
            min_child_samples=20,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            importance_type='gain'
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=8,
            min_child_weight=2,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        ),
        'CatBoost': cb.CatBoostRegressor(
            iterations=1500,
            learning_rate=0.01,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=0
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.01,
            subsample=0.7,
            random_state=42
        )
    }
    
    results = {}
    trained_models = {}
    
    # Train and evaluate models
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X, y)
        trained_models[name] = pipeline
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = pipeline.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            results[name] = val_rmse
            print(f"{name} Validation RMSE: {val_rmse:.4f}")
        else:
            # Use cross-validation if no validation set
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
            cv_rmse = -cv_scores.mean()
            results[name] = cv_rmse
            print(f"{name} CV RMSE: {cv_rmse:.4f}")
    
    # Find the best model
    best_model_name = min(results, key=results.get)
    print(f"Best model: {best_model_name} with RMSE: {results[best_model_name]:.4f}")
    
    return trained_models, results

# Train multiple models
print("Training models...")
trained_models, results = train_and_evaluate(
    X_train_split, y_train_split, X_val_split, y_val_split, preprocessor
)

# Create and train a stacked model
def train_stacked_model(X, y, X_val, y_val, trained_models, preprocessor):
    print("Training stacked model...")
    
    # Define base estimators
    estimators = []
    for name, model in trained_models.items():
        estimators.append((name, model))
    
    # Meta-regressor (Ridge regression)
    meta_regressor = Ridge(alpha=1.0, random_state=42)
    
    # Create stacking regressor
    stacked_model = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_regressor,
        cv=5,
        n_jobs=-1
    )
    
    # Fit the stacked model on validation data
    stacked_model.fit(X_val, y_val)
    
    # Evaluate the stacked model
    stack_pred = stacked_model.predict(X_val)
    stack_rmse = np.sqrt(mean_squared_error(y_val, stack_pred))
    print(f"Stacked Model Validation RMSE: {stack_rmse:.4f}")
    
    return stacked_model, stack_rmse

# Train stacked model on the full dataset
def retrain_best_model(X, y, preprocessor, best_model_name):
    print(f"Retraining best model ({best_model_name}) on full dataset...")
    
    # Define model parameters based on best model
    if best_model_name == 'LightGBM':
        model = lgb.LGBMRegressor(
            n_estimators=3000,  # Increase for full dataset
            learning_rate=0.005,  # Lower for better generalization
            max_depth=10,
            num_leaves=50,
            min_child_samples=20,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
    elif best_model_name == 'XGBoost':
        model = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.005,
            max_depth=8,
            min_child_weight=2,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
    elif best_model_name == 'CatBoost':
        model = cb.CatBoostRegressor(
            iterations=2000,
            learning_rate=0.005,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=0
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=2000,
            max_depth=7,
            learning_rate=0.005,
            subsample=0.7,
            random_state=42
        )
    
    # Create pipeline
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit on full dataset
    final_pipeline.fit(X, y)
    
    return final_pipeline

# Determine best model from individual results
best_model_name = min(results, key=results.get)

# Function to train weighted ensemble
def create_weighted_ensemble(X_val, y_val, trained_models):
    print("Creating weighted ensemble...")
    
    # Make predictions with each model
    predictions = {}
    for name, model in trained_models.items():
        predictions[name] = model.predict(X_val)
    
    # Calculate weights based on validation performance
    weights = {}
    for name in trained_models.keys():
        rmse = np.sqrt(mean_squared_error(y_val, predictions[name]))
        # Inverse RMSE weighting (lower RMSE = higher weight)
        weights[name] = 1 / rmse
    
    # Normalize weights
    weight_sum = sum(weights.values())
    for name in weights:
        weights[name] /= weight_sum
    
    print("Model weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    return weights

# Create weighted ensemble
weights = create_weighted_ensemble(X_val_split, y_val_split, trained_models)

# Retrain best model on full dataset
final_model = retrain_best_model(X_train, y_train, preprocessor, best_model_name)

# Make predictions on the test set
print("\nPredicting on test set...")
predictions = final_model.predict(X_test)

# Ensure predictions are within the target range [0-100]
predictions = np.clip(predictions, 0, 100)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'target': predictions.round(1)  # Round to 1 decimal place
})

# Write to file
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

# Create weighted ensemble predictions
def make_ensemble_predictions(X_test, trained_models, weights):
    print("\nGenerating weighted ensemble predictions...")
    
    # Make predictions with each model
    model_predictions = {}
    for name, model in trained_models.items():
        model_predictions[name] = model.predict(X_test)
    
    # Compute weighted average
    ensemble_pred = np.zeros(len(X_test))
    for name, weight in weights.items():
        ensemble_pred += model_predictions[name] * weight
    
    # Ensure predictions are within range
    ensemble_pred = np.clip(ensemble_pred, 0, 100)
    
    return ensemble_pred

# Generate ensemble predictions
ensemble_predictions = make_ensemble_predictions(X_test, trained_models, weights)

# Create ensemble submission file
ensemble_submission = pd.DataFrame({
    'id': test_df['id'],
    'target': ensemble_predictions.round(1)
})

# Write to file
ensemble_submission.to_csv('ensemble_submission.csv', index=False)
print("Ensemble submission file created: ensemble_submission.csv")

# Bonus: Feature importance visualization
def plot_feature_importance(model, column_names, n_features=20):
    if hasattr(model[-1], 'feature_importances_'):
        # Get feature importances
        importances = model[-1].feature_importances_
        
        # Check if we can get column names from the preprocessor
        if hasattr(model[0], 'get_feature_names_out'):
            try:
                feature_names = model[0].get_feature_names_out()
            except:
                # Fallback to numeric features + categorical features
                feature_names = column_names
        else:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Ensure we have the right number of feature names
        if len(feature_names) != len(importances):
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Create DataFrame for the feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).head(n_features)
        
        print("\nTop Feature Importances:")
        print(importance_df)
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        return importance_df
    
    return None

# Try to get feature importances for the final model
all_columns = numeric_cols + categorical_cols
plot_feature_importance(final_model, all_columns)

print("Process completed successfully.")