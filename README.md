# Music Track Popularity Prediction

## Project Overview
This repository contains a machine learning solution for predicting the popularity of music tracks based on various audio features, metadata, and contextual information. The project implements an ensemble approach combining multiple advanced regression models to achieve high prediction accuracy.

## Features
- **Advanced Feature Engineering**: Comprehensive feature extraction from temporal, audio, and metadata attributes
- **Ensemble Modeling**: Combines LightGBM, XGBoost, CatBoost, and Gradient Boosting models
- **Weighted Prediction**: Uses validation performance to weight model predictions
- **Robust Preprocessing**: Handles missing values, categorical features, and data scaling

## Repository Structure
```
├── data/                  # Data directory (not included in repo)
│   ├── train.csv          # Training dataset
│   ├── test.csv           # Test dataset
│   └── sample_submission.csv  # Submission format
├── mlX2.0Regression.py    # Main Python script with full implementation
├── mlX2.0Regression.ipynb # Jupyter notebook version
├── submission.csv         # Generated predictions
└── README.md              # This file
```

## Requirements
The solution requires the following Python packages:
```
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
matplotlib
```

## Implementation Details

### Feature Engineering
The solution implements extensive feature engineering including:
- Temporal features from publication date (year, month, day, cyclical representations)
- Audio feature aggregations (mean, std, max, min, range)
- Interaction terms between key features
- Track metadata processing
- Categorical encoding for lunar phases, seasons, and weekdays

### Model Pipeline
1. **Data Preprocessing**:
   - KNN imputation for missing values
   - Power transformation for numerical features
   - One-hot encoding for categorical features

2. **Model Training**:
   - Multiple gradient boosting variants (LightGBM, XGBoost, CatBoost, GradientBoosting)
   - Validation-based model evaluation
   - Weighted ensemble prediction

3. **Prediction**:
   - Weighted averaging of model predictions
   - Clipping to ensure predictions are within valid range

## Usage
To run the prediction pipeline:

```bash
python mlX2.0Regression.py
```

This will:
1. Load and preprocess the data
2. Train multiple regression models
3. Create a weighted ensemble
4. Generate predictions
5. Save the results to submission.csv and ensemble_submission.csv

## Jupyter Notebook
For interactive exploration, use the Jupyter notebook version:

```bash
jupyter notebook mlX2.0Regression.ipynb
```

## Performance
The model achieves a validation RMSE of approximately 13.6, demonstrating strong predictive performance on the music popularity task.

## License
[MIT License](https://opensource.org/licenses/MIT)
