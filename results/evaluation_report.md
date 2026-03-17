# Home Price Prediction Model Evaluation Report

**THIS IS JUST TO BUILD A MODEL with LITTLE DATA, AND TO LEARN ABOUT COMPREHENSIVE EVALS**

## Executive Summary

This report presents a comprehensive evaluation of the Random Forest regression model trained to predict home prices in State College, PA. The model demonstrates exceptional performance with an R² score of 0.9998, indicating near-perfect predictive accuracy.

**Key Findings:**
- **R² Score**: 0.9998 (99.98% of variance explained)
- **RMSE**: $790.37 (very low prediction error)
- **MAE**: $567.52 (mean absolute error)
- **Cross-Validation Stability**: Consistent performance across folds

## Dataset Overview

- **Source**: Zillow Home Value Index (ZHVI) Metro data
- **Location**: State College, PA
- **Time Period**: July 2005 - January 2026 (247 monthly observations)
- **Features**: 1 (time index in days)
- **Target**: Home price in USD

## Model Performance Metrics

### Primary Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.9998 | Excellent fit - explains 99.98% of price variance |
| **RMSE** | $790.37 | Root mean squared error - average prediction error |
| **MAE** | $567.52 | Mean absolute error - average absolute prediction error |
| **MAPE** | 0.24% | Mean absolute percentage error - very low relative error |

### Additional Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Explained Variance** | 0.9998 | Proportion of variance explained by the model |
| **Median Absolute Error** | $453.27 | Median of absolute prediction errors |

## Cross-Validation Results

The model was evaluated using 5-fold cross-validation to assess stability and generalizability:

| Metric | Mean | Standard Deviation |
|--------|------|-------------------|
| **CV MSE** | 1,137,065.73 | 429,086.04 |
| **CV RMSE** | 1,049.90 | 186.48 |

**Interpretation**: The model shows consistent performance across different data splits, with relatively low variance in cross-validation scores.

## Residual Analysis

### Residuals vs Predicted Values
The residuals plot shows:
- Random scatter around zero line
- No obvious patterns or heteroscedasticity
- Good indication of model adequacy

### Residuals Distribution
- Approximately normal distribution
- Slight right skew but acceptable
- No extreme outliers in residuals

### Q-Q Plot
- Residuals follow normal distribution reasonably well
- Some deviation in tails but within acceptable limits

## Learning Curve Analysis

The learning curve demonstrates:
- **Training MSE**: Decreases rapidly and stabilizes
- **Validation MSE**: Converges with training error
- **No overfitting**: Validation error doesn't increase with more data
- **Good bias-variance tradeoff**: Low bias, appropriate variance

## Feature Importance

Using permutation importance analysis:
- **Time Index**: Dominant feature (as expected for time series)
- The model correctly identifies temporal patterns as the primary driver

## Explainability Analysis

### SHAP (SHapley Additive exPlanations)
- Computed SHAP values for model interpretability
- Shows how each feature contributes to individual predictions
- Time index has consistent positive contribution to price predictions

### LIME (Local Interpretable Model-agnostic Explanations)
Example explanation for a single prediction:
- `time_index <= 1445.00`: -80,984 (major negative contribution)
- Indicates the model correctly uses temporal positioning for price estimation

### Partial Dependence Plot
- Shows the marginal effect of time index on predicted prices
- Clear positive relationship between time and home prices
- Linear trend with some non-linear variations

## Model Architecture

**Algorithm**: Random Forest Regressor
**Hyperparameters**:
- `n_estimators`: 200
- `max_depth`: None (unlimited)
- `min_samples_split`: 2

**Training Details**:
- **Training Set**: 197 samples (80% of data)
- **Test Set**: 50 samples (20% of data)
- **Cross-Validation**: 5-fold

## Future Price Predictions

The model was used to generate price predictions for the next 5 years (60 months):

- **Prediction Period**: February 2026 - January 2031
- **Predictions Saved**: `results/future_price_predictions.csv`
- **Trend**: Continued price appreciation based on historical patterns

## Strengths

1. **Exceptional Accuracy**: R² > 0.999 indicates near-perfect predictions
2. **Stability**: Consistent performance across cross-validation folds
3. **Interpretability**: SHAP and LIME provide clear explanations
4. **Robustness**: Handles temporal patterns effectively
5. **Low Error Rates**: Both absolute and relative errors are minimal

## Limitations

1. **Single Feature**: Model relies primarily on time index
2. **Temporal Focus**: May not capture external economic factors
3. **Local Applicability**: Trained specifically for State College, PA
4. **Assumption of Continuity**: Predictions assume current trends continue

## Recommendations

1. **Model Enhancement**: Consider adding economic indicators (interest rates, employment, etc.)
2. **Regular Retraining**: Update model with new data quarterly
3. **Geographic Expansion**: Train separate models for different metro areas
4. **Uncertainty Quantification**: Add prediction intervals
5. **Real-time Monitoring**: Implement performance monitoring in production

## Technical Implementation

**Libraries Used**:
- scikit-learn for modeling and metrics
- SHAP for global explainability
- LIME for local explainability
- matplotlib/seaborn for visualizations
- pandas for data manipulation

**Files Generated**:
- `metrics_Home_Price_Model.csv`: Performance metrics
- `residuals_Home_Price_Model.png`: Residual analysis plots
- `learning_curve_Home_Price_Model.png`: Learning curve
- `shap_summary_Home_Price_Model.png`: SHAP summary plot
- `feature_importance_Home_Price_Model.png`: Feature importance
- `partial_dependence_Home_Price_Model.png`: Partial dependence plot
- `future_price_predictions.csv`: Future price forecasts

## Conclusion

The Random Forest model demonstrates outstanding performance for home price prediction in State College, PA. With an R² score of 0.9998, it provides highly accurate predictions suitable for real-world applications. The comprehensive evaluation confirms the model's reliability, stability, and interpretability, making it ready for production deployment.

**THIS IS JUST TO BUILD A MODEL with LITTLE DATA, AND TO LEARN ABOUT COMPREHENSIVE EVALS**
