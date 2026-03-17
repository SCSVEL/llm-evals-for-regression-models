"""
Model Evaluation Module for Regression Pipeline
Comprehensive evaluation with industry-standard metrics and best practices for LLM evals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from scipy import stats
from typing import Dict, Any, Tuple
import logging
import shap
import lime
import lime.lime_tabular

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class for comprehensive model evaluation."""

    def __init__(self):
        self.metrics = {}
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "Model") -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model for logging

        Returns:
            Dictionary of calculated metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Additional metrics
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
        median_absolute_error = np.median(np.abs(y_true - y_pred))

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Explained_Variance': explained_variance,
            'Median_Absolute_Error': median_absolute_error
        }

        self.metrics[model_name] = metrics
        logger.info(f"{model_name} Metrics: {metrics}")
        return metrics

    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                           cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation and return scores.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            cv: Number of folds

        Returns:
            Dictionary of CV scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        mse_scores = -scores
        cv_metrics = {
            'CV_MSE_Mean': mse_scores.mean(),
            'CV_MSE_Std': mse_scores.std(),
            'CV_RMSE_Mean': np.sqrt(mse_scores).mean(),
            'CV_RMSE_Std': np.sqrt(mse_scores).std()
        }
        logger.info(f"Cross-validation scores: {cv_metrics}")
        return cv_metrics

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: str = None):
        """
        Plot residuals analysis.

        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')

        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')

        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')

        # Residuals over time (if applicable)
        if len(y_true) > 1:
            axes[1, 1].plot(residuals)
            axes[1, 1].set_xlabel('Index')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals Over Time')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Residuals plot saved to {save_path}")
        plt.show()

    def plot_learning_curve(self, model: Any, X: pd.DataFrame, y: pd.Series,
                           cv: int = 5, save_path: str = None):
        """
        Plot learning curve.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            cv: Number of CV folds
            save_path: Path to save the plot
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, scoring='neg_mean_squared_error',
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_scores_mean = -train_scores.mean(axis=1)
        train_scores_std = train_scores.std(axis=1)
        val_scores_mean = -val_scores.mean(axis=1)
        val_scores_std = val_scores.std(axis=1)

        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training MSE")
        plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Validation MSE")
        plt.xlabel('Training Set Size')
        plt.ylabel('MSE')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Learning curve saved to {save_path}")
        plt.show()

    def compute_shap_values(self, model: Any, X: pd.DataFrame,
                           max_evals: int = 1000) -> shap.Explanation:
        """
        Compute SHAP values for explainability.

        Args:
            model: Trained model
            X: Feature matrix
            max_evals: Maximum evaluations for SHAP

        Returns:
            SHAP explanation object
        """
        try:
            explainer = shap.Explainer(model.predict, X)
            shap_values = explainer(X[:min(len(X), 100)])  # Sample for efficiency
            logger.info("Computed SHAP values.")
            return shap_values
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None

    def plot_shap_summary(self, shap_values: shap.Explanation, save_path: str = None):
        """
        Plot SHAP summary plot.

        Args:
            shap_values: SHAP explanation
            save_path: Path to save the plot
        """
        if shap_values is not None:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, show=False)
            if save_path:
                plt.savefig(save_path)
                logger.info(f"SHAP summary plot saved to {save_path}")
            plt.show()

    def lime_explanation(self, model: Any, X: pd.DataFrame, instance_idx: int = 0) -> str:
        """
        Generate LIME explanation for a single instance.

        Args:
            model: Trained model
            X: Feature matrix
            instance_idx: Index of instance to explain

        Returns:
            LIME explanation as string
        """
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X.values, feature_names=X.columns, mode='regression'
            )
            exp = explainer.explain_instance(
                X.iloc[instance_idx].values, model.predict, num_features=5
            )
            explanation = exp.as_list()
            logger.info(f"LIME explanation for instance {instance_idx}: {explanation}")
            return str(explanation)
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return "LIME explanation failed."

    def feature_importance_analysis(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                   save_path: str = None):
        """
        Analyze feature importance using permutation importance.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            save_path: Path to save the plot
        """
        try:
            perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            sorted_idx = perm_importance.importances_mean.argsort()

            plt.figure(figsize=(10, 6))
            plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
            plt.xlabel('Permutation Importance')
            plt.title('Feature Importance (Permutation)')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                logger.info(f"Feature importance plot saved to {save_path}")
            plt.show()
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}")

    def partial_dependence_plot(self, model: Any, X: pd.DataFrame, features: list,
                               save_path: str = None):
        """
        Plot partial dependence for specified features.

        Args:
            model: Trained model
            X: Feature matrix
            features: List of feature names or indices
            save_path: Path to save the plot
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Partial dependence plot saved to {save_path}")
            plt.show()
        except Exception as e:
            logger.error(f"Error in partial dependence plot: {e}")

    def comprehensive_evaluation(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series, model_name: str = "Model",
                                results_dir: str = "results/"):
        """
        Run comprehensive evaluation including metrics, plots, and explainability.

        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            model_name: Name of the model
            results_dir: Directory to save results
        """
        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        metrics = self.calculate_metrics(y_test, y_pred, model_name)

        # Cross-validation
        cv_metrics = self.cross_validate_model(model, X_train, y_train)

        # Plots
        self.plot_residuals(y_test, y_pred, f"{results_dir}residuals_{model_name}.png")
        self.plot_learning_curve(model, X_train, y_train, save_path=f"{results_dir}learning_curve_{model_name}.png")

        # Explainability
        shap_values = self.compute_shap_values(model, X_test)
        self.plot_shap_summary(shap_values, f"{results_dir}shap_summary_{model_name}.png")
        lime_exp = self.lime_explanation(model, X_test)
        self.feature_importance_analysis(model, X_test, y_test, f"{results_dir}feature_importance_{model_name}.png")
        self.partial_dependence_plot(model, X_test, [0], f"{results_dir}partial_dependence_{model_name}.png")

        # Save results
        results = {
            'metrics': metrics,
            'cv_metrics': cv_metrics,
            'lime_explanation': lime_exp
        }

        pd.DataFrame([metrics]).to_csv(f"{results_dir}metrics_{model_name}.csv", index=False)
        logger.info(f"Comprehensive evaluation completed for {model_name}")

        return metrics

    def plot_residuals_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                               save_path: str = None):
        """
        Create comprehensive residual analysis plots.

        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot (optional)
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted Values')

        # Residuals distribution
        sns.histplot(residuals, kde=True, ax=axes[0, 1])
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_title('Residuals Distribution')

        # Q-Q plot for normality
        stats.probplot(residuals, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot')

        # Actual vs Predicted
        axes[1, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title('Actual vs Predicted')

        # Residuals vs Order (to check for patterns)
        axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Observation Order')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Observation Order')

        # Remove empty subplot
        fig.delaxes(axes[1, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual analysis plot saved to {save_path}")

        plt.show()

    def plot_feature_importance(self, feature_names: list, importance_scores: np.ndarray,
                               save_path: str = None):
        """
        Plot feature importance for tree-based models.

        Args:
            feature_names: List of feature names
            importance_scores: Feature importance scores
            save_path: Path to save the plot (optional)
        """
        # Sort features by importance
        indices = np.argsort(importance_scores)
        features_sorted = [feature_names[i] for i in indices]
        scores_sorted = importance_scores[indices]

        plt.figure(figsize=(10, 6))
        plt.barh(features_sorted, scores_sorted)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Analysis')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.show()

    def plot_learning_curves(self, estimator, X_train: pd.DataFrame, y_train: pd.Series,
                           cv: int = 5, save_path: str = None):
        """
        Plot learning curves to diagnose bias/variance.

        Args:
            estimator: Trained model
            X_train: Training features
            y_train: Training target
            cv: Number of CV folds
            save_path: Path to save the plot (optional)
        """
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X_train, y_train, cv=cv,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )

        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = -np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("MSE")
        plt.title("Learning Curves")
        plt.legend(loc="best")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves plot saved to {save_path}")

        plt.show()

    def compare_models(self, model_results: Dict[str, Dict], save_path: str = None):
        """
        Compare multiple models based on their performance.

        Args:
            model_results: Dictionary with model names and their metrics
            save_path: Path to save the plot (optional)
        """
        model_names = list(model_results.keys())
        r2_scores = [model_results[name]['test_r2'] for name in model_names]
        rmse_scores = [model_results[name]['test_rmse'] for name in model_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # R2 comparison
        ax1.bar(model_names, r2_scores, color='skyblue')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Comparison (R²)')
        ax1.tick_params(axis='x', rotation=45)

        # RMSE comparison
        ax2.bar(model_names, rmse_scores, color='lightcoral')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Model Comparison (RMSE)')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")

        plt.show()

    def generate_evaluation_report(self, model_name: str, dataset_info: Dict[str, Any],
                                 metrics: Dict[str, float]) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            model_name: Name of the evaluated model
            dataset_info: Information about the dataset
            metrics: Performance metrics

        Returns:
            Formatted evaluation report
        """
        report = f"""
{'='*60}
REGRESSION MODEL EVALUATION REPORT
{'='*60}

Dataset Information:
- Name: {dataset_info.get('name', 'Unknown')}
- Samples: {dataset_info.get('samples', 'Unknown')}
- Features: {dataset_info.get('features', 'Unknown')}

Model: {model_name}

Performance Metrics:
- R² Score: {metrics['R2']:.4f} ({metrics['R2']*100:.1f}% variance explained)
- RMSE: {metrics['RMSE']:.4f}
- MAE: {metrics['MAE']:.4f}
- MAPE: {metrics['MAPE']:.2f}%
- Explained Variance: {metrics['Explained_Variance']:.4f}

Interpretation:
- The model explains {metrics['R2']*100:.1f}% of the variance in the target variable
- Average prediction error: {metrics['MAE']:.4f} units
- Predictions are off by approximately {metrics['MAPE']:.1f}% on average

Recommendations for LLM Evals:
- {'Excellent performance' if metrics['R2'] > 0.8 else 'Good performance' if metrics['R2'] > 0.7 else 'Needs improvement'}
- Consider cross-validation for robust evaluation
- Monitor for concept drift in production
- Implement A/B testing for model updates

{'='*60}
"""
        return report