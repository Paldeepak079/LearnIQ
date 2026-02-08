"""
Predictive Analysis Module for LearnIQ
Provides grade forecasting using Random Forest Regression
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


class GradePredictor:
    """Predicts final grades (G3) based on student behavioral patterns"""
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.mae = None
        self.r2 = None
        
    def train(self, df):
        """Train the Random Forest model on the dataset"""
        # Select features for prediction
        feature_cols = [
            'age', 'studytime', 'failures', 'absences', 'G1', 'G2',
            'goout', 'Dalc', 'Walc', 'health', 'freetime'
        ]
        
        # Prepare data
        X = df[feature_cols].copy()
        y = df['G3'].copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.mae, self.r2
    
    def predict(self, student_data):
        """Predict final grade for a student"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        feature_cols = [
            'age', 'studytime', 'failures', 'absences', 'G1', 'G2',
            'goout', 'Dalc', 'Walc', 'health', 'freetime'
        ]
        
        X = student_data[feature_cols].values.reshape(1, -1)
        prediction = self.model.predict(X)[0]
        
        # Calculate confidence interval (simplified)
        predictions = [tree.predict(X)[0] for tree in self.model.estimators_]
        std = np.std(predictions)
        
        return {
            'predicted_grade': round(prediction, 2),
            'confidence_lower': round(max(0, prediction - std), 2),
            'confidence_upper': round(min(20, prediction + std), 2),
            'pass_probability': self._calculate_pass_prob(prediction, std)
        }
    
    def _calculate_pass_prob(self, mean, std):
        """Calculate probability of passing (G3 >= 10)"""
        from scipy.stats import norm
        z_score = (10 - mean) / (std + 0.01)  # Avoid division by zero
        prob = 1 - norm.cdf(z_score)
        return round(prob * 100, 1)
    
    def what_if_analysis(self, student_data, variable, new_value):
        """
        Simulate what happens if we change a behavioral variable
        
        Args:
            student_data: Current student data
            variable: Variable to change (e.g., 'studytime', 'absences')
            new_value: New value for the variable
        """
        modified_data = student_data.copy()
        modified_data[variable] = new_value
        
        original_pred = self.predict(student_data)
        modified_pred = self.predict(modified_data)
        
        return {
            'original': original_pred['predicted_grade'],
            'modified': modified_pred['predicted_grade'],
            'change': round(modified_pred['predicted_grade'] - original_pred['predicted_grade'], 2),
            'variable': variable,
            'new_value': new_value
        }
