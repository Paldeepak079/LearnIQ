"""
Data Processing Module for Learning Pattern Analysis System

This module handles:
1. Loading student data
2. Cleaning and handling missing values
3. Feature normalization
4. Behavioral feature engineering

Educational Context:
- Focus on behavioral patterns, not just academic scores
- Create interpretable features that teachers can understand
- Maintain fairness by avoiding demographic-based features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='student_dataset.csv'):
    """
    Load student dataset with robust error handling.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded student data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Successfully loaded data: {df.shape[0]} students, {df.shape[1]} features")
        
        # Adapt dataset to expected format if needed
        df = adapt_dataset_format(df)
        
        return df
    except FileNotFoundError:
        print(f"[ERROR] Error: File '{filepath}' not found")
        raise
    except Exception as e:
        print(f"[ERROR] Error loading data: {str(e)}")
        raise


def adapt_dataset_format(df):
    """
    Adapt different dataset formats to the expected column structure.
    Handles both Portuguese student dataset and generic student datasets.
    
    Args:
        df (pd.DataFrame): Original dataset
        
    Returns:
        pd.DataFrame: Adapted dataset with expected columns
    """
    df_adapted = df.copy()
    
    # Check if this is the new format (Hours_Studied, Attendance, Exam_Score)
    if 'Hours_Studied' in df.columns and 'Exam_Score' in df.columns:
        print("[INFO] Detected new dataset format - adapting columns...")
        
        # Map new columns to expected names
        column_mapping = {
            'Hours_Studied': 'studytime',
            'Attendance': 'absences',  # Will invert later
            'Extracurricular_Activities': 'activities',
            'Tutoring_Sessions': 'paid',
            'Parental_Education_Level': 'higher',
            'School_Type': 'schoolsup',
            'Parental_Involvement': 'famsup',
            'Physical_Activity': 'goout',
            'Sleep_Hours': 'health',
            'Motivation_Level': 'freetime',
            'Exam_Score': 'G3',
            'Previous_Scores': 'G2'
        }
        
        # Rename columns that exist
        for old_col, new_col in column_mapping.items():
            if old_col in df_adapted.columns:
                df_adapted.rename(columns={old_col: new_col}, inplace=True)
        
        # Create missing essential columns with synthetic data
        if 'G1' not in df_adapted.columns and 'G2' in df_adapted.columns:
            # G1 = G2 with some noise
            df_adapted['G1'] = df_adapted['G2'] + np.random.normal(0, 2, len(df_adapted))
            df_adapted['G1'] = np.clip(df_adapted['G1'], 0, 100)
        
        # Scale studytime to 1-4 range if needed
        if df_adapted['studytime'].max() > 4:
            df_adapted['studytime'] = pd.cut(
                df_adapted['studytime'], 
                bins=4, 
                labels=[1, 2, 3, 4]
            ).astype(int)
        
        # Invert attendance to absences (attendance % -> absence days)
        if df_adapted['absences'].max() <= 100:  # If it looks like a percentage
            df_adapted['absences'] = ((100 - df_adapted['absences']) / 100 * 30).round().astype(int)
        
        # Convert numeric values to yes/no for binary features
        for col in ['activities', 'paid', 'higher', 'schoolsup', 'famsup']:
            if col in df_adapted.columns:
                if df_adapted[col].dtype in ['int64', 'float64']:
                    # Assume values > median are 'yes'
                    median_val = df_adapted[col].median()
                    df_adapted[col] = df_adapted[col].apply(
                        lambda x: 'yes' if x > median_val else 'no'
                    )
        
        # Ensure numeric columns are in correct range
        if 'goout' in df_adapted.columns and df_adapted['goout'].max() > 5:
            df_adapted['goout'] = pd.cut(df_adapted['goout'], bins=5, labels=[1,2,3,4,5]).astype(int)
        
        if 'health' in df_adapted.columns and df_adapted['health'].max() > 5:
            df_adapted['health'] = pd.cut(df_adapted['health'], bins=5, labels=[1,2,3,4,5]).astype(int)
        
        # Add missing columns with default values
        defaults = {
            'age': 16,
            'failures': 0,
            'Dalc': 1,
            'Walc': 1
        }
        
        for col, default_val in defaults.items():
            if col not in df_adapted.columns:
                df_adapted[col] = default_val
        
        print(f"[OK] Adapted {len(column_mapping)} columns to expected format")
    
    return df_adapted


def handle_missing_values(df):
    """
    Handle missing values using appropriate strategies.
    
    Strategy:
    - Numeric features: Median imputation (robust to outliers)
    - Categorical features: Mode imputation (most common value)
    
    Args:
        df (pd.DataFrame): Raw student data
        
    Returns:
        pd.DataFrame: Data with missing values handled
    """
    df_clean = df.copy()
    
    # Separate numeric and categorical columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Impute numeric columns with median
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy='median')
        df_clean[numeric_cols] = numeric_imputer.fit_transform(df_clean[numeric_cols])
    
    # Impute categorical columns with most frequent value
    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df_clean[categorical_cols] = categorical_imputer.fit_transform(df_clean[categorical_cols])
    
    missing_count = df_clean.isnull().sum().sum()
    print(f"[OK] Missing values handled: {missing_count} remaining")
    
    return df_clean


def normalize_features(df, features_to_normalize=None):
    """
    Normalize numeric features using StandardScaler (z-score normalization).
    
    Why StandardScaler?
    - Centers features around mean=0, std=1
    - Preserves distribution shape
    - Works well with clustering algorithms
    
    Args:
        df (pd.DataFrame): Student data
        features_to_normalize (list): List of column names to normalize
        
    Returns:
        pd.DataFrame: Data with normalized features
        StandardScaler: Fitted scaler (for inverse transform if needed)
    """
    df_normalized = df.copy()
    
    if features_to_normalize is None:
        # Default: normalize all numeric columns except IDs
        features_to_normalize = df.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = StandardScaler()
    df_normalized[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    
    print(f"[OK] Normalized {len(features_to_normalize)} features")
    
    return df_normalized, scaler


def engineer_behavioral_features(df):
    """
    Engineer 4 key behavioral indicators for learning pattern analysis.
    
    These features are designed to be:
    - Interpretable by teachers
    - Behaviorally meaningful (not just grade-based)
    - Predictive of learning patterns
    
    Features Created:
    1. Engagement Score (0-100): How actively student participates in learning
    2. Consistency Index (0-100): How stable their performance is over time
    3. Performance Trend (-1 to 1): Whether improving, declining, or stable
    4. Participation Stability (0-100): Regularity of attendance and support usage
    
    Args:
        df (pd.DataFrame): Student data with original features
        
    Returns:
        pd.DataFrame: Data with additional behavioral features
    """
    df_eng = df.copy()
    
    # ========================================================================
    # 1. ENGAGEMENT SCORE (0-100)
    # ========================================================================
    # Composite metric combining:
    # - Study time (studytime: 1-4 scale)
    # - Activities participation (activities: yes/no)
    # - Paid extra classes (paid: yes/no)
    # - Wants higher education (higher: yes/no)
    # 
    # Educational rationale: Engaged students invest time, seek opportunities,
    # and show commitment to learning beyond minimum requirements.
    # ========================================================================
    
    # Convert yes/no to binary
    df_eng['activities_binary'] = (df_eng['activities'] == 'yes').astype(int)
    df_eng['paid_binary'] = (df_eng['paid'] == 'yes').astype(int)
    df_eng['higher_binary'] = (df_eng['higher'] == 'yes').astype(int)
    
    # Normalize studytime to 0-1 scale (originally 1-4)
    studytime_normalized = (df_eng['studytime'] - 1) / 3
    
    # Calculate engagement score (weighted average, then scale to 0-100)
    df_eng['engagement_score'] = (
        (studytime_normalized * 0.4) +           # 40% weight on study time
        (df_eng['activities_binary'] * 0.2) +    # 20% weight on activities
        (df_eng['paid_binary'] * 0.2) +          # 20% weight on extra classes
        (df_eng['higher_binary'] * 0.2)          # 20% weight on higher ed aspiration
    ) * 100
    
    # ========================================================================
    # 2. CONSISTENCY INDEX (0-100)
    # ========================================================================
    # Measures how stable a student's performance is across G1, G2, G3.
    # 
    # Using Coefficient of Variation (CV) inverted:
    # - CV = (std / mean) measures variability
    # - Lower CV = more consistent
    # - We invert it: Consistency = 1 - normalized_CV
    # 
    # Educational rationale: Consistent students have predictable learning
    # patterns, while inconsistent students may need intervention.
    # ========================================================================
    
    # Vectorized consistency calculation
    grades_cols = ['G1', 'G2', 'G3']
    grade_data = df_eng[grades_cols].values
    
    mean_grade = np.mean(grade_data, axis=1)
    std_grade = np.std(grade_data, axis=1)
    
    # Avoid division by zero
    cv = np.zeros_like(mean_grade)
    mask = mean_grade > 0
    cv[mask] = std_grade[mask] / mean_grade[mask]
    
    # Invert and normalize (high score = high consistency)
    df_eng['consistency_index'] = (1 - np.minimum(cv, 1.0)) * 100
    
    # ========================================================================
    # 3. PERFORMANCE TREND (-1 to 1)
    # ========================================================================
    # Captures whether student is improving, declining, or stable.
    # 
    # Using linear regression slope of grades over time:
    # - Positive slope = improving
    # - Negative slope = declining
    # - Near-zero slope = stable
    # 
    # Educational rationale: Trend reveals learning trajectory, which is
    # often more informative than absolute scores.
    # ========================================================================
    
    # Vectorized trend calculation (slope of 3 points)
    # For time points [1, 2, 3], slope = (y3 - y1) / 2
    # Assuming maximum realistic change is ±10 points per quarter
    df_eng['performance_trend'] = np.clip((df_eng['G3'] - df_eng['G1']) / 20, -1, 1)
    
    # ========================================================================
    # 4. PARTICIPATION STABILITY (0-100)
    # ========================================================================
    # Combines attendance regularity and support system engagement.
    # 
    # Components:
    # - Attendance rate (inverse of absences)
    # - School support usage (schoolsup: yes/no)
    # - Family support (famsup: yes/no)
    # 
    # Educational rationale: Stable participation indicates commitment and
    # supportive environment. High absences with low support = risk factor.
    # ========================================================================
    
    # Normalize absences (inverse: fewer absences = higher stability)
    # Absences range typically 0-30+, cap at 30 for normalization
    max_absences = 30
    attendance_score = (1 - np.minimum(df_eng['absences'], max_absences) / max_absences)
    
    # Convert support indicators to binary
    df_eng['schoolsup_binary'] = (df_eng['schoolsup'] == 'yes').astype(int)
    df_eng['famsup_binary'] = (df_eng['famsup'] == 'yes').astype(int)
    
    # Calculate participation stability (weighted average, scale to 0-100)
    df_eng['participation_stability'] = (
        (attendance_score * 0.6) +               # 60% weight on attendance
        (df_eng['schoolsup_binary'] * 0.2) +     # 20% weight on school support
        (df_eng['famsup_binary'] * 0.2)          # 20% weight on family support
    ) * 100
    
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print("\n[OK] Engineered Behavioral Features:")
    print(f"  - Engagement Score: {df_eng['engagement_score'].mean():.1f} (avg)")
    print(f"  - Consistency Index: {df_eng['consistency_index'].mean():.1f} (avg)")
    print(f"  - Performance Trend: {df_eng['performance_trend'].mean():.3f} (avg)")
    print(f"  - Participation Stability: {df_eng['participation_stability'].mean():.1f} (avg)")
    
    return df_eng


def prepare_clustering_features(df):
    """
    Prepare the final feature set for clustering.
    
    Selects only behavioral features (excludes demographics for fairness).
    
    Args:
        df (pd.DataFrame): Data with engineered features
        
    Returns:
        pd.DataFrame: Feature matrix ready for clustering
        list: Feature names
    """
    # Core behavioral features for clustering
    clustering_features = [
        'engagement_score',
        'consistency_index',
        'performance_trend',
        'participation_stability',
        'studytime',
        'failures',
        'absences',
        'G1', 'G2', 'G3'  # Include grades for context
    ]
    
    # Ensure all features exist
    available_features = [f for f in clustering_features if f in df.columns]
    
    X = df[available_features].copy()
    
    print(f"\n[OK] Prepared {len(available_features)} features for clustering")
    
    return X, available_features


# ============================================================================
# Main Pipeline Function
# ============================================================================

def process_data(filepath='student_dataset.csv'):
    """
    Complete data processing pipeline.
    
    Steps:
    1. Load data
    2. Handle missing values
    3. Engineer behavioral features
    4. Prepare clustering features
    5. Normalize
    
    Args:
        filepath (str): Path to student CSV
        
    Returns:
        tuple: (processed_df, normalized_features, scaler, feature_names)
    """
    print("="*80)
    print("LEARNING PATTERN ANALYSIS - DATA PROCESSING PIPELINE")
    print("="*80)
    
    # Step 1: Load
    df = load_data(filepath)
    
    # Step 2: Clean
    df = handle_missing_values(df)
    
    # Step 3: Engineer features
    df = engineer_behavioral_features(df)
    
    # Step 4: Prepare clustering features
    X, feature_names = prepare_clustering_features(df)
    
    # Step 5: Normalize
    X_normalized, scaler = normalize_features(X, feature_names)
    
    print("\n" + "="*80)
    print("[OK] DATA PROCESSING COMPLETE")
    print("="*80)
    
    return df, X_normalized, scaler, feature_names


if __name__ == "__main__":
    # Test the pipeline
    df, X_normalized, scaler, feature_names = process_data()
    print("\n[DATA] Sample processed data:")
    print(df[['engagement_score', 'consistency_index', 'performance_trend', 'participation_stability']].head())
