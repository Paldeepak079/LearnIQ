"""
Risk Detection Module for Learning Pattern Analysis System

This module implements early warning system for student disengagement.

Detection Rules:
1. Attendance drop (high absences)
2. Engagement decline before grade drop
3. Performance volatility
4. Participation decrease

Risk Levels:
- Normal: 0 flags
- Watchlist: 1-2 flags
- High Risk: 3+ flags

Educational Context:
- Early intervention is key to student success
- Multiple indicators provide stronger signal than single metric
- Focus on actionable, observable behaviors
"""

import pandas as pd
import numpy as np


def detect_attendance_drop(student_data, threshold_percentile=75):
    """
    Flag students with high absence rates.
    
    Args:
        student_data (pd.Series): Student row
        threshold_percentile (float): Percentile threshold
        
    Returns:
        tuple: (is_flagged, reason)
    """
    absences = student_data['absences']
    
    # Simple threshold: more than 10 absences is concerning SHAP-
    if absences > 10:
        return True, f"High absences ({absences:.0f} days)"
    
    return False, ""


def detect_engagement_decline(student_data):
    """
    Flag students with low engagement scores.
    
    Educational insight: Low engagement often precedes declining grades.
    
    Args:
        student_data (pd.Series): Student row
        
    Returns:
        tuple: ( is_flagged, reason)
    """
    engagement = student_data['engagement_score']
    avg_grade = (student_data['G1'] + student_data['G2'] + student_data['G3']) / 3
    
    # Flag if engagement is low (< 30) even if grades haven't fully dropped yet
    if engagement < 30 and avg_grade > 0:
        return True, f"Low engagement ({engagement:.1f}/100)"
    
    return False, ""


def detect_performance_volatility(student_data):
    """
    Flag students with highly inconsistent performance.
    
    Args:
        student_data (pd.Series): Student row
        
    Returns:
        tuple: (is_flagged, reason)
    """
    consistency = student_data['consistency_index']
    
    # Flag if consistency is very low (< 50)
    if consistency < 50:
        return True, f"Inconsistent performance (consistency: {consistency:.1f}/100)"
    
    return False, ""


def detect_declining_trend(student_data):
    """
    Flag students with negative performance trends.
    
    Args:
        student_data (pd.Series): Student row
        
    Returns:
        tuple: (is_flagged, reason)
    """
    trend = student_data['performance_trend']
    
    # Flag if trend is significantly negative
    if trend < -0.15:
        return True, f"Declining grades (trend: {trend:.3f})"
    
    return False, ""


def detect_low_participation(student_data):
    """
    Flag students with low participation stability.
    
    Args:
        student_data (pd.Series): Student row
        
    Returns:
        tuple: (is_flagged, reason)
    """
    participation = student_data['participation_stability']
    
    # Flag if participation is low (< 35)
    if participation < 35:
        return True, f"Low participation ({participation:.1f}/100)"
    
    return False, ""


def classify_risk(student_data):
    """
    Classify student risk level based on multiple flags.
    
    Args:
        student_data (pd.Series): Student row
        
    Returns:
        tuple: (risk_level, flags, reasons)
    """
    flags = []
    reasons = []
    
    # Run all detection functions
    is_flagged, reason = detect_attendance_drop(student_data)
    if is_flagged:
        flags.append('attendance')
        reasons.append(reason)
    
    is_flagged, reason = detect_engagement_decline(student_data)
    if is_flagged:
        flags.append('engagement')
        reasons.append(reason)
    
    is_flagged, reason = detect_performance_volatility(student_data)
    if is_flagged:
        flags.append('volatility')
        reasons.append(reason)
    
    is_flagged, reason = detect_declining_trend(student_data)
    if is_flagged:
        flags.append('declining')
        reasons.append(reason)
    
    is_flagged, reason = detect_low_participation(student_data)
    if is_flagged:
        flags.append('participation')
        reasons.append(reason)
    
    # Determine risk level
    num_flags = len(flags)
    
    if num_flags == 0:
        risk_level = "Normal"
    elif num_flags <= 2:
        risk_level = "Watchlist"
    else:
        risk_level = "High Risk"
    
    return risk_level, flags, reasons


def explain_risk(risk_level, flags, reasons):
    """
    Generate explanation for risk classification.
    
    Args:
        risk_level (str): Normal/Watchlist/High Risk
        flags (list): List of flag types
        reasons (list): List of reason strings
        
    Returns:
        str: Plain English explanation
    """
    if risk_level == "Normal":
        return "No concerning patterns detected. Student is on track."
    
    explanation = f"**Risk Level: {risk_level}**\n\n"
    explanation += f"**Flags: {len(flags)}**\n"
    
    for i, reason in enumerate(reasons, 1):
        explanation += f"{i}. {reason}\n"
    
    if risk_level == "Watchlist":
        explanation += "\n**Recommendation**: Monitor student progress. Consider light check-in."
    else:
        explanation += "\n**Recommendation**: Immediate intervention recommended. Schedule meeting with student."
    
    return explanation


def analyze_all_students(df):
    """
    Analyze risk for all students.
    
    Args:
        df (pd.DataFrame): Student dataset with behavioral features
        
    Returns:
        pd.DataFrame: Dataset with risk analysis added
    """
    print("\n" + "="*80)
    print("RISK DETECTION ANALYSIS")
    print("="*80)
    
    df_risk = df.copy()
    
    risk_levels = []
    risk_flags = []
    risk_reasons = []
    risk_explanations = []
    
    for idx, row in df_risk.iterrows():
        risk_level, flags, reasons = classify_risk(row)
        explanation = explain_risk(risk_level, flags, reasons)
        
        risk_levels.append(risk_level)
        risk_flags.append(flags)
        risk_reasons.append(reasons)
        risk_explanations.append(explanation)
    
    df_risk['risk_level'] = risk_levels
    df_risk['risk_flags'] = risk_flags
    df_risk['risk_reasons'] = risk_reasons
    df_risk['risk_explanation'] = risk_explanations
    
    # Summary statistics
    risk_dist = df_risk['risk_level'].value_counts()
    print("\nRisk Distribution:")
    for level in ['Normal', 'Watchlist', 'High Risk']:
        count = risk_dist.get(level, 0)
        pct = (count / len(df_risk)) * 100
        print(f"  • {level}: {count} students ({pct:.1f}%)")
    
    print("\n✓ Risk analysis complete")
    
    return df_risk


if __name__ == "__main__":
    # Test risk detection
    from data_processor import process_data
    
    print("="*80)
    print("RISK DETECTION - DEMO")
    print("="*80)
    
    # Load data
    df, X_normalized, scaler, feature_names = process_data()
    
    # Analyze risk
    df_risk = analyze_all_students(df)
    
    # Show high risk students
    print("\n" + "="*80)
    print("HIGH RISK STUDENTS (Sample)")
    print("="*80)
    
    high_risk = df_risk[df_risk['risk_level'] == 'High Risk'].head(5)
    
    for idx, row in high_risk.iterrows():
        print(f"\nStudent (index {idx}):")
        print(row['risk_explanation'])
        print("-"*80)
