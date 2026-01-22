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
    Analyze risk for all students using vectorized operations.
    """
    print("\n" + "="*80)
    print("RISK DETECTION ANALYSIS")
    print("="*80)
    
    df_risk = df.copy()
    
    # 1. Attendance flags: absences > 10
    abs_flags = df_risk['absences'] > 10
    abs_reasons = np.where(abs_flags, df_risk['absences'].apply(lambda x: f"High absences ({x:.0f} days)"), "")
    
    # 2. Engagement flags: engagement < 30
    eng_flags = df_risk['engagement_score'] < 30
    eng_reasons = np.where(eng_flags, df_risk['engagement_score'].apply(lambda x: f"Low engagement ({x:.1f}/100)"), "")
    
    # 3. Volatility flags: consistency < 50
    vol_flags = df_risk['consistency_index'] < 50
    vol_reasons = np.where(vol_flags, df_risk['consistency_index'].apply(lambda x: f"Inconsistent performance (consistency: {x:.1f}/100)"), "")
    
    # 4. Trend flags: performance_trend < -0.15
    trend_flags = df_risk['performance_trend'] < -0.15
    trend_reasons = np.where(trend_flags, df_risk['performance_trend'].apply(lambda x: f"Declining grades (trend: {x:.3f})"), "")
    
    # 5. Participation flags: participation < 35
    part_flags = df_risk['participation_stability'] < 35
    part_reasons = np.where(part_flags, df_risk['participation_stability'].apply(lambda x: f"Low participation ({x:.1f}/100)"), "")
    
    # Combine flags and calculate count
    all_flags = [abs_flags, eng_flags, vol_flags, trend_flags, part_flags]
    num_flags = np.sum(all_flags, axis=0)
    
    # Assign risk level
    # 0 = Normal, 1-2 = Watchlist, 3+ = High Risk
    df_risk['risk_level'] = np.select(
        [num_flags == 0, num_flags <= 2],
        ["Normal", "Watchlist"],
        default="High Risk"
    )
    
    # Create lists of reasons for each student (needed for explanations)
    all_reasons = [abs_reasons, eng_reasons, vol_reasons, trend_reasons, part_reasons]
    
    # Zip together flags and reasons (this is the only non-vectorized part but much faster than full iterrows)
    # We only do this for flags that are actually set
    def get_student_reasons(row_idx):
        active_reasons = [all_reasons[i][row_idx] for i in range(5) if all_flags[i][row_idx]]
        active_flag_names = [f for i, f in enumerate(['attendance', 'engagement', 'volatility', 'declining', 'participation']) if all_flags[i][row_idx]]
        return active_flag_names, active_reasons

    zipped = [get_student_reasons(i) for i in range(len(df_risk))]
    df_risk['risk_flags'] = [z[0] for z in zipped]
    df_risk['risk_reasons'] = [z[1] for z in zipped]
    
    # Generate explanations using list comprehension (faster than apply)
    df_risk['risk_explanation'] = [
        explain_risk(row['risk_level'], row['risk_flags'], row['risk_reasons']) 
        for _, row in df_risk.iterrows()
    ]
    
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
