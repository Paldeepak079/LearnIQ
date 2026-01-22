"""
Explainability Module for Learning Pattern Analysis System

This module provides human-readable explanations for cluster assignments.

Key Features:
1. Feature importance calculation
2. Distance-to-centroid explanations
3. Plain English generation
4. SHAP-style contribution analysis

Educational Goal:
- Make AI decisions transparent to teachers
- Explain WHY students belong to their assigned pattern
- Identify key behavioral factors
"""

import pandas as pd
import numpy as np


def calculate_feature_importance(student_features, cluster_centroid, feature_names):
    """
    Calculate feature importance based on deviation from cluster centroid.
    
    Logic:
    - Features with larger deviations from centroid are more "important"
    - Importance = |student_value - centroid_value| / std_dev
    
    Args:
        student_features (array-like): Student's feature values
        cluster_centroid (array-like): Cluster center coordinates
        feature_names (list): Names of features
        
    Returns:
        pd.DataFrame: Features ranked by importance with direction
    """
    # Calculate absolute deviations
    deviations = np.abs(np.array(student_features) - np.array(cluster_centroid))
    
    # Normalize to get importance scores
    importance_scores = deviations / (np.std(deviations) + 1e-10)
    
    # Determine direction (above or below centroid)
    directions = np.where(np.array(student_features) > np.array(cluster_centroid), "above", "below")
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores,
        'direction': directions,
        'student_value': student_features,
        'centroid_value': cluster_centroid
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def explain_assignment(student_data, cluster_id, persona_name, cluster_centroid, 
                      feature_names, cluster_profiles):
    """
    Generate a plain English explanation for why a student was assigned to a cluster.
    
    Format:
    - State the assigned pattern
    - Highlight top 3 distinguishing features
    - Provide context and interpretation
    
    Args:
        student_data (pd.Series): Student's complete data
        cluster_id (int): Assigned cluster ID
        persona_name (str): Human-readable persona name
        cluster_centroid (arraylike): Cluster center
        feature_names (list): Feature names
        cluster_profiles (pd.DataFrame): Mean values for all clusters
        
    Returns:
        str: Plain English explanation
    """
    # Get feature values for clustering features only
    student_features = [student_data[f] for f in feature_names]
    
    # Calculate feature importance
    importance_df = calculate_feature_importance(student_features, cluster_centroid, feature_names)
    
    #Get top 3 features
    top_features = importance_df.head(3)
    
    # Build explanation
    explanation = f"Student assigned to: **{persona_name}**\n\n"
    explanation += "**Key Factors:**\n"
    
    for idx, row in top_features.iterrows():
        feature = row['feature']
        student_val = row['student_value']
        centroid_val = row['centroid_value']
        direction = row['direction']
        
        # Create human-readable feature descriptions
        if feature == 'engagement_score':
            explanation += f"• **Engagement**: {student_val:.1f}/100 ({direction} cluster average of {centroid_val:.1f})\n"
        elif feature == 'consistency_index':
            explanation += f"• **Consistency**: {student_val:.1f}/100 ({direction} cluster average of {centroid_val:.1f})\n"
        elif feature == 'performance_trend':
            trend_desc = "improving" if student_val > 0 else ("declining" if student_val < 0 else "stable")
            explanation += f"• **Performance Trend**: {trend_desc} ({student_val:.3f})\n"
        elif feature == 'participation_stability':
            explanation += f"• **Participation**: {student_val:.1f}/100 ({direction} cluster average of {centroid_val:.1f})\n"
        elif feature == 'absences':
            explanation += f"• **Absences**: {student_val:.0f} days ({direction} cluster average of {centroid_val:.1f})\n"
        elif feature in ['G1', 'G2', 'G3']:
            explanation += f"• **{feature} Grade**: {student_val:.1f}/20\n"
        else:
            explanation += f"• **{feature}**: {student_val:.2f}\n"
    
    # Add interpretive summary based on persona
    explanation += "\n**Interpretation:**\n"
    
    if "Consistent Achievers" in persona_name:
        explanation += "This student demonstrates strong, stable performance with high engagement levels. They consistently participate and show commitment to learning."
    elif "Silent Performers" in persona_name:
        explanation += "This student achieves good grades but may not show high visible engagement metrics. They may be self-directed learners who work independently."
    elif "At-Risk Learners" in persona_name:
        explanation += "This student shows concerning patterns in engagement or performance. Early intervention and additional support may be beneficial."
    elif "Irregular Learners" in persona_name:
        explanation += "This student's performance varies significantly over time. They may benefit from strategies to build consistency and stability."
    elif "Emerging Stars" in persona_name:
        explanation += "This student shows positive improvement trends. Continued encouragement and support can help maintain this upward trajectory."
    elif "Declining Performers" in persona_name:
        explanation += "This student's performance is trending downward. Investigation into potential causes and targeted support is recommended."
    else:
        explanation += "This student exhibits a mixed learning pattern. Individual assessment is recommended to identify specific needs and strengths."
    
    return explanation


def generate_student_explanations(df, labels, personas, kmeans_model, feature_names, cluster_profiles):
    """
    Generate explanations for all students using optimized operations.
    """
    df_explained = df.copy()
    df_explained['cluster'] = labels
    df_explained['persona'] = df_explained['cluster'].map(personas)
    
    # Pre-calculate centroids for all students
    centroids = kmeans_model.cluster_centers_[labels]
    
    # Pre-extract student features
    student_f_matrix = df_explained[feature_names].values
    
    # Generate explanations using list comprehension (much faster than iterrows)
    # We pass the pre-calculated features and centroids to avoid repeated lookups
    df_explained['explanation'] = [
        explain_assignment(
            row, labels[i], df_explained['persona'].iloc[i], centroids[i],
            feature_names, cluster_profiles
        )
        for i, (idx, row) in enumerate(df_explained.iterrows())
    ]
    
    print(f"\n✓ Generated explanations for {len(df_explained)} students")
    
    return df_explained


if __name__ == "__main__":
    # Test explainability
    from data_processor import process_data
    from pattern_discovery import discover_patterns
    
    print("="*80)
    print("EXPLAINABILITY LAYER - DEMO")
    print("="*80)
    
    # Load data
    df, X_normalized, scaler, feature_names = process_data()
    
    # Discover patterns
    labels, kmeans, personas, cluster_profiles = discover_patterns(df, X_normalized, feature_names, n_clusters=4)
    
    # Generate explanations
    df_explained = generate_student_explanations(df, labels, personas, kmeans, feature_names, cluster_profiles)
    
    # Show sample explanations
    print("\n" + "="*80)
    print("SAMPLE EXPLANATIONS")
    print("="*80)
    
    for i in range(3):
        print(f"\nStudent {i+1}:")
        print(df_explained.iloc[i]['explanation'])
        print("-"*80)
