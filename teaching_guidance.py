"""
Teaching Guidance Module for Learning Pattern Analysis System

This module maps learning patterns to evidence-based teaching strategies.

Educational Framework:
- Based on differentiated instruction principles
- Growth mindset focused (non-punitive)
- Actionable and specific
- Aligned to individual learning needs

Intervention Strategies:
- Mentorship and peer learning
- Active learning prompts
- Personalized check-ins
- Study skills development
- Advanced challenges
"""

import pandas as pd


# Strategy database (evidence-based interventions)
STRATEGY_DATABASE = {
    "Consistent Achievers": [
        "**Provide Advanced Challenges**: Offer enrichment activities to maintain engagement",
        "**Peer Mentorship**: Pair with struggling students for mutual benefit",
        "**Leadership Opportunities**: Encourage class participation and group leadership",
        "**Self-Directed Projects**: Allow exploration of topics of personal interest",
        "**Recognition**: Acknowledge achievements to maintain motivation"
    ],
    
    "Silent Performers": [
        "**Encourage Participation**: Create low-pressure opportunities to share (written responses, small groups)",
        "**Build Confidence**: Provide positive feedback on strengths",
        "**Explore Learning Style**: Understand if student prefers independent work",
        "**Offer Choice**: Allow alternative demonstration methods (presentations, projects)",
        "**Check Understanding**: Regular one-on-one check-ins to ensure comprehension"
    ],
    
    "At-Risk Learners": [
        "**Immediate Intervention**: Schedule individual meeting to understand barriers",
        "**Study Skills Workshop**: Provide training on organization, time management, note-taking",
        "**Frequent Check-ins**: Weekly progress monitoring and support",
        "**Counseling Referral**: Connect with school counselor for additional support",
        "**Parent Communication**: Engage family in support plan",
        "**Peer Tutoring**: Assign supportive peer mentor",
        "**Break Down Assignments**: Provide scaffolded, smaller tasks"
    ],
    
    "Irregular Learners": [
        "**Establish Routines**: Help create consistent study schedule",
        "**Frequent Formative Assessment**: Regular low-stakes quizzes to track understanding",
        "**Active Learning Prompts**: In-class activities to boost engagement",
        "**Flexible Deadlines**: Consider extensions with structured plans",
        "**Identify Patterns**: Investigate causes of inconsistency (health, home life, etc.)",
        "**Goal Setting**: Work together to set small, achievable goals"
    ],
    
    "Emerging Stars": [
        "**Positive Reinforcement**: Celebrate improvements to maintain momentum",
        "**Growth Mindset Messaging**: Emphasize effort and progress over fixed ability",
        "**Incremental Challenges**: Gradually increase difficulty to build confidence",
        "**Regular Feedback**: Provide specific, constructive feedback on progress",
        "**Share Success**: Highlight as model for growth (with permission)"
    ],
    
    "Declining Performers": [
        "**Investigate Causes**: Meet with student to understand what changed",
        "**Intervention Plan**: Create targeted support based on identified issues",
        "**Re-engagement Strategies**: Rekindle interest through relevance and choice",
        "**Academic Support**: Provide tutoring or additional resources",
        "**Monitor Closely**: Frequent check-ins until trend reverses",
        "**Counselor Involvement**: Check for personal or emotional factors"
    ],
    
    "Mixed Pattern Learners": [
        "**Individual Assessment**: Conduct detailed review of student's unique profile",
        "**Differentiated Instruction**: Tailor approach based on specific needs",
        "**Regular Monitoring**: Track progress to identify emerging patterns",
        "**Flexible Support**: Adjust strategies as patterns become clearer",
        "**Student Voice**: Ask student what support would be most helpful"
    ]
}


def get_pattern_strategies(persona_name):
    """
    Get teaching strategies for a specific learning pattern.
    
    Args:
        persona_name (str): Learning pattern persona
        
    Returns:
        list: List of recommended strategies
    """
    return STRATEGY_DATABASE.get(persona_name, STRATEGY_DATABASE["Mixed Pattern Learners"])


def generate_personalized_recommendations(student_data, persona_name, risk_level):
    """
    Generate personalized teaching recommendations for a student.
    
    Combines:
    - Pattern-based strategies
    - Risk-level urgency
    - Individual behavioral context
    
    Args:
        student_data (pd.Series): Student information
        persona_name (str): Assigned learning pattern
        risk_level (str): Risk classification
        
    Returns:
        dict: Personalized recommendations
    """
    # Get base strategies for pattern
    base_strategies = get_pattern_strategies(persona_name)
    
    recommendations = {
        'pattern': persona_name,
        'risk_level': risk_level,
        'priority': 'High' if risk_level == 'High Risk' else ('Medium' if risk_level == 'Watchlist' else 'Low'),
        'strategies': base_strategies.copy()
    }
    
    # Add risk-specific urgent actions
    if risk_level == "High Risk":
        urgent_actions = [
            "**URGENT**: Immediate intervention required",
            "**Schedule meeting within 48 hours**",
            "**Develop individualized support plan**"
        ]
        recommendations['strategies'] = urgent_actions + recommendations['strategies']
    
    elif risk_level == "Watchlist":
        watch_actions = [
            "**MONITOR**: Keep close watch on progress",
            "**Check in within next week**"
        ]
        recommendations['strategies'] = watch_actions + recommendations['strategies']
    
    # Add context-specific recommendations based on student data
    additional = []
    
    # High absences
    if student_data.get('absences', 0) > 10:
        additional.append("**Attendance Support**: Investigate barriers to attendance, connect with family")
    
    # Low studytime
    if student_data.get('studytime', 2) == 1:
        additional.append("**Time Management**: Provide study time management resources")
    
    # No school/family support
    if student_data.get('schoolsup', 'no') == 'no' and student_data.get('famsup', 'no') == 'no':
        additional.append("**Support Systems**: Connect student with school support services")
    
    # Multiple failures
    if student_data.get('failures', 0) >= 2:
        additional.append("**Academic Recovery**: Develop plan to address foundational gaps")
    
    if additional:
        recommendations['context_specific'] = additional
    
    return recommendations


def generate_class_guidance(df_with_patterns):
    """
    Generate class-level teaching guidance summary.
    
    Args:
        df_with_patterns (pd.DataFrame): Student data with patterns and risk
        
    Returns:
        dict: Class-level recommendations
    """
    class_guidance = {
        'total_students': len(df_with_patterns),
        'pattern_distribution': df_with_patterns['persona'].value_counts().to_dict(),
        'risk_distribution': df_with_patterns['risk_level'].value_counts().to_dict(),
        'priorities': []
    }
    
    # Identify class priorities
    high_risk_count = len(df_with_patterns[df_with_patterns['risk_level'] == 'High Risk'])
    at_risk_count = len(df_with_patterns[df_with_patterns['persona'].str.contains('At-Risk', na=False)])
    
    if high_risk_count > 0:
        class_guidance['priorities'].append(
            f"**PRIORITY 1**: {high_risk_count} students at high risk - immediate intervention needed"
        )
    
    watchlist_count = len(df_with_patterns[df_with_patterns['risk_level'] == 'Watchlist'])
    if watchlist_count > 0:
        class_guidance['priorities'].append(
            f"**PRIORITY 2**: {watchlist_count} students on watchlist - monitor closely"
        )
    
    # Pattern-based class strategies
    dominant_pattern = df_with_patterns['persona'].mode()[0]
    class_guidance['dominant_pattern'] = dominant_pattern
    class_guidance['classwide_strategies'] = [
        f"**Differentiate for {dominant_pattern}**: This is your largest group",
        "**Peer Learning**: Leverage diverse patterns for collaborative learning",
        "**Flexible Grouping**: Create dynamic groups based on current needs",
        "**Regular Assessment**: Monitor shifts in patterns over time"
    ]
    
    return class_guidance


def format_recommendations_text(recommendations):
    """
    Format recommendations as readable text.
    
    Args:
        recommendations (dict): Recommendation dictionary
        
    Returns:
        str: Formatted text
    """
    text = f"**Learning Pattern**: {recommendations['pattern']}\n"
    text += f"**Risk Level**: {recommendations['risk_level']} (Priority: {recommendations['priority']})\n\n"
    
    text += "**Recommended Strategies:**\n"
    for i, strategy in enumerate(recommendations['strategies'], 1):
        text += f"{i}. {strategy}\n"
    
    if 'context_specific' in recommendations:
        text += "\n**Additional Context-Specific Actions:**\n"
        for item in recommendations['context_specific']:
            text += f"• {item}\n"
    
    return text


def generate_all_recommendations(df):
    """
    Generate recommendations for all students using optimized operations.
    """
    print("\n" + "="*80)
    print("GENERATING TEACHING GUIDANCE")
    print("="*80)
    
    df_guide = df.copy()
    
    # Use list comprehension for faster generation
    # persona and risk_level are already in df
    zipped_recs = [
        generate_personalized_recommendations(
            row, 
            row.get('persona', 'Mixed Pattern Learners'), 
            row.get('risk_level', 'Normal')
        )
        for _, row in df_guide.iterrows()
    ]
    
    df_guide['recommendations'] = zipped_recs
    df_guide['recommendations_text'] = [format_recommendations_text(r) for r in zipped_recs]
    
    print(f"\n✓ Generated recommendations for {len(df_guide)} students")
    
    return df_guide


if __name__ == "__main__":
    # Test teaching guidance
    from data_processor import process_data
    from pattern_discovery import discover_patterns
    from risk_detection import analyze_all_students
    
    print("="*80)
    print("TEACHING GUIDANCE - DEMO")
    print("="*80)
    
    # Load data
    df, X_normalized, scaler, feature_names = process_data()
    
    # Discover patterns
    labels, kmeans, personas, cluster_profiles = discover_patterns(df, X_normalized, feature_names, n_clusters=4)
    
    # Add patterns to df
    df['cluster'] = labels
    df['persona'] = df['cluster'].map(personas)
    
    # Analyze risk
    df = analyze_all_students(df)
    
    # Generate guidance
    df = generate_all_recommendations(df)
    
    # Show samples
    print("\n" + "="*80)
    print("SAMPLE RECOMMENDATIONS")
    print("="*80)
    
    for i in range(3):
        print(f"\nStudent {i+1}:")
        print(df.iloc[i]['recommendations_text'])
        print("-"*80)
    
    # Class-level guidance
    class_guidance = generate_class_guidance(df)
    print("\n" + "="*80)
    print("CLASS-LEVEL GUIDANCE")
    print("="*80)
    print(f"Total Students: {class_guidance['total_students']}")
    print(f"\nDominant Pattern: {class_guidance['dominant_pattern']}")
    print("\nPriorities:")
    for priority in class_guidance['priorities']:
        print(f"  {priority}")
