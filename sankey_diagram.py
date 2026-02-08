"""
Sankey Diagram Generator for LearnIQ
Visualizes behavioral flow patterns
"""

import plotly.graph_objects as go
import pandas as pd


def create_behavioral_sankey(df):
    """
    Create a Sankey diagram showing the flow from behaviors to outcomes
    
    Args:
        df: Student dataset with behavioral features
    
    Returns:
        Plotly figure object
    """
    # Define behavioral categories
    df = df.copy()
    
    # Categorize study time
    df['study_category'] = pd.cut(
        df['studytime'],
        bins=[0, 2, 3, 5],
        labels=['Low Study Time', 'Medium Study Time', 'High Study Time']
    )
    
    # Categorize absences
    df['absence_category'] = pd.cut(
        df['absences'],
        bins=[-1, 5, 15, 100],
        labels=['Low Absences', 'Medium Absences', 'High Absences']
    )
    
    # Categorize final performance
    df['performance_category'] = pd.cut(
        df['G3'],
        bins=[0, 10, 14, 20],
        labels=['At Risk', 'Average', 'Excellent']
    )
    
    # Build Sankey node list
    nodes = [
        # Study categories (0-2)
        'Low Study Time', 'Medium Study Time', 'High Study Time',
        # Absence categories (3-5)
        'Low Absences', 'Medium Absences', 'High Absences',
        # Performance outcomes (6-8)
        'At Risk', 'Average', 'Excellent'
    ]
    
    # Build flows
    sources = []
    targets = []
    values = []
    
    # Study Time → Absences
    for i, study in enumerate(['Low Study Time', 'Medium Study Time', 'High Study Time']):
        for j, absence in enumerate(['Low Absences', 'Medium Absences', 'High Absences']):
            count = len(df[(df['study_category'] == study) & (df['absence_category'] == absence)])
            if count > 0:
                sources.append(i)
                targets.append(j + 3)
                values.append(count)
    
    # Absences → Performance
    for i, absence in enumerate(['Low Absences', 'Medium Absences', 'High Absences']):
        for j, perf in enumerate(['At Risk', 'Average', 'Excellent']):
            count = len(df[(df['absence_category'] == absence) & (df['performance_category'] == perf)])
            if count > 0:
                sources.append(i + 3)
                targets.append(j + 6)
                values.append(count)
    
    # Define colors
    node_colors = [
        'rgba(255, 99, 71, 0.8)',    # Low Study - red
        'rgba(255, 165, 0, 0.8)',    # Medium Study - orange
        'rgba(50, 205, 50, 0.8)',    # High Study - green
        'rgba(50, 205, 50, 0.8)',    # Low Absences - green
        'rgba(255, 165, 0, 0.8)',    # Medium Absences - orange
        'rgba(255, 99, 71, 0.8)',    # High Absences - red
        'rgba(220, 20, 60, 0.8)',    # At Risk - dark red
        'rgba(255, 215, 0, 0.8)',    # Average - gold
        'rgba(0, 128, 0, 0.8)'       # Excellent - dark green
    ]
    
    link_colors = []
    for src, tgt in zip(sources, targets):
        if tgt >= 6:  # Performance nodes
            link_colors.append(node_colors[tgt].replace('0.8', '0.3'))
        else:
            link_colors.append(node_colors[src].replace('0.8', '0.3'))
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=2),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title={
            'text': "Behavioral Flow: Study Habits → Attendance → Performance",
            'font': {'size': 20, 'color': '#A8A2FF', 'family': 'Arial'}
        },
        font=dict(size=12, color='white', family='Arial'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_risk_flow_sankey(df, pattern_col='pattern', risk_col='risk_level'):
    """
    Create a Sankey showing pattern → risk → outcome flow
    
    Args:
        df: Student dataset with pattern and risk columns
        pattern_col: Name of the pattern column
        risk_col: Name of the risk level column
    
    Returns:
        Plotly figure object
    """
    df = df.copy()
    
    # Categorize final performance
    df['outcome'] = pd.cut(
        df['G3'],
        bins=[0, 10, 14, 20],
        labels=['Below Average', 'Average', 'Above Average']
    )
    
    # Get unique patterns and risk levels
    patterns = df[pattern_col].unique().tolist()
    risks = df[risk_col].unique().tolist()
    outcomes = ['Below Average', 'Average', 'Above Average']
    
    # Build node list
    nodes = patterns + risks + outcomes
    
    # Build flows
    sources = []
    targets = []
    values = []
    
    # Patterns → Risk Levels
    for i, pattern in enumerate(patterns):
        for j, risk in enumerate(risks):
            count = len(df[(df[pattern_col] == pattern) & (df[risk_col] == risk)])
            if count > 0:
                sources.append(i)
                targets.append(len(patterns) + j)
                values.append(count)
    
    # Risk Levels → Outcomes
    for i, risk in enumerate(risks):
        for j, outcome in enumerate(outcomes):
            count = len(df[(df[risk_col] == risk) & (df['outcome'] == outcome)])
            if count > 0:
                sources.append(len(patterns) + i)
                targets.append(len(patterns) + len(risks) + j)
                values.append(count)
    
    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=2),
            label=nodes,
            color='rgba(168, 162, 255, 0.8)'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(168, 162, 255, 0.3)'
        )
    )])
    
    fig.update_layout(
        title={
            'text': "Learning Pattern → Risk Level → Academic Outcome",
            'font': {'size': 20, 'color': '#A8A2FF', 'family': 'Arial'}
        },
        font=dict(size=12, color='white', family='Arial'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig
