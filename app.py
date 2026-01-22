"""
Learning Pattern Analysis & Teaching Guidance System
Interactive Teacher Dashboard

This Streamlit app provides teachers with:
1. Class overview of learning patterns
2. Individual student profiles
3. Pattern explanations
4. Early risk alerts
5. Teaching recommendations
6. Ethical use guidelines

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modules
from data_processor import process_data
from pattern_discovery import discover_patterns
from explainability import generate_student_explanations
from risk_detection import analyze_all_students
from teaching_guidance import generate_all_recommendations, generate_class_guidance

# Plotly chart configuration - Remove modebar and add animations
PLOTLY_CONFIG = {
    'displayModeBar': False,  # Remove toolbar/slider
    'displaylogo': False,
    'staticPlot': False
}

# Page configuration
st.set_page_config(
    page_title="LearnIQ - Learning Pattern Analysis",
    page_icon="📄",  # Simple document icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme with Animations
st.markdown("""
<style>
    /* Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 10px rgba(102, 126, 234, 0.2); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.4); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    /* Main dark background - simpler for performance */
    .main {
        background-color: #0a0a0a;
        background-image: radial-gradient(circle at 50% 50%, #1a1a2e 0%, #0a0a0a 100%);
    }
    
    .block-container {
        background-color: rgba(20, 20, 20, 0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
        border: 1px solid rgba(102, 126, 234, 0.2);
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Updated header style: moved higher and smaller accent line */
    .main-header {
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 0.25rem 0;
        margin-bottom: 0.1rem;
        margin-top: -2rem; /* Move up */
        letter-spacing: 1px;
        text-shadow: none;
        position: relative;
        animation: fadeIn 0.8s ease-out;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px; /* Smaller header line */
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        border-radius: 2px;
    }
    
    /* Metric cards with dark background, neon borders, and animations */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea !important;
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.8);
        animation: fadeIn 0.6s ease-out;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(rgba(20, 20, 20, 0.9), rgba(20, 20, 20, 0.9)) padding-box,
                    linear-gradient(135deg, #667eea, #764ba2) border-box;
        border: 2px solid transparent;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), 
                    box-shadow 0.4s ease;
        animation: slideInLeft 0.8s ease-out;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar dark gradient with slide animation */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
        animation: slideInLeft 0.6s ease-out;
    }
    
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: #e0e0e0 !important;
        transition: all 0.3s ease;
    }
    
    /* Enhanced sidebar radio buttons with premium styling */
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 0.75rem;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #e0e0e0 !important;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    section[data-testid="stSidebar"] .stRadio label::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.6s;
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
        border-color: #667eea;
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover::before {
        left: 100%;
    }
    
    /* Selected radio button styling */
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child {
        background-color: #667eea !important;
        border-color: #667eea !important;
    }
    
    section[data-testid="stSidebar"] .stRadio input:checked + div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Dark info boxes with neon borders and animations */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
        background-color: rgba(30, 30, 30, 0.8) !important;
        box-shadow: 0 2px 15px rgba(102, 126, 234, 0.2);
        color: #e0e0e0 !important;
        animation: slideInRight 0.8s ease-out;
        transition: all 0.3s ease;
    }
    
    .stAlert:hover {
        transform: translateX(3px);
    }
    
    /* Risk level badges with glow and pulse animation */
    .risk-high {
        color: #ff4444;
        font-weight: 800;
        font-size: 1.2rem;
        text-shadow: 0 0 15px rgba(255, 68, 68, 0.8);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .risk-medium {
        color: #ffaa00;
        font-weight: 800;
        font-size: 1.2rem;
        text-shadow: 0 0 15px rgba(255, 170, 0, 0.8);
    }
    
    .risk-normal {
        color: #00ff88;
        font-weight: 800;
        font-size: 1.2rem;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.8);
    }
    
    /* Enhanced tabs for dark mode with animations */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(30, 30, 30, 0.6);
        padding: 10px;
        border-radius: 10px;
        animation: fadeIn 1s ease-out;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: rgba(40, 40, 40, 0.8);
        border-radius: 8px;
        font-weight: 600;
        color: #a0a0a0 !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 0 25px rgba(102, 126, 234, 0.8);
        transform: translateY(-3px) scale(1.05);
    }
    
    /* Neon button styling with animations */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.6);
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton button:hover::before {
        left: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 8px 35px rgba(102, 126, 234, 0.9);
    }
    
    /* Dark section headers with glow and animation */
    h1, h2, h3 {
        color: #667eea !important;
        font-weight: 700;
        margin-top: 2rem;
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
        animation: fadeIn 0.8s ease-out;
        transition: all 0.3s ease;
    }
    
    h2:hover, h3:hover {
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.9);
        transform: translateX(5px);
    }
    
    /* Dark dataframe styling with animation */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.5);
        background-color: rgba(30, 30, 30, 0.8) !important;
        animation: fadeIn 1s ease-out;
        transition: all 0.3s ease;
    }
    
    .dataframe:hover {
        box-shadow: 0 4px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Text colors */
    p, span, div {
        color: #e0e0e0 !important;
    }
    
    /* Selectbox and input styling with animations */
    .stSelectbox, .stMultiSelect {
        color: #e0e0e0 !important;
    }
    
    input, select, textarea {
        background-color: rgba(40, 40, 40, 0.8) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease;
    }
    
    input:focus, select:focus, textarea:focus {
        border-color: rgba(102, 126, 234, 0.8) !important;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.4);
        transform: scale(1.02);
    }
    
    /* Expander styling with animation */
    .streamlit-expanderHeader {
        background-color: rgba(30, 30, 30, 0.8) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(40, 40, 40, 0.9) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Success/Warning/Error boxes with animations */
    .stSuccess {
        background-color: rgba(0, 255, 136, 0.1) !important;
        color: #00ff88 !important;
        border-left: 4px solid #00ff88 !important;
        animation: slideInRight 0.6s ease-out;
    }
    
    .stWarning {
        background-color: rgba(255, 170, 0, 0.1) !important;
        color: #ffaa00 !important;
        border-left: 4px solid #ffaa00 !important;
        animation: slideInRight 0.6s ease-out;
    }
    
    .stError {
        background-color: rgba(255, 68, 68, 0.1) !important;
        color: #ff4444 !important;
        border-left: 4px solid #ff4444 !important;
        animation: slideInRight 0.6s ease-out;
    }
    
    /* Plotly chart animations */
    .js-plotly-plot {
        animation: fadeIn 1.2s ease-out;
        transition: all 0.3s ease;
    }
    
    .js-plotly-plot:hover {
        transform: scale(1.01);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Loading spinner animation */
    .stSpinner > div {
        border-color: #667eea transparent #764ba2 transparent !important;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data ():
    """Load and process all student data with caching."""
    # Process data
    df, X_normalized, scaler, feature_names = process_data()
    
    # Discover patterns
    labels, kmeans, personas, cluster_profiles = discover_patterns(
        df, X_normalized, feature_names, n_clusters=4
    )
    
    # Add patterns
    df['cluster'] = labels
    df['persona'] = df['cluster'].map(personas)
    
    # Generate explanations
    df = generate_student_explanations(df, labels, personas, kmeans, feature_names, cluster_profiles)
    
    # Analyze risk
    df = analyze_all_students(df)
    
    # Generate recommendations
    df = generate_all_recommendations(df)
    
    return df, personas, cluster_profiles


def main():
    # Header
    st.markdown('<div class="main-header">LearnIQ</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data (hide spinner for cleaner UX)
    df, personas, cluster_profiles = load_and_process_data()
    
    # Sidebar Navigation
    st.sidebar.markdown('<p style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem;">Navigation</p>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "navigation",
        ["Class Overview", "Student Profiles", "Risk Alerts", 
         "Teaching Strategies", "About & Ethics"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Total Students**: {len(df)}")
    st.sidebar.success(f"**Learning Patterns**: {len(personas)}")
    
    # Main content based on selection
    if page == "Class Overview":
        show_class_overview(df, personas)
    
    elif page == "Student Profiles":
        show_student_profiles(df)
    
    elif page == "Risk Alerts":
        show_risk_alerts(df)
    
    elif page == "Teaching Strategies":
        show_teaching_strategies(df, personas)
    
    elif page == "About & Ethics":
        show_about_ethics()
        
    # Footer
    st.markdown("---")
    st.markdown('<p style="text-align: center; color: #666; font-size: 0.8rem; margin-top: 2rem;">© 2026 MADTech. All rights reserved.</p>', unsafe_allow_html=True)


def show_class_overview(df, personas):
    """Display class-level overview and statistics."""
    st.header("Class Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    
    with col2:
        avg_engagement = df['engagement_score'].mean()
        st.metric("Avg Engagement", f"{avg_engagement:.1f}/100")
    
    with col3:
        high_risk_count = len(df[df['risk_level'] == 'High Risk'])
        st.metric("High Risk Students", high_risk_count, 
                 delta="Needs Attention" if high_risk_count > 0 else "Good")
    
    with col4:
        avg_grade = df[['G1', 'G2', 'G3']].mean().mean()
        st.metric("Avg Grade", f"{avg_grade:.1f}/20")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Learning Pattern Distribution")
        pattern_dist = df['persona'].value_counts()
        fig = px.pie(
            values=pattern_dist.values,
            names=pattern_dist.index,
            title="Student Learning Patterns",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Students: %{value}<br>Percentage: %{percent}<extra></extra>',
            marker=dict(line=dict(color='#000000', width=2))
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0', size=12),
            hoverlabel=dict(
                bgcolor="#1a1a2e",
                font_size=14,
                font_family="Arial"
            ),
            transition_duration=800  # Smoother animation
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    with col2:
        st.subheader("Risk Level Distribution")
        risk_dist = df['risk_level'].value_counts()
        colors = {
            'Normal': '#00ff88',
            'Watchlist': '#ffaa00', 
            'High Risk': '#ff4444'
        }
        fig = px.bar(
            x=risk_dist.index,
            y=risk_dist.values,
            title="Student Risk Levels",
            labels={'x': 'Risk Level', 'y': 'Number of Students'},
            color=risk_dist.index,
            color_discrete_map=colors
        )
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Students: %{y}<extra></extra>',
            marker=dict(line=dict(color='#000000', width=2))
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0', size=12),
            hoverlabel=dict(
                bgcolor="#1a1a2e",
                font_size=14
            ),
            xaxis=dict(
                showgrid=False,
                color='#e0e0e0'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(102, 126, 234, 0.1)',
                color='#e0e0e0'
            ),
            transition_duration=800  # Smoother animation
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # Pattern characteristics
    st.markdown("---")
    st.subheader("Learning Pattern Characteristics")
    
    # Create comparison chart
    pattern_metrics = df.groupby('persona')[
        ['engagement_score', 'consistency_index', 'participation_stability']
    ].mean().round(1)
    
    fig = go.Figure()
    
    for metric in ['engagement_score', 'consistency_index', 'participation_stability']:
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=pattern_metrics.index,
            y=pattern_metrics[metric]
        ))
    
    fig.update_layout(
        title="Average Behavioral Metrics by Learning Pattern",
        xaxis_title="Learning Pattern",
        yaxis_title="Score (0-100)",
        barmode='group',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        transition_duration=500
    )
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # Class guidance
    st.markdown("---")
    st.subheader("Class-Level Guidance")
    class_guidance = generate_class_guidance(df)
    
    if class_guidance['priorities']:
        st.warning("**Action Items:**")
        for priority in class_guidance['priorities']:
            st.markdown(f"- {priority}")
    
    with st.expander("Class-Wide Teaching Strategies"):
        for strategy in class_guidance['classwide_strategies']:
            st.markdown(f"• {strategy}")


def show_student_profiles(df):
    """Display individual student profiles with search."""
    st.header("Student Profiles")
    
    # Search/Select student
    st.subheader("Select a Student")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create student selector (using index for demo)
        student_idx = st.selectbox(
            "Choose student by index:",
            options=range(len(df)),
            format_func=lambda x: f"Student {x} ({df.iloc[x]['persona']} - {df.iloc[x]['risk_level']})"
        )
    
    with col2:
        # Filter options
        filter_persona = st.selectbox(
            "Filter by Pattern:",
            options=["All"] + list(df['persona'].unique())
        )
    
    # Apply filter
    if filter_persona != "All":
        filtered_df = df[df['persona'] == filter_persona]
        student_idx = st.selectbox(
            "Filtered students:",
            options=filtered_df.index,
            format_func=lambda x: f"Student {x} ({df.loc[x]['risk_level']})"
        )
    
    # Display selected student
    student = df.loc[student_idx]
    
    st.markdown("---")
    
    # Student Overview Card
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric("Learning Pattern", student['persona'])
    
    with col2:
        risk_color = {
            'Normal': 'normal',
            'Watchlist': 'medium',
            'High Risk': 'high'
        }[student['risk_level']]
        st.markdown(f"**Risk Level:** <span class='risk-{risk_color}'>{student['risk_level']}</span>", 
                   unsafe_allow_html=True)
    
    with col3:
        avg_grade = (student['G1'] + student['G2'] + student['G3']) / 3
        st.metric("Average Grade", f"{avg_grade:.1f}/20")
    
    # Behavioral Metrics with Gauge Charts
    st.subheader("Behavioral Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig_engagement = go.Figure(go.Indicator(
            mode="gauge+number",
            value=student['engagement_score'],
            title={'text': "Engagement"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#667eea"},
                   'steps': [
                       {'range': [0, 40], 'color': "#ffcccc"},
                       {'range': [40, 70], 'color': "#fff4cc"},
                       {'range': [70, 100], 'color': "#ccffcc"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}
        ))
        fig_engagement.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10), transition_duration=500)
        st.plotly_chart(fig_engagement, use_container_width=True, config=PLOTLY_CONFIG)
    
    with col2:
        fig_consistency = go.Figure(go.Indicator(
            mode="gauge+number",
            value=student['consistency_index'],
            title={'text': "Consistency"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#764ba2"},
                   'steps': [
                       {'range': [0, 40], 'color': "#ffcccc"},
                       {'range': [40, 70], 'color': "#fff4cc"},
                       {'range': [70, 100], 'color': "#ccffcc"}]}
        ))
        fig_consistency.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10), transition_duration=500)
        st.plotly_chart(fig_consistency, use_container_width=True, config=PLOTLY_CONFIG)
    
    with col3:
        # Trend as delta indicator
        trend_val = (student['performance_trend'] + 1) * 50  # Normalize to 0-100
        trend_emoji = "📈" if student['performance_trend'] > 0 else ("📉" if student['performance_trend'] < 0 else "➡️")
        fig_trend = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=trend_val,
            title={'text': f"Trend {trend_emoji}"},
            delta={'reference': 50},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#667eea"},
                   'steps': [
                       {'range': [0, 40], 'color': "#ffcccc"},
                       {'range': [40, 60], 'color': "#fff4cc"},
                       {'range': [60, 100], 'color': "#ccffcc"}]}
        ))
        fig_trend.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10), transition_duration=500)
        st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_CONFIG)
    
    with col4:
        fig_participation = go.Figure(go.Indicator(
            mode="gauge+number",
            value=student['participation_stability'],
            title={'text': "Participation"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#764ba2"},
                   'steps': [
                       {'range': [0, 40], 'color': "#ffcccc"},
                       {'range': [40, 70], 'color': "#fff4cc"},
                       {'range': [70, 100], 'color': "#ccffcc"}]}
        ))
        fig_participation.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10), transition_duration=500)
        st.plotly_chart(fig_participation, use_container_width=True, config=PLOTLY_CONFIG)
    
    # Radar Chart for Behavioral Profile
    st.subheader("Behavioral Profile Radar")
    
    categories = ['Engagement', 'Consistency', 'Participation', 'Study Time', 'Attendance']
    values = [
        student['engagement_score'],
        student['consistency_index'],
        student['participation_stability'],
        (student['studytime'] / 4) * 100,  # Normalize to 0-100
        max(0, (1 - student['absences'] / 30) * 100)  # Normalize absences to 0-100
    ]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Student Profile',
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        transition_duration=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True, config=PLOTLY_CONFIG)
    
    # Progress Chart
    st.subheader("Grade Progression")
    grades = [student['G1'], student['G2'], student['G3']]
    quarters = ['Q1', 'Q2', 'Q3']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=quarters, y=grades,
        mode='lines+markers',
        line=dict(width=4, color='#667eea'),
        marker=dict(size=15, color='#764ba2'),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    fig.update_layout(
        yaxis_title="Grade (out of 20)",
        yaxis_range=[0, 20],
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        xaxis=dict(color='#e0e0e0'),
        yaxis=dict(color='#e0e0e0', gridcolor='rgba(102, 126, 234, 0.1)'),
        transition_duration=500
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # Explanation
    st.subheader("Pattern Explanation")
    st.info(student['explanation'])
    
    # Risk Assessment
    if student['risk_level'] != "Normal":
        st.subheader("⚠️ Risk Assessment")
        st.warning(student['risk_explanation'])
    
    # Teaching Recommendations
    st.subheader("🎯 Recommended Teaching Strategies")
    st.markdown(student['recommendations_text'])


def show_risk_alerts(df):
    """Display students flagged for early intervention."""
    st.header("⚠️ Early Risk Alerts")
    
    # Filter controls
    col1, col2 = st.columns(2)
    
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level:",
            options=['High Risk', 'Watchlist', 'Normal'],
            default=['High Risk', 'Watchlist']
        )
    
    with col2:
        pattern_filter = st.multiselect(
            "Filter by Pattern:",
            options=df['persona'].unique(),
            default=df['persona'].unique()
        )
    
    # Apply filters
    filtered = df[
        (df['risk_level'].isin(risk_filter)) &
        (df['persona'].isin(pattern_filter))
    ]
    
    st.info(f"Showing **{len(filtered)}** students matching filters")
    
    # Priority students
    if len(filtered[filtered['risk_level'] == 'High Risk']) > 0:
        st.error(f"🚨 **{len(filtered[filtered['risk_level'] == 'High Risk'])} students** require immediate intervention")
    
    # Display table
    st.subheader("Student Alert List")
    
    # Create display dataframe
    display_df = filtered[[
        'persona', 'risk_level', 'engagement_score', 
        'consistency_index', 'absences', 'performance_trend'
    ]].copy()
    
    display_df.columns = ['Pattern', 'Risk', 'Engagement', 'Consistency', 'Absences', 'Trend']
    display_df = display_df.round(1)
    display_df = display_df.sort_values('Risk', ascending=False)
    
    # Color code risk levels with dark theme friendly colors
    def highlight_risk(row):
        if row['Risk'] == 'High Risk':
            return ['background-color: rgba(255, 75, 75, 0.15); color: #ff8585; font-weight: 500'] * len(row)
        elif row['Risk'] == 'Watchlist':
            return ['background-color: rgba(255, 165, 0, 0.15); color: #ffc966; font-weight: 500'] * len(row)
        return ['color: #e0e0e0'] * len(row)
    
    st.dataframe(
        display_df.style.apply(highlight_risk, axis=1).format({
            'Engagement': '{:.1f}',
            'Consistency': '{:.1f}',
            'Trend': '{:.2f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Detailed view
    if len(filtered) > 0:
        st.subheader("Detailed Risk Assessments")
        selected_student = st.selectbox(
            "Select student for details:",
            options=filtered.index,
            format_func=lambda x: f"Student {x} - {df.loc[x]['persona']} ({df.loc[x]['risk_level']})"
        )
        
        student = df.loc[selected_student]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Explanation:**")
            st.warning(student['risk_explanation'])
        
        with col2:
            st.markdown("**Recommended Actions:**")
            recs = student['recommendations']
            if 'strategies' in recs:
                for strategy in recs['strategies'][:5]:  # Show top 5
                    st.markdown(f"• {strategy}")


def show_teaching_strategies(df, personas):
    """Display teaching strategies by pattern."""
    st.header("🎯 Teaching Strategies")
    
    # Pattern selector
    selected_pattern = st.selectbox(
        "Select Learning Pattern:",
        options=list(personas.values())
    )
    
    # Get students with this pattern
    pattern_students = df[df['persona'] == selected_pattern]
    
    st.info(f"**{len(pattern_students)} students** identified as **{selected_pattern}**")
    
    # Strategy overview
    st.subheader(f"Strategies for {selected_pattern}")
    
    if len(pattern_students) > 0:
        sample_student = pattern_students.iloc[0]
        recs = sample_student['recommendations']
        
        if 'strategies' in recs:
            for i, strategy in enumerate(recs['strategies'], 1):
                st.markdown(f"{i}. {strategy}")
    
    # Pattern characteristics
    st.markdown("---")
    st.subheader("Pattern Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics for this pattern
        avg_engagement = pattern_students['engagement_score'].mean()
        avg_consistency = pattern_students['consistency_index'].mean()
        avg_grade = pattern_students[['G1', 'G2', 'G3']].mean().mean()
        
        st.metric("Average Engagement", f"{avg_engagement:.1f}/100")
        st.metric("Average Consistency", f"{avg_consistency:.1f}/100")
        st.metric("Average Grade", f"{avg_grade:.1f}/20")
    
    with col2:
        # Risk distribution for this pattern
        risk_dist = pattern_students['risk_level'].value_counts()
        fig = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title=f"Risk Distribution - {selected_pattern}",
            color_discrete_map={
                'Normal': '#2ca02c',
                'Watchlist': '#ff7f0e',
                'High Risk': '#d62728'
            }
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            transition_duration=500
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # Example students
    st.markdown("---")
    st.subheader("Example Students")
    
    if len(pattern_students) > 0:
        for idx in pattern_students.head(3).index:
            student = df.loc[idx]
            with st.expander(f"Student {idx} - {student['risk_level']}"):
                st.markdown(student['recommendations_text'])


def show_about_ethics():
    """Display information about the system and ethical guidelines."""
    st.header("About & Ethical Guidelines")
    
    st.subheader("About This System")
    st.markdown("""
    The **Learning Pattern Analysis & Teaching Guidance System** is a machine learning-powered tool designed to help 
    educators understand student learning behaviors and adapt teaching strategies accordingly.
    
    **Key Features:**
    - **Pattern Discovery**: Identifies natural groupings of learning behaviors
    - **Transparent Classification**: Provides clear reasoning for all classifications
    - **Early Warning System**: Detects signs of disengagement before grades drop
    - **Personalized Guidance**: Suggests evidence-based teaching interventions
    - **Class-Level Insights**: Offers overview of overall class dynamics
    """)
    
    st.markdown("---")
    
    st.subheader("Ethical Use Guidelines")
    st.warning("""
    **IMPORTANT: This system is designed to ASSIST teacher judgment, not replace it.**
    
    **Ethical Principles:**
    
    1. **No Permanent Labels**: Learning patterns are dynamic snapshots, not fixed categorizations
    
    2. **Fairness & Privacy**: 
       - No demographic attributes (gender, address, family background) used in clustering
       - All analyses focus on observable behaviors and performance
       - Results should be kept confidential
    
    3. **Growth-Oriented**: 
       - All recommendations are non-punitive
       - Focus on support and development, not judgment
       - Emphasize student potential and improvement
    
    4. **Teacher Autonomy**: 
       - System provides suggestions, teachers make final decisions
       - Use professional judgment to contextualize findings
       - Consider factors the system cannot see (student circumstances, recent events, etc.)
    
    5. **Regular Review**: 
       - Patterns should be reassessed periodically
       - Students can move between patterns
       - System is a starting point for investigation, not a final diagnosis
    
    6. **Transparency**: 
       - Students and parents should understand how the system works
       - Explanations can be shared to support student growth
       - Open dialogue about learning patterns is encouraged
    """)
    
    st.markdown("---")
    
    st.subheader("Technical Details")
    st.markdown("""
    **Methodology:**
    - **Clustering Algorithm**: K-Means with optimal k determination via Silhouette Score
    - **Behavioral Features**: Engagement Score, Consistency Index, Performance Trend, Participation Stability
    - **Risk Detection**: Multi-factor rule-based system
    - **Recommendations**: Evidence-based strategies from educational research
    
    **Data Processing:**
    - Missing values handled via median/mode imputation
    - Features normalized using StandardScaler
    - No student identifiers required for analysis
    
    **Interpretability:**
    - Feature importance calculated via deviation from cluster centroids
    - Plain English explanations generated for all assignments
    - Visual dashboards for easy comprehension
    """)
    
    st.markdown("---")
    
    st.subheader("Feedback & Support")
    st.info("""
    This system is designed for educators. Your feedback helps improve it!
    
    **For Questions or Suggestions:**
    - Consult your school's data privacy officer
    - Participate in teacher training sessions
    - Share insights on what works in your classroom
    """)
    
    st.success("**Remember**: You are the expert on your students. Use this tool to enhance, not replace, your professional judgment.")


if __name__ == "__main__":
    main()
