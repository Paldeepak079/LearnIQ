"""
Pattern Discovery Module for Learning Pattern Analysis System

This module implements:
1. Optimal cluster number determination (Elbow + Silhouette)
2. KMeans clustering
3. Human-readable persona assignment
4. Cluster validation

Educational Goal:
- Discover natural learning behavior groupings
- Assign interpretable labels that teachers can understand
- Ensure patterns reflect behavior, not just grades
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def determine_optimal_clusters(X, k_range=range(2, 9), show_plot=True):
    """
    Determine optimal number of clusters using Elbow Method and Silhouette Score.
    
    Methods:
    1. Elbow Method: Find "elbow" in inertia curve
    2. Silhouette Score: Measure cluster separation quality
    
    Args:
        X (pd.DataFrame): Normalized feature matrix
        k_range (range): Range of k values to test
        show_plot (bool): Whether to display diagnostic plots
        
    Returns:
        int: Optimal number of clusters
        dict: Diagnostic metrics for each k
    """
    print("\n" + "="*80)
    print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
    print("="*80)
    
    inertias = []
    silhouette_scores = []
    db_scores = []
    
    for k in k_range:
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))
        
        print(f"  k={k}: Silhouette={silhouette_scores[-1]:.3f}, DB Index={db_scores[-1]:.3f}")
    
    # Find optimal k using silhouette score (higher is better)
    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    
    print(f"\n✓ Optimal clusters: k={optimal_k}")
    print(f"  Silhouette Score: {max(silhouette_scores):.3f}")
    
    # Visualization
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
        axes[0].legend()
        
        # Silhouette plot
        axes[1].plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('cluster_optimization.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved diagnostics: cluster_optimization.png")
        plt.close()
    
    metrics = {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'db_scores': db_scores,
        'optimal_k': optimal_k
    }
    
    return optimal_k, metrics


def perform_clustering(X, n_clusters, random_state=42):
    """
    Perform KMeans clustering on student data.
    
    Args:
        X (pd.DataFrame): Normalized feature matrix
        n_clusters (int): Number of clusters
        random_state (int): Random seed for reproducibility
        
    Returns:
        np.array: Cluster labels for each student
        KMeans: Fitted KMeans model
    """
    print("\n" + "="*80)
    print(f"PERFORMING KMEANS CLUSTERING (k={n_clusters})")
    print("="*80)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X)
    
    # Cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster Distribution:")
    for cluster_id, count in zip(unique, counts):
        pct = (count / len(labels)) * 100
        print(f"  Cluster {cluster_id}: {count} students ({pct:.1f}%)")
    
    print(f"\n✓ Clustering complete")
    
    return labels, kmeans


def assign_personas(df, labels, feature_names):
    """
    Assign human-readable persona names to clusters based on behavioral profiles.
    
    Strategy:
    - Analyze mean feature values for each cluster
    - Identify distinguishing characteristics
    - Assign descriptive, teacher-friendly names
    
    Possible Personas:
    - "Consistent Achievers": High engagement, high consistency, good grades
    - "Silent Performers": Low engagement metrics but good grades
    - "At-Risk Learners": Low engagement, declining trend, high absences
    - "Irregular Learners": Low consistency, volatile performance
    - "Emerging Stars": Positive trend, improving engagement
    - "Disengaged Strugglers": Low across all metrics
    
    Args:
        df (pd.DataFrame): Student data with engineered features
        labels (np.array): Cluster assignments
        feature_names (list): Features used for clustering
        
    Returns:
        dict: Mapping of cluster ID to persona name
        pd.DataFrame: Cluster profiles (mean values)
    """
    print("\n" + "="*80)
    print("ASSIGNING LEARNING PERSONAS")
    print("="*80)
    
    # Add cluster labels to dataframe
    df['cluster'] = labels
    
    # Calculate mean values for each cluster
    behavioral_features = ['engagement_score', 'consistency_index', 'performance_trend', 'participation_stability']
    # Only include features that exist in both lists
    features_to_use = [f for f in feature_names if f in df.columns]
    all_features = list(set(features_to_use + behavioral_features))
    cluster_profiles = df.groupby('cluster')[[f for f in all_features if f in df.columns]].mean()
    
    # Also calculate grade average
    cluster_profiles['avg_grade'] = df.groupby('cluster')[['G1', 'G2', 'G3']].mean().mean(axis=1)
    
    personas = {}
    
    for cluster_id in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster_id]
        
        # Extract key metrics
        engagement = profile['engagement_score']
        consistency = profile['consistency_index']
        trend = profile['performance_trend']
        participation = profile['participation_stability']
        avg_grade = profile['avg_grade']
        absences = profile['absences']
        
        # Decision logic for persona assignment
        # (Based on educational research on learning patterns)
        
        # Convert to scalar values for comparison
        avg_grade_val = float(avg_grade)
        consistency_val = float(consistency)
        engagement_val = float(engagement)
        trend_val = float(trend)
        participation_val = float(participation)
        
        if avg_grade_val >= 12 and consistency_val >= 75 and engagement_val >= 50:
            # High performers with stable patterns
            personas[cluster_id] = "Consistent Achievers"
        
        elif avg_grade_val >= 12 and consistency_val >= 70 and engagement_val < 50:
            # Good grades but low visible engagement
            personas[cluster_id] = "Silent Performers"
        
        elif trend_val > 0.1 and avg_grade_val < 12:
            # Improving students
            personas[cluster_id] = "Emerging Stars"
        
        elif avg_grade_val < 10 and (engagement_val < 40 or participation_val < 40):
            # Low grades with concerning engagement/participation
            personas[cluster_id] = "At-Risk Learners"
        
        elif consistency_val < 60:
            # High variability in performance
            personas[cluster_id] = "Irregular Learners"
        
        elif trend_val < -0.1:
            # Declining performance
            personas[cluster_id] = "Declining Performers"
        
        else:
            # Default catch-all
            personas[cluster_id] = "Mixed Pattern Learners"
        
        # Print persona summary
        print(f"\nCluster {cluster_id}: {personas[cluster_id]}")
        print(f"  • Avg Grade: {avg_grade:.1f}/20")
        print(f"  • Engagement: {engagement:.1f}/100")
        print(f"  • Consistency: {consistency:.1f}/100")
        print(f"  • Trend: {trend:.3f}")
        print(f"  • Participation: {participation:.1f}/100")
        print(f"  • Absences: {absences:.1f}")
    
    print("\n✓ Personas assigned")
    
    return personas, cluster_profiles


def visualize_clusters(df, labels, personas, feature_names):
    """
    Create visualizations of cluster patterns.
    
    Visualizations:
    1. Cluster distribution pie chart
    2. Feature comparison radar/bar charts
    3. 2D cluster scatter plot (using PCA if needed)
    
    Args:
        df (pd.DataFrame): Student data with features
        labels (np.array): Cluster assignments
        personas (dict): Cluster ID to persona name mapping
        feature_names (list): Features used for clustering
    """
    df_viz = df.copy()
    df_viz['cluster'] = labels
    df_viz['persona'] = df_viz['cluster'].map(personas)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Cluster Distribution Pie Chart
    ax1 = plt.subplot(2, 3, 1)
    cluster_counts = df_viz['persona'].value_counts()
    colors = sns.color_palette('Set2', n_colors=len(cluster_counts))
    ax1.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Learning Pattern Distribution', fontsize=14, fontweight='bold')
    
    # 2. Engagement Score by Cluster
    ax2 = plt.subplot(2, 3, 2)
    sns.boxplot(data=df_viz, x='persona', y='engagement_score', palette='Set2', ax=ax2)
    ax2.set_title('Engagement by Pattern', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('Engagement Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Consistency by Cluster
    ax3 = plt.subplot(2, 3, 3)
    sns.boxplot(data=df_viz, x='persona', y='consistency_index', palette='Set2', ax=ax3)
    ax3.set_title('Consistency by Pattern', fontsize=12, fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_ylabel('Consistency Index')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Performance Trend by Cluster
    ax4 = plt.subplot(2, 3, 4)
    sns.boxplot(data=df_viz, x='persona', y='performance_trend', palette='Set2', ax=ax4)
    ax4.set_title('Performance Trend by Pattern', fontsize=12, fontweight='bold')
    ax4.set_xlabel('')
    ax4.set_ylabel('Trend')
    ax4.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Average Grades by Cluster
    ax5 = plt.subplot(2, 3, 5)
    df_viz['avg_grade'] = df_viz[['G1', 'G2', 'G3']].mean(axis=1)
    persona_avg_grades = df_viz.groupby('persona')['avg_grade'].mean().sort_values(ascending=False)
    sns.barplot(x=persona_avg_grades.values, y=persona_avg_grades.index, palette='Set2', ax=ax5)
    ax5.set_title('Average Grade by Pattern', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Average Grade (G1, G2, G3)')
    ax5.set_ylabel('')
    
    # 6. Absences by Cluster
    ax6 = plt.subplot(2, 3, 6)
    sns.boxplot(data=df_viz, x='persona', y='absences', palette='Set2', ax=ax6)
    ax6.set_title('Absences by Pattern', fontsize=12, fontweight='bold')
    ax6.set_xlabel('')
    ax6.set_ylabel('Absences')
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('cluster_patterns.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: cluster_patterns.png")
    plt.close()


# ============================================================================
# Main Pipeline Function
# ============================================================================

def discover_patterns(df, X_normalized, feature_names, n_clusters=None):
    """
    Complete pattern discovery pipeline.
    
    Args:
        df (pd.DataFrame): Full student dataset
        X_normalized (pd.DataFrame): Normalized features for clustering
        feature_names (list): Names of features used
        n_clusters (int): Number of clusters (if None, auto-determine)
        
    Returns:
        tuple: (labels, kmeans_model, personas, cluster_profiles)
    """
    # Determine optimal clusters if not specified
    if n_clusters is None:
        n_clusters, metrics = determine_optimal_clusters(X_normalized)
    
    # Perform clustering
    labels, kmeans = perform_clustering(X_normalized, n_clusters)
    
    # Assign personas
    personas, cluster_profiles = assign_personas(df, labels, feature_names)
    
    # Visualize
    visualize_clusters(df, labels, personas, feature_names)
    
    return labels, kmeans, personas, cluster_profiles


if __name__ == "__main__":
    # Test the pattern discovery
    from data_processor import process_data
    
    print("="*80)
    print("LEARNING PATTERN DISCOVERY - DEMO")
    print("="*80)
    
    # Load and process data
    df, X_normalized, scaler, feature_names = process_data()
    
    # Discover patterns
    labels, kmeans, personas, cluster_profiles = discover_patterns(df, X_normalized, feature_names)
    
    print("\n" + "="*80)
    print("✓ PATTERN DISCOVERY COMPLETE")
    print("="*80)
    
    # Show sample assignments
    df['cluster'] = labels
    df['persona'] = df['cluster'].map(personas)
    print("\n📊 Sample student assignments:")
    print(df[['persona', 'engagement_score', 'consistency_index', 'performance_trend', 'participation_stability']].head(10))
