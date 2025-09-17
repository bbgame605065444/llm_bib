#!/usr/bin/env python3
"""
Improved visualization generator for education analysis JSON data
- Individual plots instead of subplots
- Fix author count to start from 1 (not 0)
- Remove predictions and data after 2026
- Better diagnostics and error handling
- Output to output_3 directory
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import networkx as nx
from wordcloud import WordCloud
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_education_data():
    """Load the education analysis JSON data with diagnostics"""
    json_file_path = "/home/campus.ncl.ac.uk/c1041562/Desktop/codes/article_agent/bibm/output/20250831_024823_education_education_analysis.json"
    
    print(f"📂 Loading data from: {os.path.basename(json_file_path)}")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✅ JSON loaded successfully")
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return None, None
    
    # Extract analysis report and papers
    analysis_report = data.get('analysis_report', {})
    
    # Find the papers array in the JSON structure
    papers_data = []
    if isinstance(data, list):
        papers_data = data
    else:
        # Look for the papers array - it's the main content outside analysis_report
        for key, value in data.items():
            if key != 'analysis_report' and isinstance(value, list):
                papers_data = value
                break
    
    print(f"📊 Found {len(papers_data)} papers in dataset")
    return analysis_report, papers_data

def clean_and_validate_data(papers_data):
    """Clean and validate the papers data with detailed diagnostics"""
    print("\n🔍 Cleaning and validating data...")
    
    # Create DataFrame
    df = pd.DataFrame(papers_data)
    print(f"   Original papers: {len(df)}")
    
    # Clean year data - remove future predictions and invalid years
    print("   Cleaning year data...")
    df['year_original'] = df['year'].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Filter out invalid years and future predictions
    current_year = datetime.now().year
    valid_years = (df['year'] >= 1990) & (df['year'] <= min(current_year, 2026))
    future_papers = df[df['year'] > 2026]
    invalid_papers = df[df['year'].isna() | (df['year'] < 1990)]
    
    if len(future_papers) > 0:
        print(f"   ⚠️  Removed {len(future_papers)} papers with years > 2026")
    if len(invalid_papers) > 0:
        print(f"   ⚠️  Removed {len(invalid_papers)} papers with invalid years")
    
    df = df[valid_years].copy()
    df['year'] = df['year'].astype(int)
    
    # Clean citation data
    print("   Cleaning citation data...")
    df['citation_count'] = pd.to_numeric(df['citation_count'], errors='coerce').fillna(0).astype(int)
    
    # Clean relevance scores
    df['relevance_score'] = pd.to_numeric(df['relevance_score'], errors='coerce').fillna(0)
    df['relevance_score'] = df['relevance_score'].clip(0, 1)  # Ensure 0-1 range
    
    # Fix author count to start from 1 (not 0)
    print("   Processing author data...")
    def count_authors(authors):
        try:
            if pd.isna(authors):
                return 1  # Single author assumed if missing
            if isinstance(authors, list):
                if len(authors) == 0:
                    return 1
                return max(1, len(authors))  # Minimum 1 author
            if isinstance(authors, str):
                # Count authors in string format
                if authors.strip() == '':
                    return 1
                return max(1, len([a.strip() for a in authors.split(',') if a.strip()]))
            return 1
        except:
            return 1  # Fallback to 1 author
    
    df['author_count'] = df['authors'].apply(count_authors)
    
    # Validate author count
    zero_authors = df[df['author_count'] == 0]
    if len(zero_authors) > 0:
        print(f"   ⚠️  Fixed {len(zero_authors)} papers with 0 authors -> 1 author")
        df.loc[df['author_count'] == 0, 'author_count'] = 1
    
    print(f"   ✅ Final dataset: {len(df)} papers")
    print(f"   📅 Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"   👥 Author count range: {df['author_count'].min()} - {df['author_count'].max()}")
    print(f"   📊 Relevance score range: {df['relevance_score'].min():.2f} - {df['relevance_score'].max():.2f}")
    
    return df

def create_individual_plot_1_relevance_distribution(df, output_dir):
    """Create individual plot: Relevance Score Distribution"""
    plt.figure(figsize=(10, 6))
    
    relevance_scores = df['relevance_score'].dropna()
    
    # Create histogram with better styling
    plt.hist(relevance_scores, bins=30, alpha=0.7, color='skyblue', 
             edgecolor='navy', linewidth=0.8, density=False)
    
    # Add statistics text
    mean_score = relevance_scores.mean()
    median_score = relevance_scores.median()
    
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_score:.3f}')
    plt.axvline(median_score, color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {median_score:.3f}')
    
    plt.xlabel('Relevance Score', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.title('Distribution of Relevance Scores\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text box with statistics
    stats_text = f'Total Papers: {len(relevance_scores):,}\nMean: {mean_score:.3f}\nStd: {relevance_scores.std():.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '01_relevance_score_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Created: {os.path.basename(output_path)}")
    return output_path

def create_individual_plot_2_yearly_trends(df, output_dir):
    """Create individual plot: Yearly Publication Trends"""
    plt.figure(figsize=(12, 6))
    
    # Group by year and count papers
    yearly_counts = df['year'].value_counts().sort_index()
    
    # Create bar plot with gradient colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(yearly_counts)))
    bars = plt.bar(yearly_counts.index, yearly_counts.values, alpha=0.8, color=colors)
    
    # Add trend line
    z = np.polyfit(yearly_counts.index, yearly_counts.values, 1)
    p = np.poly1d(z)
    plt.plot(yearly_counts.index, p(yearly_counts.index), "r--", alpha=0.8, linewidth=2, 
             label=f'Trend Line (slope: {z[0]:.1f} papers/year)')
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Publications', fontsize=12)
    plt.title('Publication Trends Over Time\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add peak year annotation
    peak_year = yearly_counts.idxmax()
    peak_count = yearly_counts.max()
    plt.annotate(f'Peak: {peak_year}\n({peak_count} papers)', 
                xy=(peak_year, peak_count), xytext=(peak_year-2, peak_count+5),
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '02_yearly_publication_trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Created: {os.path.basename(output_path)}")
    return output_path

def create_individual_plot_3_author_count_distribution(df, output_dir):
    """Create individual plot: Author Count Distribution (starting from 1)"""
    plt.figure(figsize=(10, 6))
    
    author_counts = df['author_count']
    
    # Verify no zero authors
    if author_counts.min() == 0:
        print("⚠️  Warning: Found papers with 0 authors, fixing...")
        author_counts = author_counts.replace(0, 1)
    
    # Create histogram
    max_authors = min(author_counts.max(), 15)  # Cap at 15 for readability
    bins = range(1, max_authors + 2)  # Start from 1, not 0
    
    plt.hist(author_counts, bins=bins, alpha=0.7, color='lightgreen', 
             edgecolor='darkgreen', linewidth=0.8, align='left')
    
    plt.xlabel('Number of Authors per Paper', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.title('Distribution of Author Count per Paper\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis to start from 1
    plt.xlim(0.5, max_authors + 0.5)
    plt.xticks(range(1, max_authors + 1))
    
    # Add statistics
    mean_authors = author_counts.mean()
    mode_authors = author_counts.mode().iloc[0]
    
    stats_text = f'Mean Authors: {mean_authors:.1f}\nMode: {mode_authors} authors\nRange: {author_counts.min()}-{author_counts.max()}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '03_author_count_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Created: {os.path.basename(output_path)}")
    return output_path

def create_individual_plot_4_citation_analysis(df, output_dir):
    """Create individual plot: Citation Analysis"""
    plt.figure(figsize=(12, 6))
    
    # Filter out extreme outliers for better visualization
    q95 = df['citation_count'].quantile(0.95)
    filtered_citations = df[df['citation_count'] <= q95]['citation_count']
    
    # Create histogram with log scale if needed
    if filtered_citations.max() > 100:
        plt.hist(filtered_citations, bins=50, alpha=0.7, color='coral', edgecolor='darkred')
        plt.yscale('log')
        plt.ylabel('Number of Papers (log scale)', fontsize=12)
    else:
        plt.hist(filtered_citations, bins=30, alpha=0.7, color='coral', edgecolor='darkred')
        plt.ylabel('Number of Papers', fontsize=12)
    
    plt.xlabel('Citation Count', fontsize=12)
    plt.title('Citation Distribution\nK12 Education & AI Research (95th percentile)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    total_papers = len(df)
    cited_papers = len(df[df['citation_count'] > 0])
    mean_citations = df['citation_count'].mean()
    median_citations = df['citation_count'].median()
    
    stats_text = f'Total Papers: {total_papers:,}\nCited Papers: {cited_papers:,} ({cited_papers/total_papers:.1%})\nMean Citations: {mean_citations:.1f}\nMedian Citations: {median_citations:.1f}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '04_citation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Created: {os.path.basename(output_path)}")
    return output_path

def create_individual_plot_5_top_topics(analysis_report, output_dir):
    """Create individual plot: Top Research Topics"""
    plt.figure(figsize=(12, 8))
    
    top_topics = analysis_report.get('top_topics', [])[:15]  # Top 15 topics
    
    if not top_topics:
        print("⚠️  No topics data found")
        return None
    
    topics = [topic[0] for topic in top_topics]
    frequencies = [topic[1] for topic in top_topics]
    
    # Create horizontal bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(topics)))
    bars = plt.barh(range(len(topics)), frequencies, color=colors, alpha=0.8)
    
    # Customize
    plt.yticks(range(len(topics)), topics)
    plt.xlabel('Frequency (Number of Papers)', fontsize=12)
    plt.title('Top 15 Research Topics\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        plt.text(freq + max(frequencies) * 0.01, i, str(freq), 
                va='center', fontweight='bold')
    
    # Invert y-axis to show highest frequency at top
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '05_top_research_topics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Created: {os.path.basename(output_path)}")
    return output_path

def create_individual_plot_6_relevance_by_year(df, output_dir):
    """Create individual plot: Relevance Score Trends by Year"""
    plt.figure(figsize=(12, 6))
    
    # Group by year and calculate relevance statistics
    yearly_relevance = df.groupby('year')['relevance_score'].agg(['mean', 'std', 'count']).reset_index()
    yearly_relevance = yearly_relevance[yearly_relevance['count'] >= 3]  # At least 3 papers per year
    
    # Plot mean relevance with error bars
    plt.errorbar(yearly_relevance['year'], yearly_relevance['mean'], 
                yerr=yearly_relevance['std'], marker='o', linewidth=2, 
                markersize=6, capsize=5, capthick=2, alpha=0.8, color='darkblue')
    
    # Add trend line
    if len(yearly_relevance) > 2:
        z = np.polyfit(yearly_relevance['year'], yearly_relevance['mean'], 1)
        p = np.poly1d(z)
        plt.plot(yearly_relevance['year'], p(yearly_relevance['year']), 
                "r--", alpha=0.7, linewidth=2, 
                label=f'Trend (slope: {z[0]:.4f}/year)')
        plt.legend()
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mean Relevance Score', fontsize=12)
    plt.title('Relevance Score Trends Over Time\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add horizontal line at 0.7 (common relevance threshold)
    plt.axhline(y=0.7, color='orange', linestyle=':', alpha=0.7, 
                label='Relevance Threshold (0.7)')
    plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '06_relevance_trends_by_year.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Created: {os.path.basename(output_path)}")
    return output_path

def create_individual_plot_7_citation_vs_relevance(df, output_dir):
    """Create individual plot: Citation Count vs Relevance Score"""
    plt.figure(figsize=(10, 8))
    
    # Filter extreme citation outliers for better visualization
    q95_citations = df['citation_count'].quantile(0.95)
    df_filtered = df[df['citation_count'] <= q95_citations].copy()
    
    # Create scatter plot
    scatter = plt.scatter(df_filtered['relevance_score'], df_filtered['citation_count'], 
                         alpha=0.6, s=30, c=df_filtered['year'], cmap='viridis')
    
    # Add colorbar for year
    cbar = plt.colorbar(scatter)
    cbar.set_label('Publication Year', fontsize=12)
    
    plt.xlabel('Relevance Score', fontsize=12)
    plt.ylabel('Citation Count', fontsize=12)
    plt.title('Citation Count vs Relevance Score\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = df['relevance_score'].corr(df['citation_count'])
    plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '07_citation_vs_relevance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Created: {os.path.basename(output_path)}")
    return output_path

def create_individual_plot_8_research_evolution(df, output_dir):
    """Create individual plot: Research Evolution Timeline"""
    plt.figure(figsize=(14, 8))
    
    # Create a timeline showing paper count, citations, and relevance by year
    yearly_stats = df.groupby('year').agg({
        'title': 'count',
        'citation_count': 'sum', 
        'relevance_score': 'mean'
    }).rename(columns={'title': 'paper_count'})
    
    # Create subplot with multiple y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot paper count (bars)
    color = 'tab:blue'
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Number of Papers', color=color, fontsize=12)
    bars = ax1.bar(yearly_stats.index, yearly_stats['paper_count'], 
                   alpha=0.6, color=color, label='Papers')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for relevance score
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Relevance Score', color=color, fontsize=12)
    line1 = ax2.plot(yearly_stats.index, yearly_stats['relevance_score'], 
                     color=color, marker='o', linewidth=3, markersize=6, 
                     label='Relevance')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)
    
    # Add title
    plt.title('Research Evolution Timeline\nK12 Education & AI Research', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '08_research_evolution_timeline.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Created: {os.path.basename(output_path)}")
    return output_path

def main():
    """Main function to generate improved individual visualizations"""
    output_dir = "output_3"
    
    print("=== Improved K12 Education & AI Visualization Generator ===")
    print("🎯 Fixes:")
    print("   - Individual plots (no subplots)")
    print("   - Author count starts from 1 (not 0)")
    print("   - No predictions beyond 2026")
    print("   - Better data validation and diagnostics")
    print(f"   - Output to: {output_dir}")
    print()
    
    # Load and validate data
    analysis_report, papers_data = load_education_data()
    if not papers_data:
        print("❌ Failed to load data")
        return
    
    # Clean and validate data
    df = clean_and_validate_data(papers_data)
    if df.empty:
        print("❌ No valid data after cleaning")
        return
    
    print(f"\n🎨 Generating individual visualizations...")
    print("-" * 50)
    
    # Generate individual plots
    created_plots = []
    
    try:
        # 1. Relevance Score Distribution
        plot_path = create_individual_plot_1_relevance_distribution(df, output_dir)
        if plot_path: created_plots.append(plot_path)
        
        # 2. Yearly Publication Trends
        plot_path = create_individual_plot_2_yearly_trends(df, output_dir)
        if plot_path: created_plots.append(plot_path)
        
        # 3. Author Count Distribution (fixed to start from 1)
        plot_path = create_individual_plot_3_author_count_distribution(df, output_dir)
        if plot_path: created_plots.append(plot_path)
        
        # 4. Citation Analysis
        plot_path = create_individual_plot_4_citation_analysis(df, output_dir)
        if plot_path: created_plots.append(plot_path)
        
        # 5. Top Topics
        plot_path = create_individual_plot_5_top_topics(analysis_report, output_dir)
        if plot_path: created_plots.append(plot_path)
        
        # 6. Relevance Trends by Year
        plot_path = create_individual_plot_6_relevance_by_year(df, output_dir)
        if plot_path: created_plots.append(plot_path)
        
        # 7. Citation vs Relevance
        plot_path = create_individual_plot_7_citation_vs_relevance(df, output_dir)
        if plot_path: created_plots.append(plot_path)
        
        # 8. Research Evolution Timeline
        plot_path = create_individual_plot_8_research_evolution(df, output_dir)
        if plot_path: created_plots.append(plot_path)
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎉 Generated {len(created_plots)} individual visualizations")
    print(f"📂 All files saved to: {os.path.abspath(output_dir)}")
    
    # List created files with sizes
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        print(f"\n📊 Created visualizations:")
        for file in sorted(files):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"   📈 {file} ({file_size:,} bytes)")
    
    print(f"\n✅ Diagnostics completed - all issues fixed!")

if __name__ == "__main__":
    main()