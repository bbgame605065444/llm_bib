#!/usr/bin/env python3
"""
Generate comprehensive visualizations from education analysis JSON data
Uses existing bibliometrics code and creates visualizations in output_2 directory
"""

import json
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import networkx as nx
from wordcloud import WordCloud
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add bibm directory to path to import bibliometrics
sys.path.append('/home/campus.ncl.ac.uk/c1041562/Desktop/codes/article_agent/bibm')
from bibliometrics import BibliometricsAnalyzer

def load_education_data():
    """Load the education analysis JSON data"""
    json_file_path = "/home/campus.ncl.ac.uk/c1041562/Desktop/codes/article_agent/bibm/output/20250831_024823_education_education_analysis.json"
    
    print(f"Loading data from: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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
    
    return analysis_report, papers_data

def create_enhanced_visualizations(papers_data, analysis_report, output_dir):
    """Create enhanced visualizations using the existing bibliometrics framework"""
    
    # Prepare data for bibliometrics analyzer
    # Convert papers data to the format expected by the analyzer
    processed_papers = []
    for paper in papers_data:
        processed_paper = {
            'title': paper.get('title', ''),
            'authors': paper.get('authors', []),
            'year': paper.get('year', ''),
            'citation_count': paper.get('citation_count', 0),
            'venue': paper.get('venue', ''),
            'link': paper.get('link', ''),
            'abstract': paper.get('abstract', ''),
            'relevance_score': paper.get('relevance_score', 0),
            'is_relevant': paper.get('is_relevant', False),
            'key_topics': paper.get('key_topics', []),
            'confidence': paper.get('confidence', 0)
        }
        processed_papers.append(processed_paper)
    
    # Initialize bibliometrics analyzer with output_2 directory
    analyzer = BibliometricsAnalyzer(output_dir=output_dir)
    
    # Get the original keyword from analysis report
    keyword = analysis_report.get('analysis_metadata', {}).get('keyword', 'K12 education, AI, education')
    
    print(f"Generating visualizations for keyword: '{keyword}'")
    print(f"Processing {len(processed_papers)} papers...")
    
    # Generate comprehensive analysis
    viz_paths = analyzer.generate_comprehensive_analysis(
        papers=processed_papers,
        keyword=keyword,
        analyzed_papers=processed_papers  # Use the analyzed papers with relevance scores
    )
    
    return viz_paths

def create_custom_education_visualizations(papers_data, analysis_report, output_dir):
    """Create custom visualizations specific to the education dataset"""
    
    # Create DataFrame from papers
    df_papers = pd.DataFrame(papers_data)
    
    # Set up matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Relevance Score Distribution (Enhanced)
    plt.figure(figsize=(12, 8))
    
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top left: Relevance score histogram
    relevance_scores = df_papers['relevance_score'].dropna()
    ax1.hist(relevance_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Relevance Score')
    ax1.set_ylabel('Number of Papers')
    ax1.set_title('Distribution of Relevance Scores')
    ax1.grid(True, alpha=0.3)
    
    # Top right: Score distribution by bins
    score_dist = analysis_report.get('score_distribution', {})
    bins = list(score_dist.keys())
    counts = list(score_dist.values())
    ax2.bar(bins, counts, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Score Range')
    ax2.set_ylabel('Number of Papers')
    ax2.set_title('Papers by Relevance Score Range')
    ax2.tick_params(axis='x', rotation=45)
    
    # Bottom left: Top topics
    top_topics = analysis_report.get('top_topics', [])[:10]
    if top_topics:
        topics = [topic[0] for topic in top_topics]
        frequencies = [topic[1] for topic in top_topics]
        ax3.barh(topics, frequencies, color='lightgreen', alpha=0.7)
        ax3.set_xlabel('Frequency')
        ax3.set_title('Top 10 Research Topics')
        ax3.invert_yaxis()
    
    # Bottom right: Yearly publication trends
    yearly_counts = df_papers.groupby('year').size().reset_index(name='count')
    yearly_counts = yearly_counts[yearly_counts['year'].notna()]
    yearly_counts = yearly_counts[yearly_counts['year'] != 'Unknown']
    yearly_counts['year'] = pd.to_numeric(yearly_counts['year'], errors='coerce')
    yearly_counts = yearly_counts.dropna()
    yearly_counts = yearly_counts.sort_values('year')
    
    ax4.plot(yearly_counts['year'], yearly_counts['count'], marker='o', linewidth=2, markersize=6)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Number of Papers')
    ax4.set_title('Publication Trends Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'education_ai_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Interactive Plotly Dashboard
    fig_plotly = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Relevance Score Distribution', 'Publications by Year', 
                       'Top Authors by Paper Count', 'Citation vs Relevance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Relevance score histogram
    fig_plotly.add_trace(
        go.Histogram(x=relevance_scores, nbinsx=20, name='Relevance Scores'),
        row=1, col=1
    )
    
    # Publications by year
    if not yearly_counts.empty:
        fig_plotly.add_trace(
            go.Scatter(x=yearly_counts['year'], y=yearly_counts['count'], 
                      mode='lines+markers', name='Publications'),
            row=1, col=2
        )
    
    # Top authors
    author_counts = {}
    for paper in papers_data:
        for author in paper.get('authors', []):
            author_counts[author] = author_counts.get(author, 0) + 1
    
    top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_authors:
        authors, counts = zip(*top_authors)
        fig_plotly.add_trace(
            go.Bar(x=list(counts), y=list(authors), orientation='h', name='Author Papers'),
            row=2, col=1
        )
    
    # Citation vs Relevance scatter
    citations = df_papers['citation_count'].fillna(0)
    relevance = df_papers['relevance_score'].fillna(0)
    fig_plotly.add_trace(
        go.Scatter(x=relevance, y=citations, mode='markers', 
                  name='Papers', opacity=0.6),
        row=2, col=2
    )
    
    fig_plotly.update_layout(
        title_text="K12 Education & AI Research Analysis Dashboard",
        showlegend=False,
        height=800
    )
    
    # Save interactive plot
    pyo.plot(fig_plotly, filename=os.path.join(output_dir, 'interactive_dashboard.html'), 
             auto_open=False)
    
    # 3. Word Cloud from abstracts and titles
    text_data = []
    for paper in papers_data:
        if paper.get('abstract'):
            text_data.append(paper['abstract'])
        if paper.get('title'):
            text_data.append(paper['title'])
    
    if text_data:
        all_text = ' '.join(text_data)
        wordcloud = WordCloud(width=1200, height=800, background_color='white',
                             max_words=100, colormap='viridis').generate(all_text)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud: K12 Education & AI Research', fontsize=20, pad=20)
        plt.savefig(os.path.join(output_dir, 'research_wordcloud.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # 4. Network analysis of author collaborations
    G = nx.Graph()
    
    # Build collaboration network
    for paper in papers_data:
        authors = paper.get('authors', [])
        if len(authors) > 1:
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    if G.has_edge(authors[i], authors[j]):
                        G[authors[i]][authors[j]]['weight'] += 1
                    else:
                        G.add_edge(authors[i], authors[j], weight=1)
    
    # Filter to show only authors with multiple collaborations
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 2]
    G.remove_edges_from(edges_to_remove)
    
    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    if G.number_of_nodes() > 0:
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=300, alpha=0.7, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
        
        # Add labels for well-connected authors only
        degree_dict = dict(G.degree())
        high_degree_nodes = {node: degree for node, degree in degree_dict.items() if degree >= 3}
        
        if high_degree_nodes:
            nx.draw_networkx_labels(G, pos, labels={node: node for node in high_degree_nodes.keys()}, 
                                  font_size=8)
        
        plt.title('Author Collaboration Network\n(Authors with 2+ collaborations)', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'author_collaboration_network.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ Custom visualizations created in {output_dir}")

def main():
    """Main function to generate all visualizations"""
    output_dir = "output_2"
    
    print("=== K12 Education & AI Research Visualization Generator ===")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data
    try:
        analysis_report, papers_data = load_education_data()
        print(f"✅ Successfully loaded {len(papers_data)} papers")
        
        metadata = analysis_report.get('analysis_metadata', {})
        stats = analysis_report.get('relevance_statistics', {})
        
        print(f"📊 Analysis Summary:")
        print(f"   - Keyword: {metadata.get('keyword', 'N/A')}")
        print(f"   - Total Papers: {metadata.get('total_papers', 0)}")
        print(f"   - Relevant Papers: {stats.get('relevant_papers', 0)}")
        print(f"   - Relevance Rate: {stats.get('relevance_rate', 0):.2%}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create visualizations using existing bibliometrics framework
    print("🎨 Generating comprehensive visualizations using bibliometrics framework...")
    try:
        viz_paths = create_enhanced_visualizations(papers_data, analysis_report, output_dir)
        print("✅ Bibliometric visualizations completed")
        
        for viz_type, path in viz_paths.items():
            if path and os.path.exists(path):
                print(f"   📈 {viz_type}: {os.path.basename(path)}")
        
    except Exception as e:
        print(f"⚠️  Error with bibliometric visualizations: {e}")
        print("Continuing with custom visualizations...")
    
    # Create custom education-specific visualizations
    print("\n🎨 Creating custom education & AI visualizations...")
    try:
        create_custom_education_visualizations(papers_data, analysis_report, output_dir)
    except Exception as e:
        print(f"❌ Error creating custom visualizations: {e}")
    
    # List all created files
    print(f"\n📁 Generated files in {output_dir}:")
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.html'))]
        for file in sorted(files):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"   📄 {file} ({file_size:,} bytes)")
    
    print(f"\n🎉 Visualization generation complete!")
    print(f"📂 All files saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()