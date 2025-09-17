#!/usr/bin/env python3
"""
Comprehensive visualization generator with all missing plots from original bibliometrics
- Individual plots instead of subplots  
- Author count starts from 1 (not 0)
- No predictions beyond 2026
- Proper text wrapping and layout
- Complete set of visualizations matching original scope
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
import textwrap
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def load_education_data():
    """Load the education analysis JSON data with diagnostics"""
    json_file_path = "/home/campus.ncl.ac.uk/c1041562/Desktop/codes/article_agent/bibm/output/20250831_024823_education_education_analysis.json"
    
    print(f"📂 Loading data from: {os.path.basename(json_file_path)}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    analysis_report = data.get('analysis_report', {})
    papers_data = []
    if isinstance(data, list):
        papers_data = data
    else:
        for key, value in data.items():
            if key != 'analysis_report' and isinstance(value, list):
                papers_data = value
                break
    
    print(f"📊 Found {len(papers_data)} papers in dataset")
    return analysis_report, papers_data

def clean_and_validate_data(papers_data):
    """Clean and validate the papers data"""
    print("\n🔍 Cleaning and validating data...")
    
    df = pd.DataFrame(papers_data)
    print(f"   Original papers: {len(df)}")
    
    # Clean year data - remove future predictions and invalid years
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    current_year = datetime.now().year
    valid_years = (df['year'] >= 1990) & (df['year'] <= min(current_year, 2026))
    df = df[valid_years].copy()
    df['year'] = df['year'].astype(int)
    
    # Clean citation data
    df['citation_count'] = pd.to_numeric(df['citation_count'], errors='coerce').fillna(0).astype(int)
    
    # Clean relevance scores
    df['relevance_score'] = pd.to_numeric(df['relevance_score'], errors='coerce').fillna(0)
    df['relevance_score'] = df['relevance_score'].clip(0, 1)
    
    # Fix author count to start from 1
    def count_authors(authors):
        try:
            if pd.isna(authors):
                return 1
            if isinstance(authors, list):
                return max(1, len(authors)) if len(authors) > 0 else 1
            if isinstance(authors, str):
                if authors.strip() == '':
                    return 1
                return max(1, len([a.strip() for a in authors.split(',') if a.strip()]))
            return 1
        except:
            return 1
    
    df['author_count'] = df['authors'].apply(count_authors)
    
    # Extract domain/field from venue
    def extract_domain(venue):
        if not venue:
            return 'Unknown'
        venue_lower = venue.lower()
        domain_keywords = {
            'AI/ML': ['artificial intelligence', 'machine learning', 'neural', 'deep learning', 'nlp'],
            'Education': ['education', 'learning', 'teaching', 'curriculum', 'pedagogy', 'school'],
            'Computer Science': ['computer science', 'computing', 'software', 'algorithm'],
            'Technology': ['technology', 'digital', 'tech', 'innovation'],
            'Other': []
        }
        for domain, keywords in domain_keywords.items():
            if any(keyword in venue_lower for keyword in keywords):
                return domain
        return 'Other'
    
    df['domain'] = df['venue'].apply(extract_domain)
    
    print(f"   ✅ Final dataset: {len(df)} papers")
    print(f"   📅 Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"   👥 Author count range: {df['author_count'].min()} - {df['author_count'].max()}")
    
    return df

def wrap_labels(labels, width=25):
    """Wrap long labels for better display"""
    return [textwrap.fill(label, width) for label in labels]

def create_plot_01_temporal_analysis(df, output_dir):
    """Temporal Analysis - 4 individual plots"""
    
    # Plot 1: Publications per year
    plt.figure(figsize=(12, 6))
    yearly_counts = df['year'].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(yearly_counts)))
    bars = plt.bar(yearly_counts.index, yearly_counts.values, alpha=0.8, color=colors)
    
    # Add trend line
    z = np.polyfit(yearly_counts.index, yearly_counts.values, 1)
    p = np.poly1d(z)
    plt.plot(yearly_counts.index, p(yearly_counts.index), "r--", alpha=0.8, linewidth=2)
    
    plt.title('Publications per Year\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Publications', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01a_publications_per_year.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Cumulative publications
    plt.figure(figsize=(12, 6))
    cumulative = yearly_counts.cumsum()
    plt.plot(cumulative.index, cumulative.values, marker='o', linewidth=3, markersize=6, color='darkgreen')
    plt.fill_between(cumulative.index, cumulative.values, alpha=0.3, color='lightgreen')
    plt.title('Cumulative Publications Over Time\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Cumulative Count', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01b_cumulative_publications.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Total citations per year
    plt.figure(figsize=(12, 6))
    yearly_citations = df.groupby('year')['citation_count'].sum()
    plt.bar(yearly_citations.index, yearly_citations.values, alpha=0.7, color='orange')
    plt.title('Total Citations per Year\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Citations', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01c_citations_per_year.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Average citations per paper by year
    plt.figure(figsize=(12, 6))
    avg_citations = df.groupby('year')['citation_count'].mean()
    plt.plot(avg_citations.index, avg_citations.values, marker='s', linewidth=3, markersize=8, color='red')
    plt.title('Average Citations per Paper by Year\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Citations', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01d_avg_citations_per_paper.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created: Temporal Analysis (4 plots)")

def create_plot_02_citation_analysis(df, output_dir):
    """Citation Analysis - 4 individual plots"""
    
    # Plot 1: Citation distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['citation_count'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Citation Distribution\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.xlabel('Citation Count', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02a_citation_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Top cited papers
    plt.figure(figsize=(14, 10))
    top_cited = df.nlargest(20, 'citation_count')
    if len(top_cited) > 0:
        wrapped_titles = wrap_labels([title[:60] for title in top_cited['title']], width=50)
        plt.barh(range(len(top_cited)), top_cited['citation_count'].values, alpha=0.7, color='green')
        plt.title('Top 20 Most Cited Papers\nK12 Education & AI Research', fontsize=14, fontweight='bold')
        plt.xlabel('Citation Count', fontsize=12)
        plt.yticks(range(len(top_cited)), wrapped_titles, fontsize=9)
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02b_top_cited_papers.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Citations vs Year scatter
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['year'], df['citation_count'], alpha=0.6, c=df['relevance_score'], 
                         cmap='viridis', s=30)
    plt.colorbar(scatter, label='Relevance Score')
    plt.title('Citations vs Publication Year\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Citation Count', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02c_citations_vs_year.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: H-index analysis
    plt.figure(figsize=(10, 6))
    sorted_citations = sorted(df['citation_count'].values, reverse=True)
    h_index = 0
    for i, citations in enumerate(sorted_citations):
        if citations >= i + 1:
            h_index = i + 1
        else:
            break
    
    plt.plot(range(1, len(sorted_citations) + 1), sorted_citations, 'b-', linewidth=2, label='Citation count')
    plt.plot(range(1, len(sorted_citations) + 1), range(1, len(sorted_citations) + 1), 'r--', linewidth=2, label='Rank')
    plt.axvline(x=h_index, color='green', linestyle=':', linewidth=3, label=f'H-index: {h_index}')
    plt.title(f'H-index Analysis\nK12 Education & AI Research (H-index: {h_index})', fontsize=14, fontweight='bold')
    plt.xlabel('Paper Rank', fontsize=12)
    plt.ylabel('Citation Count', fontsize=12)
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02d_h_index_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created: Citation Analysis (4 plots)")

def create_plot_03_author_analysis(df, output_dir):
    """Author Analysis - 4 individual plots"""
    
    # Extract all authors
    all_authors = []
    for authors_list in df['authors']:
        if isinstance(authors_list, list):
            all_authors.extend(authors_list)
    
    author_counts = Counter(all_authors)
    top_authors = author_counts.most_common(20)
    
    # Plot 1: Top authors by publication count
    plt.figure(figsize=(12, 10))
    if top_authors:
        authors, counts = zip(*top_authors)
        wrapped_authors = wrap_labels(authors, width=30)
        plt.barh(range(len(authors)), counts, alpha=0.7, color='teal')
        plt.title('Top 20 Authors by Publication Count\nK12 Education & AI Research', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Publications', fontsize=12)
        plt.yticks(range(len(authors)), wrapped_authors, fontsize=9)
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03a_top_authors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Authors per paper distribution
    plt.figure(figsize=(10, 6))
    author_count_dist = df['author_count'].value_counts().sort_index()
    max_authors = min(author_count_dist.index.max(), 15)
    author_count_dist = author_count_dist[author_count_dist.index <= max_authors]
    
    plt.bar(author_count_dist.index, author_count_dist.values, alpha=0.7, color='orange')
    plt.title('Distribution of Authors per Paper\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Authors', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.xticks(range(1, max_authors + 1))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03b_authors_per_paper.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Author productivity vs citations
    plt.figure(figsize=(10, 8))
    author_productivity = defaultdict(lambda: {'papers': 0, 'citations': 0})
    for _, paper in df.iterrows():
        if isinstance(paper['authors'], list):
            for author in paper['authors']:
                author_productivity[author]['papers'] += 1
                author_productivity[author]['citations'] += paper['citation_count']
    
    top_productive = sorted(author_productivity.items(), key=lambda x: x[1]['papers'], reverse=True)[:20]
    if top_productive:
        papers_counts = [data['papers'] for _, data in top_productive]
        citation_counts = [data['citations'] for _, data in top_productive]
        
        scatter = plt.scatter(papers_counts, citation_counts, alpha=0.7, s=100, color='red')
        plt.title('Author Productivity vs Citations\nK12 Education & AI Research', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Papers', fontsize=12)
        plt.ylabel('Total Citations', fontsize=12)
        
        # Annotate top 5 authors
        for i, (author, data) in enumerate(top_productive[:5]):
            author_short = author[:15] + '...' if len(author) > 15 else author
            plt.annotate(author_short, (data['papers'], data['citations']), 
                        fontsize=8, ha='center', va='bottom')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03c_author_productivity_vs_citations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Author collaboration network
    plt.figure(figsize=(12, 10))
    G = nx.Graph()
    top_author_names = [author for author, _ in top_authors[:15]]  # Top 15 for better visualization
    
    # Add edges for co-authorships
    for _, paper in df.iterrows():
        if isinstance(paper['authors'], list) and len(paper['authors']) > 1:
            authors = [a for a in paper['authors'] if a in top_author_names]
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    if G.has_edge(authors[i], authors[j]):
                        G[authors[i]][authors[j]]['weight'] += 1
                    else:
                        G.add_edge(authors[i], authors[j], weight=1)
    
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw network with better labels
        node_sizes = [G.degree(node) * 300 + 300 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
        
        # Add shortened labels
        labels = {node: node[:10] + '...' if len(node) > 10 else node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        plt.title('Author Collaboration Network\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No collaboration data available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('Author Collaboration Network\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03d_author_collaboration_network.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created: Author Analysis (4 plots)")

def create_plot_04_venue_analysis(df, output_dir):
    """Venue Analysis - 4 individual plots"""
    
    # Plot 1: Top venues by publication count
    plt.figure(figsize=(14, 10))
    venue_counts = df['venue'].value_counts().head(15)
    if len(venue_counts) > 0:
        wrapped_venues = wrap_labels([venue[:50] for venue in venue_counts.index], width=40)
        plt.barh(range(len(venue_counts)), venue_counts.values, alpha=0.7, color='purple')
        plt.title('Top 15 Publication Venues\nK12 Education & AI Research', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Publications', fontsize=12)
        plt.yticks(range(len(venue_counts)), wrapped_venues, fontsize=9)
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04a_top_venues.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Domain distribution
    plt.figure(figsize=(10, 8))
    domain_counts = df['domain'].value_counts()
    if len(domain_counts) > 0:
        colors = plt.cm.Set3(np.linspace(0, 1, len(domain_counts)))
        wedges, texts, autotexts = plt.pie(domain_counts.values, labels=domain_counts.index, 
                                          autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('Research Domain Distribution\nK12 Education & AI Research', fontsize=14, fontweight='bold')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04b_domain_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Venue impact (average citations)
    plt.figure(figsize=(12, 8))
    venue_impact = df.groupby('venue').agg({
        'citation_count': ['mean', 'count']
    }).round(2)
    venue_impact.columns = ['avg_citations', 'paper_count']
    venue_impact = venue_impact[venue_impact['paper_count'] >= 2].sort_values('avg_citations', ascending=False).head(10)
    
    if len(venue_impact) > 0:
        wrapped_venues = wrap_labels([venue[:40] for venue in venue_impact.index], width=30)
        plt.barh(range(len(venue_impact)), venue_impact['avg_citations'].values, alpha=0.7, color='green')
        plt.title('Top 10 Venues by Average Citations\nK12 Education & AI Research', fontsize=14, fontweight='bold')
        plt.xlabel('Average Citations per Paper', fontsize=12)
        plt.yticks(range(len(venue_impact)), wrapped_venues, fontsize=9)
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04c_venue_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Publications vs Citations by domain
    plt.figure(figsize=(10, 8))
    domain_stats = df.groupby('domain').agg({
        'title': 'count',
        'citation_count': 'sum'
    }).rename(columns={'title': 'publications', 'citation_count': 'total_citations'})
    
    if len(domain_stats) > 0:
        scatter = plt.scatter(domain_stats['publications'], domain_stats['total_citations'], 
                            alpha=0.7, s=200, c=range(len(domain_stats)), cmap='viridis')
        plt.title('Domain: Publications vs Citations\nK12 Education & AI Research', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Publications', fontsize=12)
        plt.ylabel('Total Citations', fontsize=12)
        
        for domain, stats in domain_stats.iterrows():
            plt.annotate(domain, (stats['publications'], stats['total_citations']), 
                        fontsize=10, ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04d_domain_publications_vs_citations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created: Venue Analysis (4 plots)")

def create_plot_05_text_analysis(df, output_dir):
    """Text Analysis - 4 individual plots"""
    
    # Combine titles and abstracts for text analysis
    all_text = ' '.join(df['title'].fillna('') + ' ' + df['abstract'].fillna(''))
    all_text = re.sub(r'[^\w\s]', ' ', all_text.lower())
    all_text = re.sub(r'\s+', ' ', all_text)
    
    # Plot 1: Word cloud
    plt.figure(figsize=(14, 10))
    if all_text:
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                        'this', 'that', 'these', 'those', 'i', 'we', 'you', 'he', 'she', 'it', 'they',
                        'paper', 'study', 'research', 'method', 'approach', 'result', 'results', 'conclusion'])
        
        wordcloud = WordCloud(width=1200, height=800, 
                            background_color='white',
                            stopwords=stop_words,
                            max_words=100,
                            collocations=False,
                            colormap='viridis').generate(all_text)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud\nK12 Education & AI Research', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05a_word_cloud.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Most frequent words
    plt.figure(figsize=(12, 10))
    words = all_text.split()
    word_freq = Counter([word for word in words if len(word) > 3 and word not in 
                       ['paper', 'study', 'research', 'method', 'approach', 'result', 'results']])
    
    if word_freq:
        top_words = word_freq.most_common(20)
        words, counts = zip(*top_words)
        plt.barh(range(len(words)), counts, alpha=0.7, color='orange')
        plt.title('Top 20 Most Frequent Words\nK12 Education & AI Research', fontsize=14, fontweight='bold')
        plt.xlabel('Frequency', fontsize=12)
        plt.yticks(range(len(words)), words, fontsize=10)
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05b_frequent_words.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Title length distribution
    plt.figure(figsize=(10, 6))
    title_lengths = df['title'].str.len().dropna()
    plt.hist(title_lengths, bins=30, alpha=0.7, color='lightblue', edgecolor='navy')
    plt.axvline(title_lengths.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {title_lengths.mean():.1f} chars')
    plt.title('Title Length Distribution\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.xlabel('Title Length (characters)', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05c_title_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Abstract length distribution  
    plt.figure(figsize=(10, 6))
    abstract_lengths = df['abstract'].str.len().dropna()
    if len(abstract_lengths) > 0:
        plt.hist(abstract_lengths, bins=30, alpha=0.7, color='lightcoral', edgecolor='darkred')
        plt.axvline(abstract_lengths.mean(), color='blue', linestyle='--', linewidth=2, 
                    label=f'Mean: {abstract_lengths.mean():.0f} chars')
        plt.title('Abstract Length Distribution\nK12 Education & AI Research', fontsize=14, fontweight='bold')
        plt.xlabel('Abstract Length (characters)', fontsize=12)
        plt.ylabel('Number of Papers', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No abstract data available', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.title('Abstract Length Distribution\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05d_abstract_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created: Text Analysis (4 plots)")

def create_plot_06_relevance_analysis(df, output_dir):
    """Relevance Analysis - 4 individual plots"""
    
    # Plot 1: Relevance score distribution
    plt.figure(figsize=(10, 6))
    relevance_scores = df['relevance_score'].dropna()
    plt.hist(relevance_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.axvline(relevance_scores.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {relevance_scores.mean():.3f}')
    plt.axvline(relevance_scores.median(), color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {relevance_scores.median():.3f}')
    plt.title('Relevance Score Distribution\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.xlabel('Relevance Score', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06a_relevance_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Relevance by year
    plt.figure(figsize=(12, 6))
    yearly_relevance = df.groupby('year')['relevance_score'].agg(['mean', 'std', 'count']).reset_index()
    yearly_relevance = yearly_relevance[yearly_relevance['count'] >= 3]
    
    plt.errorbar(yearly_relevance['year'], yearly_relevance['mean'], 
                yerr=yearly_relevance['std'], marker='o', linewidth=2, 
                markersize=6, capsize=5, alpha=0.8, color='darkblue')
    plt.axhline(y=0.7, color='orange', linestyle=':', alpha=0.7, linewidth=2,
                label='Common Relevance Threshold (0.7)')
    plt.title('Relevance Score Trends by Year\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mean Relevance Score', fontsize=12)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06b_relevance_by_year.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Relevance vs Citations
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['relevance_score'], df['citation_count'], 
                         alpha=0.6, s=30, c=df['year'], cmap='viridis')
    plt.colorbar(scatter, label='Publication Year')
    correlation = df['relevance_score'].corr(df['citation_count'])
    plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
    plt.title('Relevance Score vs Citation Count\nK12 Education & AI Research', fontsize=14, fontweight='bold')
    plt.xlabel('Relevance Score', fontsize=12)
    plt.ylabel('Citation Count', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06c_relevance_vs_citations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: High relevance papers over time
    plt.figure(figsize=(12, 6))
    high_rel_threshold = 0.8
    high_rel_papers = df[df['relevance_score'] >= high_rel_threshold]
    high_rel_yearly = high_rel_papers['year'].value_counts().sort_index()
    total_yearly = df['year'].value_counts().sort_index()
    
    # Calculate percentage of high relevance papers per year
    percentage_high_rel = (high_rel_yearly / total_yearly * 100).fillna(0)
    
    plt.bar(percentage_high_rel.index, percentage_high_rel.values, alpha=0.7, color='green')
    plt.title(f'High Relevance Papers (≥{high_rel_threshold}) by Year\nK12 Education & AI Research', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Percentage of High Relevance Papers', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06d_high_relevance_papers_by_year.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created: Relevance Analysis (4 plots)")

def main():
    """Main function to generate comprehensive individual visualizations"""
    output_dir = "output_4"
    
    
    print("=== Comprehensive K12 Education & AI Visualization Generator ===")
    print("🎯 Complete feature set:")
    print("   - All missing plots from original bibliometrics code")
    print("   - Individual plots (no subplots)")
    print("   - Proper text wrapping and layout")
    print("   - Author count starts from 1 (not 0)")
    print("   - No predictions beyond 2026")
    print("   - Enhanced styling and annotations")
    print(f"   - Output to: {output_dir}")
    print()
    
    # Load and validate data
    analysis_report, papers_data = load_education_data()
    if not papers_data:
        return
    
    df = clean_and_validate_data(papers_data)
    if df.empty:
        return
    
    print(f"\n🎨 Generating comprehensive visualizations...")
    print("=" * 60)
    
    # Generate all visualization sets
    try:
        create_plot_01_temporal_analysis(df, output_dir)
        create_plot_02_citation_analysis(df, output_dir)
        create_plot_03_author_analysis(df, output_dir)
        create_plot_04_venue_analysis(df, output_dir)
        create_plot_05_text_analysis(df, output_dir)
        create_plot_06_relevance_analysis(df, output_dir)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        print(f"🎉 Generated {len(files)} comprehensive visualizations")
        print(f"📂 All files saved to: {os.path.abspath(output_dir)}")
        
        print(f"\n📊 Generated visualization categories:")
        categories = {
            '01': 'Temporal Analysis (4 plots)',
            '02': 'Citation Analysis (4 plots)',
            '03': 'Author Analysis (4 plots)',
            '04': 'Venue Analysis (4 plots)',
            '05': 'Text Analysis (4 plots)',
            '06': 'Relevance Analysis (4 plots)'
        }
        
        for prefix, description in categories.items():
            count = len([f for f in files if f.startswith(prefix)])
            print(f"   📈 {description}: {count} files")

if __name__ == "__main__":
    main()