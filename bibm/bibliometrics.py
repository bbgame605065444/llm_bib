import json
import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
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
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BibliometricsAnalyzer:
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _extract_top_keywords(self, keyword_string: str, num_words: int = 3) -> str:
        """Extract top N meaningful keywords from search term"""
        # Remove common stop words and clean the string
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Split and clean keywords
        words = re.findall(r'\b[a-zA-Z]{2,}\b', keyword_string.lower())
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Take top N words (or all if fewer than N)
        top_words = meaningful_words[:num_words] if len(meaningful_words) >= num_words else meaningful_words
        
        return '_'.join(top_words)
    
    def _generate_filename_prefix(self, keyword: str) -> str:
        """Generate filename prefix with datetime and top 3 keywords"""
        # Get current datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract top 3 keywords
        top_keywords = self._extract_top_keywords(keyword, 3)
        
        # Combine timestamp and keywords
        return f"{timestamp}_{top_keywords}"
        
    def generate_comprehensive_analysis(self, papers: List[Dict], keyword: str, 
                                      analyzed_papers: Optional[List[Dict]] = None) -> Dict[str, str]:
        """
        Generate comprehensive bibliometric analysis with all visualizations
        
        Args:
            papers: List of paper dictionaries from scraping
            keyword: Original search keyword
            analyzed_papers: Optional list of papers with relevance analysis
            
        Returns:
            Dictionary with paths to generated visualizations
        """
        logger.info(f"Starting comprehensive bibliometric analysis for '{keyword}'")
        
        # Generate filename prefix for consistent naming
        filename_prefix = self._generate_filename_prefix(keyword)
        
        # Use analyzed papers if available, otherwise use all papers
        analysis_data = analyzed_papers or papers
        
        # Convert to DataFrame for easier analysis
        df = self._prepare_dataframe(analysis_data)
        
        # Generate all visualizations
        viz_paths = {}
        
        try:
            # 1. Temporal Analysis
            viz_paths['temporal'] = self._create_temporal_analysis(df, keyword, filename_prefix)
            
            # 2. Citation Analysis
            viz_paths['citations'] = self._create_citation_analysis(df, keyword, filename_prefix)
            
            # 3. Author Analysis
            viz_paths['authors'] = self._create_author_analysis(df, keyword, filename_prefix)
            
            # 4. Venue Analysis
            viz_paths['venues'] = self._create_venue_analysis(df, keyword, filename_prefix)
            
            # 5. Text Analysis (Word Clouds, Topic Analysis)
            viz_paths['text'] = self._create_text_analysis(df, keyword, filename_prefix)
            
            # 6. Relevance Analysis (if analyzed papers available)
            if analyzed_papers:
                viz_paths['relevance'] = self._create_relevance_analysis(df, keyword, filename_prefix)
            
            # 7. Network Analysis
            viz_paths['network'] = self._create_network_analysis(df, keyword, filename_prefix)
            
            # 8. Cutting-edge Analysis (Bertology/LLM specific)
            viz_paths['cutting_edge'] = self._create_cutting_edge_analysis(df, keyword, filename_prefix)
            
            # 9. Interactive Dashboard
            viz_paths['dashboard'] = self._create_interactive_dashboard(df, keyword, filename_prefix)
            
            # Generate comprehensive report
            viz_paths['report'] = self._generate_analysis_report(df, keyword, viz_paths, filename_prefix)
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise e
        
        logger.info(f"Analysis completed. Generated {len(viz_paths)} visualization sets")
        return viz_paths
    
    def _prepare_dataframe(self, papers: List[Dict]) -> pd.DataFrame:
        """Convert papers list to pandas DataFrame for analysis"""
        
        processed_papers = []
        for paper in papers:
            processed_paper = {
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', paper.get('snippet', '')),
                'authors': paper.get('authors', []),
                'year': paper.get('year'),
                'venue': paper.get('venue', ''),
                'citation_count': paper.get('citation_count', paper.get('cited_by', 0)),
                'link': paper.get('link', ''),
                'relevance_score': paper.get('relevance_score', None),
                'is_relevant': paper.get('is_relevant', None),
                'key_topics': paper.get('key_topics', [])
            }
            
            # Clean and process data
            if isinstance(processed_paper['authors'], list):
                processed_paper['author_count'] = len(processed_paper['authors'])
                processed_paper['first_author'] = processed_paper['authors'][0] if processed_paper['authors'] else ''
                processed_paper['authors_str'] = ', '.join(processed_paper['authors'])
            else:
                processed_paper['author_count'] = 0
                processed_paper['first_author'] = ''
                processed_paper['authors_str'] = ''
            
            # Extract domain/field from venue
            processed_paper['domain'] = self._extract_domain(processed_paper['venue'])
            
            # Text length metrics
            processed_paper['title_length'] = len(processed_paper['title'])
            processed_paper['abstract_length'] = len(processed_paper['abstract'])
            
            processed_papers.append(processed_paper)
        
        df = pd.DataFrame(processed_papers)
        
        # Clean year data
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)
        
        # Clean citation data
        df['citation_count'] = pd.to_numeric(df['citation_count'], errors='coerce').fillna(0).astype(int)
        
        return df
    
    def _extract_domain(self, venue: str) -> str:
        """Extract research domain from venue string"""
        if not venue:
            return 'Unknown'
        
        venue_lower = venue.lower()
        
        # Define domain keywords
        domain_keywords = {
            'AI/ML': ['artificial intelligence', 'machine learning', 'neural', 'deep learning', 'nlp', 'natural language'],
            'Computer Vision': ['computer vision', 'image', 'visual', 'cvpr', 'iccv', 'eccv'],
            'NLP': ['natural language processing', 'computational linguistics', 'nlp', 'language', 'text'],
            'Robotics': ['robotics', 'robot', 'autonomous', 'control'],
            'Theory': ['theory', 'theoretical', 'algorithm', 'complexity'],
            'Systems': ['systems', 'distributed', 'database', 'network'],
            'Security': ['security', 'cryptography', 'privacy'],
            'HCI': ['human-computer', 'interaction', 'interface', 'usability']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in venue_lower for keyword in keywords):
                return domain
        
        return 'Other'
    
    def _create_temporal_analysis(self, df: pd.DataFrame, keyword: str, filename_prefix: str) -> str:
        """Create temporal analysis visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Temporal Analysis - {keyword}', fontsize=16, fontweight='bold')
        
        # 1. Publications per year
        yearly_counts = df['year'].value_counts().sort_index()
        axes[0, 0].bar(yearly_counts.index, yearly_counts.values, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Publications per Year')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Number of Publications')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Cumulative publications
        cumulative = yearly_counts.cumsum()
        axes[0, 1].plot(cumulative.index, cumulative.values, marker='o', linewidth=2, color='darkgreen')
        axes[0, 1].set_title('Cumulative Publications')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Cumulative Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Citations over time
        yearly_citations = df.groupby('year')['citation_count'].sum()
        axes[1, 0].bar(yearly_citations.index, yearly_citations.values, alpha=0.7, color='orange')
        axes[1, 0].set_title('Total Citations per Year')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Total Citations')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Average citations per paper by year
        avg_citations = df.groupby('year')['citation_count'].mean()
        axes[1, 1].plot(avg_citations.index, avg_citations.values, marker='s', linewidth=2, color='red')
        axes[1, 1].set_title('Average Citations per Paper')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Average Citations')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{filename_prefix}_temporal_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_citation_analysis(self, df: pd.DataFrame, keyword: str, filename_prefix: str) -> str:
        """Create citation analysis visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Citation Analysis - {keyword}', fontsize=16, fontweight='bold')
        
        # 1. Citation distribution
        axes[0, 0].hist(df['citation_count'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 0].set_title('Citation Distribution')
        axes[0, 0].set_xlabel('Citation Count')
        axes[0, 0].set_ylabel('Number of Papers')
        axes[0, 0].set_yscale('log')
        
        # 2. Top cited papers
        top_cited = df.nlargest(20, 'citation_count')
        axes[0, 1].barh(range(len(top_cited)), top_cited['citation_count'].values, alpha=0.7, color='green')
        axes[0, 1].set_title('Top 20 Cited Papers')
        axes[0, 1].set_xlabel('Citation Count')
        axes[0, 1].set_yticks(range(len(top_cited)))
        axes[0, 1].set_yticklabels([title[:50] + '...' if len(title) > 50 else title 
                                   for title in top_cited['title']], fontsize=8)
        
        # 3. Citation vs Year scatter
        axes[1, 0].scatter(df['year'], df['citation_count'], alpha=0.6, color='red')
        axes[1, 0].set_title('Citations vs Publication Year')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Citation Count')
        axes[1, 0].set_yscale('log')
        
        # 4. H-index calculation visualization
        sorted_citations = sorted(df['citation_count'].values, reverse=True)
        h_index = 0
        for i, citations in enumerate(sorted_citations):
            if citations >= i + 1:
                h_index = i + 1
            else:
                break
        
        axes[1, 1].plot(range(1, len(sorted_citations) + 1), sorted_citations, 'b-', label='Citation count')
        axes[1, 1].plot(range(1, len(sorted_citations) + 1), range(1, len(sorted_citations) + 1), 'r--', label='Rank')
        axes[1, 1].axvline(x=h_index, color='green', linestyle=':', linewidth=2, label=f'H-index: {h_index}')
        axes[1, 1].set_title(f'H-index Analysis (H-index: {h_index})')
        axes[1, 1].set_xlabel('Paper Rank')
        axes[1, 1].set_ylabel('Citation Count')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{filename_prefix}_citation_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_author_analysis(self, df: pd.DataFrame, keyword: str, filename_prefix: str) -> str:
        """Create author analysis visualizations"""
        
        # Extract all authors
        all_authors = []
        for authors_list in df['authors']:
            if isinstance(authors_list, list):
                all_authors.extend(authors_list)
        
        author_counts = Counter(all_authors)
        top_authors = author_counts.most_common(20)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Author Analysis - {keyword}', fontsize=16, fontweight='bold')
        
        # 1. Top authors by publication count
        if top_authors:
            authors, counts = zip(*top_authors)
            axes[0, 0].barh(range(len(authors)), counts, alpha=0.7, color='teal')
            axes[0, 0].set_title('Top 20 Authors by Publication Count')
            axes[0, 0].set_xlabel('Number of Publications')
            axes[0, 0].set_yticks(range(len(authors)))
            axes[0, 0].set_yticklabels(authors, fontsize=8)
        
        # 2. Author collaboration network (sample)
        if len(top_authors) > 5:
            self._create_author_network(df, axes[0, 1], top_authors[:10])
        
        # 3. Authors per paper distribution
        author_count_dist = df['author_count'].value_counts().sort_index()
        axes[1, 0].bar(author_count_dist.index, author_count_dist.values, alpha=0.7, color='orange')
        axes[1, 0].set_title('Distribution of Authors per Paper')
        axes[1, 0].set_xlabel('Number of Authors')
        axes[1, 0].set_ylabel('Number of Papers')
        
        # 4. Author productivity vs citations
        author_productivity = defaultdict(lambda: {'papers': 0, 'citations': 0})
        for _, paper in df.iterrows():
            if isinstance(paper['authors'], list):
                for author in paper['authors']:
                    author_productivity[author]['papers'] += 1
                    author_productivity[author]['citations'] += paper['citation_count']
        
        # Plot for top productive authors
        top_productive = sorted(author_productivity.items(), key=lambda x: x[1]['papers'], reverse=True)[:20]
        if top_productive:
            papers_counts = [data['papers'] for _, data in top_productive]
            citation_counts = [data['citations'] for _, data in top_productive]
            
            axes[1, 1].scatter(papers_counts, citation_counts, alpha=0.7, color='red')
            axes[1, 1].set_title('Author Productivity vs Citations')
            axes[1, 1].set_xlabel('Number of Papers')
            axes[1, 1].set_ylabel('Total Citations')
            
            # Annotate top authors
            for i, (author, data) in enumerate(top_productive[:5]):
                axes[1, 1].annotate(author[:15] + '...' if len(author) > 15 else author, 
                                  (data['papers'], data['citations']), fontsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{filename_prefix}_author_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_author_network(self, df: pd.DataFrame, ax, top_authors: List[Tuple[str, int]]):
        """Create author collaboration network"""
        
        # Create co-authorship network
        G = nx.Graph()
        top_author_names = [author for author, _ in top_authors]
        
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
            pos = nx.spring_layout(G)
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
                   node_size=1000, font_size=8, font_weight='bold')
            ax.set_title('Author Collaboration Network')
        else:
            ax.text(0.5, 0.5, 'No collaboration data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Author Collaboration Network')
    
    def _create_venue_analysis(self, df: pd.DataFrame, keyword: str, filename_prefix: str) -> str:
        """Create venue analysis visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Venue Analysis - {keyword}', fontsize=16, fontweight='bold')
        
        # 1. Top venues by publication count
        venue_counts = df['venue'].value_counts().head(15)
        if len(venue_counts) > 0:
            axes[0, 0].barh(range(len(venue_counts)), venue_counts.values, alpha=0.7, color='purple')
            axes[0, 0].set_title('Top 15 Publication Venues')
            axes[0, 0].set_xlabel('Number of Publications')
            axes[0, 0].set_yticks(range(len(venue_counts)))
            axes[0, 0].set_yticklabels([venue[:40] + '...' if len(venue) > 40 else venue 
                                       for venue in venue_counts.index], fontsize=8)
        
        # 2. Domain distribution
        domain_counts = df['domain'].value_counts()
        if len(domain_counts) > 0:
            axes[0, 1].pie(domain_counts.values, labels=domain_counts.index, autopct='%1.1f%%', 
                          startangle=90)
            axes[0, 1].set_title('Research Domain Distribution')
        
        # 3. Venue impact (average citations per paper)
        venue_impact = df.groupby('venue').agg({
            'citation_count': ['mean', 'count']
        }).round(2)
        venue_impact.columns = ['avg_citations', 'paper_count']
        venue_impact = venue_impact[venue_impact['paper_count'] >= 2].sort_values('avg_citations', ascending=False).head(10)
        
        if len(venue_impact) > 0:
            axes[1, 0].barh(range(len(venue_impact)), venue_impact['avg_citations'].values, alpha=0.7, color='green')
            axes[1, 0].set_title('Top 10 Venues by Average Citations')
            axes[1, 0].set_xlabel('Average Citations per Paper')
            axes[1, 0].set_yticks(range(len(venue_impact)))
            axes[1, 0].set_yticklabels([venue[:30] + '...' if len(venue) > 30 else venue 
                                       for venue in venue_impact.index], fontsize=8)
        
        # 4. Publications vs Citations by domain
        domain_stats = df.groupby('domain').agg({
            'title': 'count',
            'citation_count': 'sum'
        }).rename(columns={'title': 'publications', 'citation_count': 'total_citations'})
        
        if len(domain_stats) > 0:
            axes[1, 1].scatter(domain_stats['publications'], domain_stats['total_citations'], 
                              alpha=0.7, s=100, color='red')
            axes[1, 1].set_title('Domain: Publications vs Citations')
            axes[1, 1].set_xlabel('Number of Publications')
            axes[1, 1].set_ylabel('Total Citations')
            
            for domain, stats in domain_stats.iterrows():
                axes[1, 1].annotate(domain, (stats['publications'], stats['total_citations']), 
                                   fontsize=8, ha='center')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{filename_prefix}_venue_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_text_analysis(self, df: pd.DataFrame, keyword: str, filename_prefix: str) -> str:
        """Create text analysis visualizations"""
        
        # Combine titles and abstracts for text analysis
        all_text = ' '.join(df['title'].fillna('') + ' ' + df['abstract'].fillna(''))
        
        # Clean text
        all_text = re.sub(r'[^\w\s]', ' ', all_text.lower())
        all_text = re.sub(r'\s+', ' ', all_text)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Text Analysis - {keyword}', fontsize=16, fontweight='bold')
        
        # 1. Word cloud
        if all_text:
            # Remove common stop words and add domain-specific ones
            stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                            'this', 'that', 'these', 'those', 'i', 'we', 'you', 'he', 'she', 'it', 'they',
                            'paper', 'study', 'research', 'method', 'approach', 'result', 'results', 'conclusion'])
            
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                stopwords=stop_words,
                                max_words=100,
                                collocations=False).generate(all_text)
            
            axes[0, 0].imshow(wordcloud, interpolation='bilinear')
            axes[0, 0].axis('off')
            axes[0, 0].set_title('Word Cloud')
        
        # 2. Most frequent words
        words = all_text.split()
        word_freq = Counter([word for word in words if len(word) > 3 and word not in 
                           ['paper', 'study', 'research', 'method', 'approach', 'result', 'results']])
        
        if word_freq:
            top_words = word_freq.most_common(20)
            words, counts = zip(*top_words)
            axes[0, 1].barh(range(len(words)), counts, alpha=0.7, color='orange')
            axes[0, 1].set_title('Top 20 Most Frequent Words')
            axes[0, 1].set_xlabel('Frequency')
            axes[0, 1].set_yticks(range(len(words)))
            axes[0, 1].set_yticklabels(words, fontsize=8)
        
        # 3. Title length distribution
        title_lengths = df['title_length']
        axes[1, 0].hist(title_lengths, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_title('Title Length Distribution')
        axes[1, 0].set_xlabel('Title Length (characters)')
        axes[1, 0].set_ylabel('Number of Papers')
        axes[1, 0].axvline(title_lengths.mean(), color='red', linestyle='--', 
                          label=f'Mean: {title_lengths.mean():.1f}')
        axes[1, 0].legend()
        
        # 4. Abstract length distribution  
        abstract_lengths = df['abstract_length']
        axes[1, 1].hist(abstract_lengths, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Abstract Length Distribution')
        axes[1, 1].set_xlabel('Abstract Length (characters)')
        axes[1, 1].set_ylabel('Number of Papers')
        axes[1, 1].axvline(abstract_lengths.mean(), color='red', linestyle='--', 
                          label=f'Mean: {abstract_lengths.mean():.1f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{filename_prefix}_text_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_relevance_analysis(self, df: pd.DataFrame, keyword: str, filename_prefix: str) -> str:
        """Create relevance analysis visualizations (only if relevance data available)"""
        
        # Filter papers with relevance data
        relevant_df = df[df['relevance_score'].notna()]
        
        if len(relevant_df) == 0:
            logger.warning("No relevance data available for analysis")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'AI Relevance Analysis - {keyword}', fontsize=16, fontweight='bold')
        
        # 1. Relevance score distribution
        axes[0, 0].hist(relevant_df['relevance_score'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 0].set_title('Relevance Score Distribution')
        axes[0, 0].set_xlabel('Relevance Score')
        axes[0, 0].set_ylabel('Number of Papers')
        axes[0, 0].axvline(relevant_df['relevance_score'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {relevant_df["relevance_score"].mean():.2f}')
        axes[0, 0].legend()
        
        # 2. Relevance vs Citations
        axes[0, 1].scatter(relevant_df['relevance_score'], relevant_df['citation_count'], alpha=0.6)
        axes[0, 1].set_title('Relevance Score vs Citation Count')
        axes[0, 1].set_xlabel('Relevance Score')
        axes[0, 1].set_ylabel('Citation Count')
        axes[0, 1].set_yscale('log')
        
        # Add trend line
        z = np.polyfit(relevant_df['relevance_score'], relevant_df['citation_count'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(relevant_df['relevance_score'], p(relevant_df['relevance_score']), "r--", alpha=0.8)
        
        # 3. Relevance over time
        yearly_relevance = relevant_df.groupby('year')['relevance_score'].mean()
        axes[1, 0].plot(yearly_relevance.index, yearly_relevance.values, marker='o', linewidth=2)
        axes[1, 0].set_title('Average Relevance Score Over Time')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Average Relevance Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. High relevance papers by year
        high_relevance = relevant_df[relevant_df['relevance_score'] >= 0.7]
        yearly_high_relevance = high_relevance['year'].value_counts().sort_index()
        
        if len(yearly_high_relevance) > 0:
            axes[1, 1].bar(yearly_high_relevance.index, yearly_high_relevance.values, 
                          alpha=0.7, color='green')
            axes[1, 1].set_title('Highly Relevant Papers by Year (Score ≥ 0.7)')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Number of Highly Relevant Papers')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{filename_prefix}_relevance_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_network_analysis(self, df: pd.DataFrame, keyword: str, filename_prefix: str) -> str:
        """Create network analysis visualization"""
        
        # Create citation network (simplified)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Network Analysis - {keyword}', fontsize=16, fontweight='bold')
        
        # 1. Author collaboration network
        G_authors = nx.Graph()
        for _, paper in df.iterrows():
            if isinstance(paper['authors'], list) and len(paper['authors']) > 1:
                authors = paper['authors'][:5]  # Limit to first 5 authors for visualization
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        if G_authors.has_edge(authors[i], authors[j]):
                            G_authors[authors[i]][authors[j]]['weight'] += 1
                        else:
                            G_authors.add_edge(authors[i], authors[j], weight=1)
        
        # Filter to most connected authors
        if G_authors.number_of_nodes() > 50:
            # Keep only nodes with degree > 1
            nodes_to_keep = [node for node, degree in dict(G_authors.degree()).items() if degree > 1]
            G_authors = G_authors.subgraph(nodes_to_keep[:50])
        
        if G_authors.number_of_nodes() > 0:
            pos = nx.spring_layout(G_authors, k=1, iterations=50)
            nx.draw(G_authors, pos, ax=axes[0], 
                   node_size=100, node_color='lightblue', 
                   font_size=6, with_labels=True, alpha=0.7)
            axes[0].set_title('Author Collaboration Network')
        else:
            axes[0].text(0.5, 0.5, 'Insufficient collaboration data', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Author Collaboration Network')
        
        # 2. Research topic network (using key topics if available)
        topic_network = nx.Graph()
        
        # Extract topics from key_topics field or titles
        all_topics = []
        for _, paper in df.iterrows():
            if paper.get('key_topics') and isinstance(paper['key_topics'], list):
                all_topics.extend(paper['key_topics'])
            else:
                # Extract topics from title
                title_words = paper['title'].lower().split()
                # Filter meaningful words
                meaningful_words = [word for word in title_words 
                                  if len(word) > 3 and word not in ['with', 'using', 'based', 'from']]
                all_topics.extend(meaningful_words[:3])  # Top 3 words from title
        
        # Create topic co-occurrence network
        topic_counter = Counter(all_topics)
        common_topics = [topic for topic, count in topic_counter.most_common(20)]
        
        # Add edges between topics that appear in same papers
        for _, paper in df.iterrows():
            paper_topics = []
            if paper.get('key_topics') and isinstance(paper['key_topics'], list):
                paper_topics = [topic for topic in paper['key_topics'] if topic in common_topics]
            else:
                title_words = paper['title'].lower().split()
                paper_topics = [word for word in title_words if word in common_topics]
            
            for i in range(len(paper_topics)):
                for j in range(i + 1, len(paper_topics)):
                    topic1, topic2 = paper_topics[i], paper_topics[j]
                    if topic_network.has_edge(topic1, topic2):
                        topic_network[topic1][topic2]['weight'] += 1
                    else:
                        topic_network.add_edge(topic1, topic2, weight=1)
        
        if topic_network.number_of_nodes() > 0:
            pos = nx.spring_layout(topic_network, k=2, iterations=50)
            edge_weights = [topic_network[u][v]['weight'] for u, v in topic_network.edges()]
            nx.draw(topic_network, pos, ax=axes[1],
                   node_size=300, node_color='lightcoral',
                   font_size=8, with_labels=True, alpha=0.7,
                   width=[w * 0.5 for w in edge_weights])
            axes[1].set_title('Research Topic Co-occurrence Network')
        else:
            axes[1].text(0.5, 0.5, 'Insufficient topic data', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Research Topic Co-occurrence Network')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{filename_prefix}_network_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_cutting_edge_analysis(self, df: pd.DataFrame, keyword: str, filename_prefix: str) -> str:
        """Create cutting-edge analysis focusing on recent trends, Bertology, and LLM research"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Cutting-Edge Analysis - {keyword}', fontsize=16, fontweight='bold')
        
        # 1. Recent trends (last 5 years focus)
        recent_years = df[df['year'] >= 2019]  # Focus on recent research
        if len(recent_years) > 0:
            yearly_trend = recent_years['year'].value_counts().sort_index()
            axes[0, 0].plot(yearly_trend.index, yearly_trend.values, marker='o', linewidth=3, markersize=8)
            axes[0, 0].set_title('Recent Research Trends (2019-present)')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Number of Publications')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. LLM/Transformer related keywords analysis
        llm_keywords = ['transformer', 'bert', 'gpt', 'llm', 'large language model', 
                       'attention', 'pretrained', 'fine-tuning', 'neural language']
        
        # Count papers mentioning LLM-related terms
        llm_mentions = {keyword: 0 for keyword in llm_keywords}
        for _, paper in df.iterrows():
            text = (paper['title'] + ' ' + paper['abstract']).lower()
            for kw in llm_keywords:
                if kw in text:
                    llm_mentions[kw] += 1
        
        # Plot LLM keyword frequency
        kw_items = [(k, v) for k, v in llm_mentions.items() if v > 0]
        if kw_items:
            kw_items.sort(key=lambda x: x[1], reverse=True)
            keywords, counts = zip(*kw_items)
            axes[0, 1].barh(range(len(keywords)), counts, alpha=0.7, color='red')
            axes[0, 1].set_title('LLM/Transformer Keywords Frequency')
            axes[0, 1].set_xlabel('Number of Papers')
            axes[0, 1].set_yticks(range(len(keywords)))
            axes[0, 1].set_yticklabels(keywords, fontsize=8)
        
        # 3. Emerging topics analysis (recent vs older papers)
        if len(df) > 20:
            # Split data into recent and older
            median_year = df['year'].median()
            recent_df = df[df['year'] > median_year]
            older_df = df[df['year'] <= median_year]
            
            # Analyze title words for emerging topics
            recent_text = ' '.join(recent_df['title'].fillna(''))
            older_text = ' '.join(older_df['title'].fillna(''))
            
            recent_words = Counter(recent_text.lower().split())
            older_words = Counter(older_text.lower().split())
            
            # Find words that are more common in recent papers
            emerging_topics = []
            for word, recent_count in recent_words.items():
                if len(word) > 3 and word not in ['the', 'and', 'for', 'with', 'using', 'based']:
                    older_count = older_words.get(word, 0)
                    if recent_count > older_count and recent_count >= 2:
                        emergence_ratio = recent_count / max(older_count, 1)
                        emerging_topics.append((word, emergence_ratio, recent_count))
            
            emerging_topics.sort(key=lambda x: x[1], reverse=True)
            top_emerging = emerging_topics[:15]
            
            if top_emerging:
                topics, ratios, _ = zip(*top_emerging)
                axes[1, 0].barh(range(len(topics)), ratios, alpha=0.7, color='green')
                axes[1, 0].set_title('Emerging Topics (Recent vs Historical)')
                axes[1, 0].set_xlabel('Emergence Ratio')
                axes[1, 0].set_yticks(range(len(topics)))
                axes[1, 0].set_yticklabels(topics, fontsize=8)
        
        # 4. High-impact recent papers
        recent_cited = df[df['year'] >= 2020].nlargest(10, 'citation_count')
        if len(recent_cited) > 0:
            axes[1, 1].scatter(recent_cited['year'], recent_cited['citation_count'], 
                              s=100, alpha=0.7, color='purple')
            axes[1, 1].set_title('High-Impact Recent Papers (2020+)')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Citation Count')
            
            # Annotate top papers
            for _, paper in recent_cited.head(3).iterrows():
                title = paper['title'][:30] + '...' if len(paper['title']) > 30 else paper['title']
                axes[1, 1].annotate(title, (paper['year'], paper['citation_count']), 
                                   fontsize=6, ha='center', rotation=15)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{filename_prefix}_cutting_edge_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_interactive_dashboard(self, df: pd.DataFrame, keyword: str, filename_prefix: str) -> str:
        """Create interactive dashboard using Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Publications Over Time', 'Citation Distribution',
                          'Top Authors', 'Research Domains',
                          'Citation vs Year', 'Relevance Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Publications over time
        yearly_counts = df['year'].value_counts().sort_index()
        fig.add_trace(
            go.Scatter(x=yearly_counts.index, y=yearly_counts.values,
                      mode='lines+markers', name='Publications',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 2. Citation distribution
        fig.add_trace(
            go.Histogram(x=df['citation_count'], nbinsx=30, name='Citations',
                        marker_color='orange', opacity=0.7),
            row=1, col=2
        )
        
        # 3. Top authors
        all_authors = []
        for authors_list in df['authors']:
            if isinstance(authors_list, list):
                all_authors.extend(authors_list)
        
        author_counts = Counter(all_authors).most_common(10)
        if author_counts:
            authors, counts = zip(*author_counts)
            fig.add_trace(
                go.Bar(x=counts, y=authors, orientation='h', name='Top Authors',
                      marker_color='green'),
                row=2, col=1
            )
        
        # 4. Research domains
        domain_counts = df['domain'].value_counts()
        if len(domain_counts) > 0:
            fig.add_trace(
                go.Pie(labels=domain_counts.index, values=domain_counts.values,
                      name='Domains'),
                row=2, col=2
            )
        
        # 5. Citation vs Year scatter
        fig.add_trace(
            go.Scatter(x=df['year'], y=df['citation_count'],
                      mode='markers', name='Papers',
                      marker=dict(size=6, opacity=0.6, color='red'),
                      text=df['title'],  # Hover text
                      hovertemplate='<b>%{text}</b><br>Year: %{x}<br>Citations: %{y}<extra></extra>'),
            row=3, col=1
        )
        
        # 6. Relevance analysis (if available)
        if 'relevance_score' in df.columns and df['relevance_score'].notna().any():
            relevance_df = df[df['relevance_score'].notna()]
            fig.add_trace(
                go.Scatter(x=relevance_df['relevance_score'], y=relevance_df['citation_count'],
                          mode='markers', name='Relevance',
                          marker=dict(size=6, opacity=0.6, color='purple'),
                          text=relevance_df['title'],
                          hovertemplate='<b>%{text}</b><br>Relevance: %{x}<br>Citations: %{y}<extra></extra>'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"Interactive Bibliometrics Dashboard - {keyword}",
            title_x=0.5,
            showlegend=False,
            height=1000,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Publications", row=1, col=1)
        
        fig.update_xaxes(title_text="Citation Count", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Publications", row=2, col=1)
        
        fig.update_xaxes(title_text="Year", row=3, col=1)
        fig.update_yaxes(title_text="Citations", row=3, col=1)
        
        if 'relevance_score' in df.columns:
            fig.update_xaxes(title_text="Relevance Score", row=3, col=2)
            fig.update_yaxes(title_text="Citations", row=3, col=2)
        
        # Save interactive plot
        output_path = os.path.join(self.output_dir, f'{filename_prefix}_interactive_dashboard.html')
        fig.write_html(output_path)
        
        return output_path
    
    def _generate_analysis_report(self, df: pd.DataFrame, keyword: str, viz_paths: Dict[str, str], filename_prefix: str) -> str:
        """Generate comprehensive analysis report"""
        
        # Calculate key statistics
        total_papers = len(df)
        year_range = f"{df['year'].min()}-{df['year'].max()}" if len(df) > 0 else "N/A"
        total_citations = df['citation_count'].sum()
        avg_citations = df['citation_count'].mean()
        median_citations = df['citation_count'].median()
        
        # Author statistics
        all_authors = []
        for authors_list in df['authors']:
            if isinstance(authors_list, list):
                all_authors.extend(authors_list)
        unique_authors = len(set(all_authors))
        
        # Domain statistics
        domain_dist = df['domain'].value_counts().to_dict()
        
        # Recent trends
        recent_papers = len(df[df['year'] >= 2020]) if len(df) > 0 else 0
        
        # Relevance statistics (if available)
        relevance_stats = {}
        if 'relevance_score' in df.columns and df['relevance_score'].notna().any():
            relevant_df = df[df['relevance_score'].notna()]
            relevance_stats = {
                'total_analyzed': len(relevant_df),
                'highly_relevant': len(relevant_df[relevant_df['relevance_score'] >= 0.7]),
                'avg_relevance': relevant_df['relevance_score'].mean(),
                'relevance_rate': len(relevant_df[relevant_df['is_relevant'] == True]) / len(relevant_df) if len(relevant_df) > 0 else 0
            }
        
        # Generate HTML report
        report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bibliometric Analysis Report - {keyword}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2 {{ color: #333; }}
        .stat {{ background: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .highlight {{ background: #e8f5e8; padding: 15px; border-left: 4px solid #4CAF50; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .visualization {{ margin: 20px 0; }}
        .viz-link {{ display: inline-block; margin: 5px; padding: 8px 12px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }}
        .viz-link:hover {{ background: #0056b3; }}
    </style>
</head>
<body>
    <h1>Bibliometric Analysis Report</h1>
    <h2>Keyword: {keyword}</h2>
    <p><em>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    
    <div class="highlight">
        <h3>Executive Summary</h3>
        <p>This report analyzes <strong>{total_papers}</strong> research papers related to "<strong>{keyword}</strong>" 
        spanning from <strong>{year_range}</strong>. The analysis includes temporal trends, citation patterns, 
        author collaborations, and emerging research topics.</p>
    </div>
    
    <h3>Key Statistics</h3>
    <div class="stat">
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Papers</td><td>{total_papers}</td></tr>
            <tr><td>Year Range</td><td>{year_range}</td></tr>
            <tr><td>Unique Authors</td><td>{unique_authors}</td></tr>
            <tr><td>Total Citations</td><td>{total_citations:,}</td></tr>
            <tr><td>Average Citations per Paper</td><td>{avg_citations:.1f}</td></tr>
            <tr><td>Median Citations per Paper</td><td>{median_citations:.1f}</td></tr>
            <tr><td>Recent Papers (2020+)</td><td>{recent_papers}</td></tr>
        </table>
    </div>
    
    <h3>Research Domain Distribution</h3>
    <div class="stat">
        <table>
            <tr><th>Domain</th><th>Papers</th><th>Percentage</th></tr>
"""
        
        for domain, count in domain_dist.items():
            percentage = (count / total_papers) * 100
            report_html += f"<tr><td>{domain}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        report_html += "</table></div>"
        
        # Add relevance statistics if available
        if relevance_stats:
            report_html += f"""
    <h3>AI Relevance Analysis</h3>
    <div class="stat">
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Papers Analyzed</td><td>{relevance_stats['total_analyzed']}</td></tr>
            <tr><td>Highly Relevant (≥0.7)</td><td>{relevance_stats['highly_relevant']}</td></tr>
            <tr><td>Average Relevance Score</td><td>{relevance_stats['avg_relevance']:.2f}</td></tr>
            <tr><td>Relevance Rate</td><td>{relevance_stats['relevance_rate']:.1%}</td></tr>
        </table>
    </div>
"""
        
        # Add visualization links
        report_html += """
    <h3>Generated Visualizations</h3>
    <div class="visualization">
        <p>The following visualizations have been generated as part of this analysis:</p>
"""
        
        viz_descriptions = {
            'temporal': 'Temporal Analysis - Publication trends over time',
            'citations': 'Citation Analysis - Citation patterns and distribution',
            'authors': 'Author Analysis - Top authors and collaboration patterns',
            'venues': 'Venue Analysis - Publication venues and domains',
            'text': 'Text Analysis - Word clouds and text metrics',
            'relevance': 'Relevance Analysis - AI-based relevance assessment',
            'network': 'Network Analysis - Collaboration and topic networks',
            'cutting_edge': 'Cutting-Edge Analysis - Recent trends and emerging topics',
            'dashboard': 'Interactive Dashboard - Comprehensive interactive visualization'
        }
        
        for viz_key, path in viz_paths.items():
            if path and viz_key in viz_descriptions:
                filename = os.path.basename(path)
                report_html += f'<a href="{filename}" class="viz-link">{viz_descriptions[viz_key]}</a><br>'
        
        report_html += """
    </div>
    
    <h3>Methodology</h3>
    <div class="stat">
        <p><strong>Data Collection:</strong> Papers were collected using Google Scholar API via SerpApi.</p>
        <p><strong>AI Analysis:</strong> Relevance analysis performed using DeepSeek API for semantic understanding.</p>
        <p><strong>Visualization:</strong> Generated using matplotlib, seaborn, plotly, and networkx libraries.</p>
        <p><strong>Text Analysis:</strong> Includes word frequency, topic extraction, and trend analysis.</p>
    </div>
    
    <h3>Recommendations</h3>
    <div class="highlight">
        <ul>
            <li>Focus on highly cited recent papers for cutting-edge insights</li>
            <li>Explore collaboration opportunities with top authors in the field</li>
            <li>Consider research gaps in under-represented domains</li>
            <li>Monitor emerging topics for future research directions</li>
        </ul>
    </div>
    
    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
        <p>Generated by Bibliometrics Analyzer v1.0</p>
    </footer>
</body>
</html>
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, f'{filename_prefix}_analysis_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Analysis report saved to: {report_path}")
        return report_path

def main():
    """Test function for bibliometrics analyzer"""
    import sys
    if len(sys.argv) < 3:
        print("Usage: python bibliometrics.py <json_file_path> <keyword>")
        return
    
    json_file = sys.argv[1]
    keyword = sys.argv[2]
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data.get("results", [])
        analyzed_papers = data.get("analyzed_papers", None)
        
        analyzer = BibliometricsAnalyzer()
        viz_paths = analyzer.generate_comprehensive_analysis(papers, keyword, analyzed_papers)
        
        print("\nBibliometric analysis completed!")
        print("Generated visualizations:")
        for viz_type, path in viz_paths.items():
            if path:
                print(f"  {viz_type}: {path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()