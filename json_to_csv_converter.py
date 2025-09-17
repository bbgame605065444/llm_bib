#!/usr/bin/env python3
"""
Convert education analysis JSON to multiple CSV files
Processes: /home/campus.ncl.ac.uk/c1041562/Desktop/codes/article_agent/bibm/output/20250831_024823_education_education_analysis.json
"""

import json
import pandas as pd
import os
from datetime import datetime

def load_json_data(json_file_path):
    """Load and parse the JSON file"""
    print(f"Loading data from: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_analysis_summary_csv(analysis_report, output_dir):
    """Create CSV with analysis metadata and statistics"""
    metadata = analysis_report.get('analysis_metadata', {})
    stats = analysis_report.get('relevance_statistics', {})
    
    summary_data = {
        'Metric': [
            'Keyword',
            'Total Papers',
            'Analysis Date',
            'Analyzer Version',
            'Relevant Papers',
            'Relevance Rate',
            'Average Relevance Score',
            'Successful Analyses',
            'Success Rate'
        ],
        'Value': [
            metadata.get('keyword', ''),
            metadata.get('total_papers', 0),
            metadata.get('analysis_date', ''),
            metadata.get('analyzer_version', ''),
            stats.get('relevant_papers', 0),
            f"{stats.get('relevance_rate', 0):.4f}",
            f"{stats.get('average_relevance_score', 0):.4f}",
            stats.get('successful_analyses', 0),
            f"{stats.get('success_rate', 0):.4f}"
        ]
    }
    
    df = pd.DataFrame(summary_data)
    output_path = os.path.join(output_dir, 'analysis_summary.csv')
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path}")
    return df

def create_top_topics_csv(analysis_report, output_dir):
    """Create CSV with top topics and their frequencies"""
    top_topics = analysis_report.get('top_topics', [])
    
    topics_data = {
        'Topic': [topic[0] for topic in top_topics],
        'Frequency': [topic[1] for topic in top_topics],
        'Rank': list(range(1, len(top_topics) + 1))
    }
    
    df = pd.DataFrame(topics_data)
    output_path = os.path.join(output_dir, 'top_topics.csv')
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path}")
    return df

def create_score_distribution_csv(analysis_report, output_dir):
    """Create CSV with relevance score distribution"""
    score_dist = analysis_report.get('score_distribution', {})
    
    dist_data = {
        'Score_Range': list(score_dist.keys()),
        'Paper_Count': list(score_dist.values())
    }
    
    df = pd.DataFrame(dist_data)
    # Calculate percentages
    total_papers = df['Paper_Count'].sum()
    df['Percentage'] = (df['Paper_Count'] / total_papers * 100).round(2)
    
    output_path = os.path.join(output_dir, 'score_distribution.csv')
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path}")
    return df

def create_highly_relevant_papers_csv(analysis_report, output_dir):
    """Create CSV with highly relevant papers from analysis report"""
    highly_relevant = analysis_report.get('highly_relevant_papers', [])
    
    if not highly_relevant:
        print("No highly relevant papers found in analysis report")
        return None
    
    papers_data = {
        'Title': [paper.get('title', '') for paper in highly_relevant],
        'Relevance_Score': [paper.get('relevance_score', 0) for paper in highly_relevant],
        'Explanation': [paper.get('explanation', '') for paper in highly_relevant]
    }
    
    df = pd.DataFrame(papers_data)
    output_path = os.path.join(output_dir, 'highly_relevant_papers.csv')
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path}")
    return df

def create_all_papers_csv(papers_data, output_dir):
    """Create comprehensive CSV with all papers"""
    if not papers_data:
        print("No papers data found")
        return None
    
    # Extract data from papers array
    papers_list = []
    for paper in papers_data:
        paper_info = {
            'Title': paper.get('title', ''),
            'Authors': '; '.join(paper.get('authors', [])),
            'Year': paper.get('year', ''),
            'Citation_Count': paper.get('citation_count', 0),
            'Venue': paper.get('venue', ''),
            'Link': paper.get('link', ''),
            'Abstract': paper.get('abstract', ''),
            'Relevance_Score': paper.get('relevance_score', 0),
            'Is_Relevant': paper.get('is_relevant', False),
            'Relevance_Explanation': paper.get('relevance_explanation', ''),
            'Key_Topics': '; '.join(paper.get('key_topics', [])),
            'Confidence': paper.get('confidence', 0),
            'Analysis_Status': paper.get('analysis_status', ''),
            'Search_Keyword': paper.get('search_keyword', ''),
            'Scraped_At': paper.get('scraped_at', '')
        }
        papers_list.append(paper_info)
    
    df = pd.DataFrame(papers_list)
    output_path = os.path.join(output_dir, 'all_papers.csv')
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path} with {len(df)} papers")
    return df

def create_relevant_papers_only_csv(papers_data, output_dir, threshold=0.7):
    """Create CSV with only relevant papers above threshold"""
    if not papers_data:
        return None
    
    relevant_papers = [
        paper for paper in papers_data 
        if paper.get('is_relevant', False) and paper.get('relevance_score', 0) >= threshold
    ]
    
    if not relevant_papers:
        print(f"No papers found with relevance score >= {threshold}")
        return None
    
    papers_list = []
    for paper in relevant_papers:
        paper_info = {
            'Title': paper.get('title', ''),
            'Authors': '; '.join(paper.get('authors', [])),
            'Year': paper.get('year', ''),
            'Citation_Count': paper.get('citation_count', 0),
            'Venue': paper.get('venue', ''),
            'Link': paper.get('link', ''),
            'Abstract': paper.get('abstract', ''),
            'Relevance_Score': paper.get('relevance_score', 0),
            'Relevance_Explanation': paper.get('relevance_explanation', ''),
            'Key_Topics': '; '.join(paper.get('key_topics', [])),
            'Confidence': paper.get('confidence', 0)
        }
        papers_list.append(paper_info)
    
    # Sort by relevance score descending
    df = pd.DataFrame(papers_list)
    df = df.sort_values('Relevance_Score', ascending=False)
    
    output_path = os.path.join(output_dir, f'relevant_papers_above_{threshold}.csv')
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path} with {len(df)} relevant papers")
    return df

def create_author_analysis_csv(papers_data, output_dir):
    """Create CSV with author analysis"""
    if not papers_data:
        return None
    
    author_stats = {}
    for paper in papers_data:
        authors = paper.get('authors', [])
        relevance_score = paper.get('relevance_score', 0)
        
        for author in authors:
            if author not in author_stats:
                author_stats[author] = {
                    'paper_count': 0,
                    'total_relevance': 0,
                    'relevant_papers': 0,
                    'citations': 0
                }
            
            author_stats[author]['paper_count'] += 1
            author_stats[author]['total_relevance'] += relevance_score
            if paper.get('is_relevant', False):
                author_stats[author]['relevant_papers'] += 1
            author_stats[author]['citations'] += paper.get('citation_count', 0)
    
    # Convert to DataFrame
    authors_list = []
    for author, stats in author_stats.items():
        if stats['paper_count'] > 0:  # Only include authors with papers
            authors_list.append({
                'Author': author,
                'Total_Papers': stats['paper_count'],
                'Relevant_Papers': stats['relevant_papers'],
                'Relevance_Rate': stats['relevant_papers'] / stats['paper_count'],
                'Average_Relevance_Score': stats['total_relevance'] / stats['paper_count'],
                'Total_Citations': stats['citations'],
                'Average_Citations_Per_Paper': stats['citations'] / stats['paper_count']
            })
    
    if not authors_list:
        print("No author data available")
        return None
    
    df = pd.DataFrame(authors_list)
    df = df.sort_values('Relevant_Papers', ascending=False)
    
    output_path = os.path.join(output_dir, 'author_analysis.csv')
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path} with {len(df)} authors")
    return df

def create_yearly_trends_csv(papers_data, output_dir):
    """Create CSV with yearly publication trends"""
    if not papers_data:
        return None
    
    yearly_stats = {}
    for paper in papers_data:
        year = paper.get('year', 'Unknown')
        if isinstance(year, (int, float)) or (isinstance(year, str) and year.isdigit()):
            year = int(year)
        else:
            year = 'Unknown'
        
        if year not in yearly_stats:
            yearly_stats[year] = {
                'total_papers': 0,
                'relevant_papers': 0,
                'total_relevance': 0,
                'total_citations': 0
            }
        
        yearly_stats[year]['total_papers'] += 1
        if paper.get('is_relevant', False):
            yearly_stats[year]['relevant_papers'] += 1
        yearly_stats[year]['total_relevance'] += paper.get('relevance_score', 0)
        yearly_stats[year]['total_citations'] += paper.get('citation_count', 0)
    
    # Convert to DataFrame
    yearly_list = []
    for year, stats in yearly_stats.items():
        if stats['total_papers'] > 0:
            yearly_list.append({
                'Year': year,
                'Total_Papers': stats['total_papers'],
                'Relevant_Papers': stats['relevant_papers'],
                'Relevance_Rate': stats['relevant_papers'] / stats['total_papers'],
                'Average_Relevance_Score': stats['total_relevance'] / stats['total_papers'],
                'Total_Citations': stats['total_citations'],
                'Average_Citations_Per_Paper': stats['total_citations'] / stats['total_papers']
            })
    
    if not yearly_list:
        print("No yearly data available")
        return None
    
    df = pd.DataFrame(yearly_list)
    # Sort by year, handling 'Unknown' values
    df_known = df[df['Year'] != 'Unknown'].sort_values('Year')
    df_unknown = df[df['Year'] == 'Unknown']
    df = pd.concat([df_known, df_unknown], ignore_index=True)
    
    output_path = os.path.join(output_dir, 'yearly_trends.csv')
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path} with {len(df)} years")
    return df

def main():
    # Input file path
    json_file_path = "/home/campus.ncl.ac.uk/c1041562/Desktop/codes/article_agent/bibm/output/20250831_024823_education_education_analysis.json"
    
    # Output directory
    output_dir = "/home/campus.ncl.ac.uk/c1041562/Desktop/codes/article_agent/csv_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Education Analysis JSON to CSV Converter ===")
    print(f"Input file: {json_file_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data
    try:
        data = load_json_data(json_file_path)
        print(f"Successfully loaded JSON data")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Extract main components
    analysis_report = data.get('analysis_report', {})
    
    # The papers data is the main array in the JSON (everything except analysis_report)
    if isinstance(data, list):
        papers_data = data
    else:
        # Find the papers array - it should be the large array in the JSON
        papers_data = []
        for key, value in data.items():
            if key != 'analysis_report' and isinstance(value, list):
                papers_data = value
                break
        
        # If still not found, try to find it as a direct array at root level
        if not papers_data and isinstance(data, dict):
            # Look for the papers in the JSON structure
            for value in data.values():
                if isinstance(value, list) and len(value) > 100:  # Assuming papers array is large
                    papers_data = value
                    break
    
    print(f"Found {len(papers_data)} papers in the dataset")
    print()
    
    # Generate CSV files
    print("Generating CSV files...")
    print("-" * 40)
    
    # 1. Analysis summary
    create_analysis_summary_csv(analysis_report, output_dir)
    
    # 2. Top topics
    create_top_topics_csv(analysis_report, output_dir)
    
    # 3. Score distribution
    create_score_distribution_csv(analysis_report, output_dir)
    
    # 4. Highly relevant papers from analysis report
    create_highly_relevant_papers_csv(analysis_report, output_dir)
    
    # 5. All papers
    create_all_papers_csv(papers_data, output_dir)
    
    # 6. Relevant papers only (threshold 0.7)
    create_relevant_papers_only_csv(papers_data, output_dir, threshold=0.7)
    
    # 7. High relevance papers (threshold 0.9)
    create_relevant_papers_only_csv(papers_data, output_dir, threshold=0.9)
    
    # 8. Author analysis
    create_author_analysis_csv(papers_data, output_dir)
    
    # 9. Yearly trends
    create_yearly_trends_csv(papers_data, output_dir)
    
    print()
    print("-" * 40)
    print("✅ Conversion completed successfully!")
    print(f"All CSV files have been saved to: {output_dir}")
    print()
    
    # List all created files
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    print("Created files:")
    for file in sorted(csv_files):
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"  📄 {file} ({file_size:,} bytes)")

if __name__ == "__main__":
    main()