#!/usr/bin/env python3
"""
Article Agent - Comprehensive Academic Research Analysis Tool

This tool performs the following operations:
1. Searches Google Scholar for papers related to a keyword
2. Retrieves up to 3000 results and saves them in JSON format
3. Uses DeepSeek API to analyze each paper's relevance to the keyword
4. Generates comprehensive bibliometric visualizations including cutting-edge analysis

Usage: python main.py <keyword> [options]
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import traceback

# Import our modules
from scholar_scraper import ScholarScraper
from ai_analyzer import DeepSeekAnalyzer  
from bibliometrics import BibliometricsAnalyzer
from config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('article_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ArticleAgent:
    def __init__(self):
        """Initialize the Article Agent with necessary components"""
        Config.validate_config()
        logger.info("Configuration validated successfully")
        
        self.scraper = None
        self.analyzer = None
        self.visualizer = None
        self.output_dir = "output"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _extract_top_keywords(self, keyword_string: str, num_words: int = 3) -> str:
        """Extract top N meaningful keywords from search term"""
        import re
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
        
    def initialize_components(self):
        """Initialize all components"""
        try:
            logger.info("Initializing components...")
            
            # Initialize scraper
            self.scraper = ScholarScraper(Config.SERPAPI_KEY)
            logger.info("Scholar scraper initialized")
            
            # Initialize AI analyzer with updated DeepSeek-V3.1 configuration
            self.analyzer = DeepSeekAnalyzer(
                api_key=Config.DEEPSEEK_API_KEY, 
                base_url=Config.DEEPSEEK_BASE_URL,
                use_reasoning=Config.USE_REASONING_MODE
            )
            logger.info(f"DeepSeek analyzer initialized with model: {Config.DEEPSEEK_MODEL if not Config.USE_REASONING_MODE else Config.DEEPSEEK_MODEL_REASONING}")
            
            # Initialize visualizer
            viz_output_dir = os.path.join(self.output_dir, "visualizations")
            self.visualizer = BibliometricsAnalyzer(viz_output_dir)
            logger.info("Bibliometrics analyzer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise e
    
    def search_and_collect_papers(self, keyword: str, max_results: int = Config.MAX_RESULTS) -> Tuple[List[Dict], str]:
        """
        Search Google Scholar and collect papers
        
        Args:
            keyword: Search keyword
            max_results: Maximum number of results to collect
            
        Returns:
            Tuple of (papers_list, json_file_path)
        """
        logger.info(f"Starting paper collection for keyword: '{keyword}'")
        logger.info(f"Target: {max_results} papers")
        
        try:
            # Search for papers
            papers = self.scraper.search_scholar(keyword, max_results)
            
            if not papers:
                logger.warning("No papers found!")
                return [], ""
            
            # Generate filename with datetime and top 3 keywords
            filename_prefix = self._generate_filename_prefix(keyword)
            json_filename = os.path.join(self.output_dir, f"{filename_prefix}_papers.json")
            
            json_path = self.scraper.save_results(json_filename, keyword)
            
            logger.info(f"Successfully collected {len(papers)} papers")
            logger.info(f"Results saved to: {json_path}")
            
            # Print collection summary
            stats = self.scraper.get_statistics()
            logger.info("Collection Summary:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            
            return papers, json_path
            
        except Exception as e:
            logger.error(f"Error during paper collection: {str(e)}")
            raise e
    
    def analyze_paper_relevance(self, papers: List[Dict], keyword: str) -> Tuple[List[Dict], str]:
        """
        Analyze paper relevance using DeepSeek AI
        
        Args:
            papers: List of paper dictionaries
            keyword: Search keyword
            
        Returns:
            Tuple of (analyzed_papers, analysis_json_path)
        """
        logger.info(f"Starting relevance analysis for {len(papers)} papers")
        
        try:
            # Perform relevance analysis
            analyzed_papers = self.analyzer.analyze_relevance(papers, keyword)
            
            # Generate analysis report
            report = self.analyzer.generate_analysis_report(analyzed_papers, keyword)
            
            # Save analysis results using new filename convention
            filename_prefix = self._generate_filename_prefix(keyword)
            analysis_filename = os.path.join(self.output_dir, f"{filename_prefix}_analysis.json")
            
            analysis_path = self.analyzer.save_analyzed_results(analyzed_papers, analysis_filename, report)
            
            # Print analysis summary
            logger.info("Analysis Summary:")
            logger.info(f"  Total papers analyzed: {report['relevance_statistics']['relevant_papers']}")
            logger.info(f"  Relevant papers: {report['relevance_statistics']['relevant_papers']}")
            logger.info(f"  Relevance rate: {report['relevance_statistics']['relevance_rate']:.2%}")
            logger.info(f"  Average relevance score: {report['relevance_statistics']['average_relevance_score']:.2f}")
            
            return analyzed_papers, analysis_path
            
        except Exception as e:
            logger.error(f"Error during relevance analysis: {str(e)}")
            # Return papers without analysis if AI analysis fails
            logger.warning("Continuing without relevance analysis...")
            return papers, ""
    
    def generate_visualizations(self, papers: List[Dict], keyword: str, analyzed_papers: Optional[List[Dict]] = None) -> Dict[str, str]:
        """
        Generate comprehensive bibliometric visualizations
        
        Args:
            papers: Original papers list
            keyword: Search keyword  
            analyzed_papers: Papers with relevance analysis (optional)
            
        Returns:
            Dictionary of visualization paths
        """
        logger.info("Starting comprehensive visualization generation")
        
        try:
            # Use analyzed papers if available, otherwise use original papers
            data_for_viz = analyzed_papers or papers
            
            # Generate all visualizations
            viz_paths = self.visualizer.generate_comprehensive_analysis(
                papers=papers,
                keyword=keyword,
                analyzed_papers=analyzed_papers
            )
            
            logger.info("Visualization generation completed")
            logger.info("Generated visualizations:")
            for viz_type, path in viz_paths.items():
                if path:
                    logger.info(f"  {viz_type}: {os.path.basename(path)}")
            
            return viz_paths
            
        except Exception as e:
            logger.error(f"Error during visualization generation: {str(e)}")
            raise e
    
    def run_complete_analysis(self, keyword: str, max_results: int = Config.MAX_RESULTS, 
                            skip_ai_analysis: bool = False) -> Dict[str, str]:
        """
        Run complete analysis pipeline
        
        Args:
            keyword: Search keyword
            max_results: Maximum papers to collect
            skip_ai_analysis: Skip AI relevance analysis
            
        Returns:
            Dictionary with paths to all generated files
        """
        logger.info("="*60)
        logger.info(f"STARTING COMPLETE ANALYSIS FOR: '{keyword}'")
        logger.info("="*60)
        
        results = {
            'keyword': keyword,
            'papers_json': '',
            'analysis_json': '',
            'visualizations': {},
            'report_html': '',
            'summary': {}
        }
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Step 1: Collect papers
            logger.info("\n" + "="*40)
            logger.info("STEP 1: COLLECTING PAPERS FROM GOOGLE SCHOLAR")
            logger.info("="*40)
            
            papers, papers_json_path = self.search_and_collect_papers(keyword, max_results)
            results['papers_json'] = papers_json_path
            
            if not papers:
                logger.error("No papers collected. Aborting analysis.")
                return results
            
            # Step 2: AI Relevance Analysis (optional)
            analyzed_papers = None
            if not skip_ai_analysis and Config.DEEPSEEK_API_KEY:
                logger.info("\n" + "="*40)
                logger.info("STEP 2: AI RELEVANCE ANALYSIS")
                logger.info("="*40)
                
                try:
                    analyzed_papers, analysis_json_path = self.analyze_paper_relevance(papers, keyword)
                    results['analysis_json'] = analysis_json_path
                except Exception as e:
                    logger.warning(f"AI analysis failed: {str(e)}")
                    logger.warning("Continuing without AI analysis...")
            else:
                logger.info("\n" + "="*40)
                logger.info("STEP 2: SKIPPING AI ANALYSIS")
                logger.info("="*40)
                if not Config.DEEPSEEK_API_KEY:
                    logger.warning("DeepSeek API key not provided")
            
            # Step 3: Generate Visualizations
            logger.info("\n" + "="*40)
            logger.info("STEP 3: GENERATING BIBLIOMETRIC VISUALIZATIONS")
            logger.info("="*40)
            
            viz_paths = self.generate_visualizations(papers, keyword, analyzed_papers)
            results['visualizations'] = viz_paths
            results['report_html'] = viz_paths.get('report', '')
            
            # Generate summary
            results['summary'] = {
                'total_papers': len(papers),
                'analyzed_papers': len(analyzed_papers) if analyzed_papers else 0,
                'relevant_papers': len([p for p in (analyzed_papers or []) if p.get('is_relevant', False)]),
                'visualizations_generated': len([v for v in viz_paths.values() if v]),
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'output_directory': self.output_dir
            }
            
            # Final summary
            logger.info("\n" + "="*60)
            logger.info("ANALYSIS COMPLETE - SUMMARY")
            logger.info("="*60)
            logger.info(f"Keyword: {keyword}")
            logger.info(f"Papers collected: {results['summary']['total_papers']}")
            logger.info(f"Papers analyzed: {results['summary']['analyzed_papers']}")
            logger.info(f"Relevant papers: {results['summary']['relevant_papers']}")
            logger.info(f"Visualizations: {results['summary']['visualizations_generated']}")
            logger.info(f"Output directory: {results['summary']['output_directory']}")
            logger.info(f"Main report: {os.path.basename(results['report_html']) if results['report_html'] else 'Not generated'}")
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise e

def setup_argparser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Article Agent - Comprehensive Academic Research Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "transformer neural networks"
  python main.py "BERT language model" --max-results 1000
  python main.py "deep learning" --skip-ai-analysis
  python main.py "attention mechanism" --max-results 500 --output-dir my_analysis
        """
    )
    
    parser.add_argument(
        "keyword",
        help="Search keyword for Google Scholar"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=Config.MAX_RESULTS,
        help=f"Maximum number of papers to collect (default: {Config.MAX_RESULTS})"
    )
    
    parser.add_argument(
        "--skip-ai-analysis",
        action="store_true",
        help="Skip AI relevance analysis (faster but less insightful)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for results (default: output)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser

def main():
    """Main entry point"""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create article agent
        agent = ArticleAgent()
        agent.output_dir = args.output_dir
        
        # Run analysis
        results = agent.run_complete_analysis(
            keyword=args.keyword,
            max_results=args.max_results,
            skip_ai_analysis=args.skip_ai_analysis
        )
        
        # Print final results
        print("\n" + "="*60)
        print("ARTICLE AGENT ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📚 Papers collected: {results['summary']['total_papers']}")
        if results['summary']['analyzed_papers'] > 0:
            print(f"🤖 Papers analyzed: {results['summary']['analyzed_papers']}")
            print(f"✅ Relevant papers: {results['summary']['relevant_papers']}")
        print(f"📊 Visualizations: {results['summary']['visualizations_generated']}")
        print(f"📁 Output directory: {results['summary']['output_directory']}")
        
        if results['report_html']:
            print(f"📄 Main report: {os.path.basename(results['report_html'])}")
            print(f"\n🌐 Open the HTML report in your browser to explore the results!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)