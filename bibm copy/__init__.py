"""
Article Agent - Comprehensive Academic Research Analysis Tool

This package provides a complete solution for academic research analysis including:
- Google Scholar scraping via SerpApi
- AI-powered relevance analysis using DeepSeek
- Comprehensive bibliometric visualizations
- Interactive dashboards and reports

Main components:
- ScholarScraper: Google Scholar data collection
- DeepSeekAnalyzer: AI relevance analysis  
- BibliometricsAnalyzer: Visualization and analysis
- ArticleAgent: Main orchestration class

Usage:
    python main.py "your research keyword"
"""

__version__ = "1.0.0"
__author__ = "Article Agent Team"
__description__ = "Comprehensive Academic Research Analysis Tool"

from .scholar_scraper import ScholarScraper
from .ai_analyzer import DeepSeekAnalyzer  
from .bibliometrics import BibliometricsAnalyzer
from .config import Config

__all__ = [
    'ScholarScraper',
    'DeepSeekAnalyzer', 
    'BibliometricsAnalyzer',
    'Config'
]