"""
CS-BIBM: Conference Scraper for Bibliometric Analysis
A comprehensive system for scraping and analyzing academic conference papers.

Key Components:
- ConferenceRegistry: Maps 22+ top conferences to scraping configurations
- ConferenceScraper: Multi-source scraper (DBLP, arXiv, venue sites)  
- DatabaseManager: SQLite storage with comprehensive schema
- ArxivMatcher: Links papers with full-text arXiv preprints
- ConferenceCLI: Command-line interface for all operations

Usage:
    from cs_bibm import ConferenceScraper, ConferenceRegistry
    
    # Quick start
    scraper = ConferenceScraper()
    results = scraper.scrape_conference('NeurIPS', [2023])
    
    # Command line
    python -m cs_bibm.conference_cli scrape --conferences neurips --years 2023
"""

__version__ = "1.0.0"
__author__ = "CS-BIBM Project"
__description__ = "Conference Scraper for Bibliometric Analysis"

# Import main classes for easy access
try:
    from .conference_registry import ConferenceRegistry, ConferenceInfo, ConferenceType
    from .conference_scraper import ConferenceScraper
    from .database_manager import DatabaseManager
    from .dblp_scraper import DBLPScraper
    from .arxiv_matcher import ArxivMatcher
    from .config import Config
    
    __all__ = [
        'ConferenceRegistry',
        'ConferenceInfo', 
        'ConferenceType',
        'ConferenceScraper',
        'DatabaseManager',
        'DBLPScraper',
        'ArxivMatcher',
        'Config'
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some cs_bibm components not available: {e}")
    __all__ = []

def get_version():
    """Get package version"""
    return __version__

def print_system_info():
    """Print system information"""
    print(f"CS-BIBM v{__version__}")
    print(f"Description: {__description__}")
    print(f"Available components: {len(__all__)}")
    
    if 'ConferenceRegistry' in globals():
        registry = ConferenceRegistry()
        print(f"Registered conferences: {len(registry.conferences)}")
        
        # Show conference types
        from collections import Counter
        types = Counter(conf.conference_type.value for conf in registry.conferences)
        print("Conference types:")
        for conf_type, count in types.items():
            print(f"  - {conf_type}: {count} conferences")

# Quick access functions
def quick_scrape(conference: str, years: list = None, max_papers: int = 100):
    """
    Quick scraping function for interactive use
    
    Args:
        conference: Conference acronym (e.g., 'NeurIPS')
        years: List of years (default: [2023])
        max_papers: Maximum papers to scrape
    
    Returns:
        Dictionary with scraping results
    """
    if 'ConferenceScraper' not in globals():
        raise ImportError("ConferenceScraper not available. Check dependencies.")
    
    if years is None:
        years = [2023]
    
    scraper = ConferenceScraper()
    return scraper.scrape_conference(conference, years, max_papers)

def list_conferences():
    """List all available conferences"""
    if 'ConferenceRegistry' not in globals():
        raise ImportError("ConferenceRegistry not available. Check dependencies.")
    
    registry = ConferenceRegistry()
    
    print("Available Conferences:")
    print("=" * 50)
    
    # Group by type
    by_type = {}
    for conf in registry.conferences:
        conf_type = conf.conference_type.value
        if conf_type not in by_type:
            by_type[conf_type] = []
        by_type[conf_type].append(conf)
    
    for conf_type, confs in sorted(by_type.items()):
        print(f"\\n{conf_type}:")
        for conf in sorted(confs, key=lambda x: x.acronym):
            years_str = f"{min(conf.years_available)}-{max(conf.years_available)}"
            print(f"  {conf.acronym:10} - {conf.full_name[:40]:40} ({years_str})")
    
    return [conf.acronym for conf in registry.conferences]

def get_estimates(conferences: list = None, years: list = None):
    """Get paper count estimates"""
    if 'ConferenceRegistry' not in globals():
        raise ImportError("ConferenceRegistry not available. Check dependencies.")
    
    registry = ConferenceRegistry()
    
    if years is None:
        years = list(range(2020, 2025))
    
    estimates = registry.estimate_total_papers(conferences, years)
    
    print(f"Paper Count Estimates ({min(years)}-{max(years)}):")
    print("=" * 50)
    
    for conf, count in sorted(estimates.items()):
        if conf != 'TOTAL':
            print(f"{conf:12}: {count:,} papers")
    
    print("-" * 30)
    print(f"{'TOTAL':12}: {estimates['TOTAL']:,} papers")
    
    return estimates

# Command-line interface entry point
def main():
    """Entry point for command-line interface"""
    try:
        from .conference_cli import ConferenceCLI
        cli = ConferenceCLI()
        cli.main()
    except ImportError as e:
        print(f"CLI not available: {e}")
        print("Install required dependencies and try again")

if __name__ == "__main__":
    main()