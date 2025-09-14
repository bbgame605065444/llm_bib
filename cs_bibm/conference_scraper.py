"""
Conference Scraper - Main orchestrator for multi-source paper scraping
Coordinates DBLP, arXiv, and venue-specific scrapers.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import time
from pathlib import Path

from conference_registry import ConferenceRegistry, ConferenceInfo
from dblp_scraper import DBLPScraper
from arxiv_matcher import ArxivMatcher
from database_manager import DatabaseManager

class ConferenceScraper:
    """Main conference scraper orchestrating multiple data sources"""
    
    def __init__(self, 
                 db_path: str = "cs_bibm/data/conference_papers.db",
                 enable_arxiv: bool = True,
                 rate_limit_delay: float = 1.0,
                 batch_size: int = 50):
        
        self.registry = ConferenceRegistry()
        self.db_manager = DatabaseManager(db_path)
        self.dblp_scraper = DBLPScraper(rate_limit_delay=rate_limit_delay)
        self.arxiv_matcher = ArxivMatcher(rate_limit_delay=rate_limit_delay) if enable_arxiv else None
        
        self.enable_arxiv = enable_arxiv
        self.batch_size = batch_size
        self.logger = self._setup_logging()
        
        # Statistics tracking
        self.stats = {
            'conferences_scraped': 0,
            'papers_found': 0,
            'papers_inserted': 0,
            'papers_updated': 0,
            'arxiv_matches': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_format)
            logger.addHandler(console_handler)
            
            # File handler
            log_dir = Path("cs_bibm/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / 'conference_scraper.log')
            file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        return logger
    
    def scrape_all_conferences(self, 
                             years: List[int] = None, 
                             max_papers_per_venue: int = 0,
                             conference_types: List[str] = None) -> Dict[str, any]:
        """
        Scrape all registered conferences
        
        Args:
            years: List of years to scrape (default: 2020-2024)
            max_papers_per_venue: Maximum papers per venue (0 for unlimited)
            conference_types: Filter by conference types
            
        Returns:
            Dictionary with scraping results and statistics
        """
        if years is None:
            years = list(range(2020, 2025))
        
        self.stats['start_time'] = datetime.now()
        self.logger.info(f"Starting comprehensive conference scraping for years {years}")
        
        # Get conferences to scrape
        conferences_to_scrape = []
        
        if conference_types:
            from .conference_registry import ConferenceType
            type_map = {t.value: t for t in ConferenceType}
            
            for conf in self.registry.conferences:
                if conf.conference_type.value in conference_types:
                    conferences_to_scrape.append(conf)
        else:
            conferences_to_scrape = self.registry.conferences
        
        # Scrape in priority order (largest conferences first)
        priority_order = self.registry.get_scraping_priority()
        ordered_conferences = []
        
        for acronym in priority_order:
            conf = self.registry.get_conference(acronym)
            if conf and conf in conferences_to_scrape:
                ordered_conferences.append(conf)
        
        self.logger.info(f"Will scrape {len(ordered_conferences)} conferences")
        
        # Scrape each conference
        results = {}
        
        for i, conf in enumerate(ordered_conferences, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Scraping conference {i}/{len(ordered_conferences)}: {conf.acronym}")
            self.logger.info(f"{'='*60}")
            
            try:
                conf_results = self.scrape_conference(
                    conf.acronym, 
                    years, 
                    max_papers_per_venue
                )
                results[conf.acronym] = conf_results
                self.stats['conferences_scraped'] += 1
                
                # Brief pause between conferences
                time.sleep(2.0)
                
            except Exception as e:
                self.logger.error(f"Failed to scrape {conf.acronym}: {e}")
                results[conf.acronym] = {'error': str(e)}
                self.stats['errors'] += 1
        
        self.stats['end_time'] = datetime.now()
        
        # Generate final report
        final_results = {
            'statistics': self.stats,
            'conference_results': results,
            'summary': self._generate_summary()
        }
        
        self._save_scraping_report(final_results)
        self.logger.info(f"\\n{self._generate_summary()}")
        
        return final_results
    
    def scrape_conference(self, 
                         acronym: str, 
                         years: List[int] = None, 
                         max_papers: int = 0) -> Dict[str, any]:
        """
        Scrape a specific conference
        
        Args:
            acronym: Conference acronym (e.g., 'NeurIPS')
            years: Years to scrape (default: all available)
            max_papers: Maximum papers to scrape (0 for unlimited)
            
        Returns:
            Dictionary with scraping results
        """
        conf = self.registry.get_conference(acronym)
        if not conf:
            raise ValueError(f"Conference '{acronym}' not found in registry")
        
        if years is None:
            years = conf.years_available
        else:
            # Filter to only available years
            years = [y for y in years if y in conf.years_available]
        
        self.logger.info(f"Scraping {conf.full_name} ({acronym}) for years: {years}")
        
        results = {
            'conference': acronym,
            'years_scraped': [],
            'total_papers': 0,
            'papers_by_year': {},
            'arxiv_matches': 0,
            'errors': []
        }
        
        for year in years:
            try:
                self.logger.info(f"\\nScraping {acronym} {year}...")
                
                # Check if already scraped recently
                if self._is_recently_scraped(acronym, year):
                    self.logger.info(f"  Skipping {acronym} {year} - recently scraped")
                    continue
                
                year_results = self._scrape_conference_year(conf, year, max_papers)
                
                results['years_scraped'].append(year)
                results['papers_by_year'][year] = len(year_results['papers'])
                results['total_papers'] += len(year_results['papers'])
                results['arxiv_matches'] += year_results.get('arxiv_matches', 0)
                
                if year_results.get('errors'):
                    results['errors'].extend(year_results['errors'])
                
                self.logger.info(f"  ✓ {acronym} {year}: {len(year_results['papers'])} papers")
                
            except Exception as e:
                error_msg = f"Error scraping {acronym} {year}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                self.stats['errors'] += 1
        
        self.logger.info(f"\\n{acronym} scraping completed: {results['total_papers']} total papers")
        return results
    
    def _scrape_conference_year(self, conf: ConferenceInfo, year: int, max_papers: int) -> Dict[str, any]:
        """Scrape a specific conference year"""
        results = {
            'papers': [],
            'arxiv_matches': 0,
            'errors': []
        }
        
        try:
            # Step 1: Get papers from DBLP
            self.logger.info(f"  Fetching from DBLP...")
            papers = self.dblp_scraper.scrape_conference(conf.dblp_key, year, max_papers)
            
            if not papers:
                self.logger.warning(f"  No papers found in DBLP for {conf.acronym} {year}")
                return results
            
            self.logger.info(f"  Found {len(papers)} papers from DBLP")
            
            # Step 2: Enhance with arXiv data (if enabled)
            if self.enable_arxiv and self.arxiv_matcher:
                self.logger.info(f"  Matching with arXiv...")
                papers = self.arxiv_matcher.match_paper_batch(papers)
                
                # Count arXiv matches
                arxiv_matches = sum(1 for paper in papers if 'arxiv_id' in paper)
                results['arxiv_matches'] = arxiv_matches
                
                if arxiv_matches > 0:
                    match_rate = (arxiv_matches / len(papers)) * 100
                    self.logger.info(f"  arXiv matches: {arxiv_matches}/{len(papers)} ({match_rate:.1f}%)")
            
            # Step 3: Enhance papers with conference metadata
            for paper in papers:
                paper.update({
                    'venue': conf.acronym,
                    'venue_full_name': conf.full_name,
                    'conference_type': conf.conference_type.value,
                    'year': year,
                    'scraped_at': datetime.now().isoformat()
                })
            
            # Step 4: Save to database
            self.logger.info(f"  Saving to database...")
            inserted, updated = self._save_papers_batch(papers)
            
            self.stats['papers_found'] += len(papers)
            self.stats['papers_inserted'] += inserted
            self.stats['papers_updated'] += updated
            self.stats['arxiv_matches'] += results['arxiv_matches']
            
            results['papers'] = papers
            
            # Step 5: Log scraping record
            self._log_scraping_record(conf.acronym, year, 'dblp', len(papers), inserted, updated)
            
        except Exception as e:
            error_msg = f"Error in _scrape_conference_year: {e}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return results
    
    def _save_papers_batch(self, papers: List[Dict]) -> Tuple[int, int]:
        """Save papers to database in batches"""
        inserted = 0
        updated = 0
        
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i:i + self.batch_size]
            
            for paper in batch:
                try:
                    # Check if paper already exists
                    exists = self.db_manager.paper_exists(
                        paper['title'],
                        paper.get('authors', []),
                        paper.get('year', 0)
                    )
                    
                    success = self.db_manager.insert_paper(paper)
                    if success:
                        if exists:
                            updated += 1
                        else:
                            inserted += 1
                
                except Exception as e:
                    self.logger.error(f"Error saving paper: {e}")
                    continue
        
        return inserted, updated
    
    def _is_recently_scraped(self, venue: str, year: int, hours_threshold: int = 24) -> bool:
        """Check if venue/year was recently scraped"""
        # This could be enhanced with actual database checking
        # For now, always scrape (implement caching logic as needed)
        return False
    
    def _log_scraping_record(self, venue: str, year: int, source: str, 
                           papers_found: int, papers_inserted: int, papers_updated: int):
        """Log scraping record to database"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO scraping_log 
                    (venue, year, source, start_time, end_time, papers_found, 
                     papers_inserted, papers_updated, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    venue, year, source, datetime.now(), datetime.now(),
                    papers_found, papers_inserted, papers_updated, True
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error logging scraping record: {e}")
    
    def _generate_summary(self) -> str:
        """Generate human-readable summary of scraping results"""
        if not self.stats['start_time']:
            return "No scraping performed"
        
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        
        summary = f"""
CONFERENCE SCRAPING SUMMARY
{'='*50}
Duration: {hours}h {minutes}m
Conferences scraped: {self.stats['conferences_scraped']}
Total papers found: {self.stats['papers_found']:,}
Papers inserted: {self.stats['papers_inserted']:,}
Papers updated: {self.stats['papers_updated']:,}
arXiv matches: {self.stats['arxiv_matches']:,}
Errors: {self.stats['errors']}

Performance:
Papers per minute: {(self.stats['papers_found'] / (duration / 60)):.1f}
Success rate: {((self.stats['papers_found'] - self.stats['errors']) / max(1, self.stats['papers_found']) * 100):.1f}%
"""
        
        if self.enable_arxiv and self.stats['papers_found'] > 0:
            arxiv_rate = (self.stats['arxiv_matches'] / self.stats['papers_found']) * 100
            summary += f"arXiv match rate: {arxiv_rate:.1f}%\\n"
        
        return summary
    
    def _save_scraping_report(self, results: Dict):
        """Save comprehensive scraping report"""
        try:
            report_dir = Path("cs_bibm/reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"scraping_report_{timestamp}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"Scraping report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving scraping report: {e}")
    
    def get_scraping_status(self) -> Dict[str, any]:
        """Get current scraping statistics"""
        db_stats = self.db_manager.get_database_stats()
        
        status = {
            'current_session': self.stats,
            'database': db_stats,
            'system_info': {
                'arxiv_enabled': self.enable_arxiv,
                'batch_size': self.batch_size,
                'registered_conferences': len(self.registry.conferences)
            }
        }
        
        return status
    
    def export_results(self, output_path: str, venue: str = None) -> bool:
        """Export scraping results to CSV"""
        return self.db_manager.export_to_csv(output_path, venue)

def main():
    """Demo usage of conference scraper"""
    print("=== Conference Scraper Demo ===")
    
    # Initialize scraper
    scraper = ConferenceScraper(
        db_path="demo_conference_papers.db",
        enable_arxiv=True,
        rate_limit_delay=0.5,
        batch_size=25
    )
    
    # Test scraping a single conference
    print("\\nScraping NeurIPS 2023 (limited to 20 papers)...")
    results = scraper.scrape_conference('NeurIPS', [2023], max_papers=20)
    
    print(f"\\nResults:")
    print(f"Years scraped: {results['years_scraped']}")
    print(f"Total papers: {results['total_papers']}")
    print(f"arXiv matches: {results['arxiv_matches']}")
    
    # Show database stats
    print("\\n=== Database Statistics ===")
    status = scraper.get_scraping_status()
    db_stats = status['database']
    
    for key, value in db_stats.items():
        if isinstance(value, dict):
            print(f"{key}: {len(value)} entries")
        elif isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()