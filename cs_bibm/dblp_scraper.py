"""
DBLP Scraper - Primary source for conference paper metadata
Uses DBLP's XML API for structured access to conference proceedings.
"""

import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import time
import logging
from urllib.parse import quote, urljoin
import re
from datetime import datetime

class DBLPScraper:
    """Scraper for DBLP Computer Science Bibliography"""
    
    def __init__(self, rate_limit_delay: float = 1.0):
        self.base_url = "https://dblp.org"
        self.api_url = "https://dblp.org/search/publ/api"
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CS-BIBM Conference Scraper (Academic Research)'
        })
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for DBLP scraper"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def scrape_conference(self, dblp_key: str, year: int, max_papers: int = 0) -> List[Dict]:
        """
        Scrape a specific conference year from DBLP
        
        Args:
            dblp_key: DBLP conference key (e.g., 'conf/nips')
            year: Conference year
            max_papers: Maximum papers to retrieve (0 for all)
            
        Returns:
            List of paper dictionaries
        """
        self.logger.info(f"Scraping {dblp_key} {year} from DBLP")
        
        papers = []
        
        try:
            # Try direct conference proceedings URL first
            proceedings_url = self._build_proceedings_url(dblp_key, year)
            papers_from_proceedings = self._scrape_proceedings_page(proceedings_url, year)
            
            if papers_from_proceedings:
                papers.extend(papers_from_proceedings)
                self.logger.info(f"Found {len(papers_from_proceedings)} papers from proceedings page")
            else:
                # Fallback to search API
                self.logger.info("Proceedings page empty, trying search API")
                papers_from_search = self._search_conference_papers(dblp_key, year, max_papers)
                papers.extend(papers_from_search)
            
            # Apply max_papers limit if specified
            if max_papers > 0:
                papers = papers[:max_papers]
            
            # Enhance papers with additional metadata
            enhanced_papers = []
            for paper in papers:
                enhanced_paper = self._enhance_paper_metadata(paper, dblp_key, year)
                enhanced_papers.append(enhanced_paper)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
            
            self.logger.info(f"Successfully scraped {len(enhanced_papers)} papers from {dblp_key} {year}")
            return enhanced_papers
            
        except Exception as e:
            self.logger.error(f"Error scraping {dblp_key} {year}: {e}")
            return []
    
    def _build_proceedings_url(self, dblp_key: str, year: int) -> str:
        """Build DBLP proceedings URL for a conference year"""
        # DBLP proceedings URLs follow patterns like:
        # https://dblp.org/db/conf/nips/nips2024.html
        conf_name = dblp_key.split('/')[-1]  # Extract conference name from key
        return f"{self.base_url}/db/{dblp_key}/{conf_name}{year}.html"
    
    def _scrape_proceedings_page(self, url: str, year: int) -> List[Dict]:
        """Scrape papers from DBLP proceedings page"""
        try:
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 404:
                self.logger.warning(f"Proceedings page not found: {url}")
                return []
            
            response.raise_for_status()
            
            # Parse HTML to extract paper information
            papers = self._parse_proceedings_html(response.text, year)
            return papers
            
        except Exception as e:
            self.logger.error(f"Error scraping proceedings page {url}: {e}")
            return []
    
    def _parse_proceedings_html(self, html: str, year: int) -> List[Dict]:
        """Parse DBLP proceedings HTML to extract paper data"""
        papers = []
        
        try:
            # Use regex to find paper entries in DBLP HTML
            # DBLP uses specific patterns for paper listings
            
            # Pattern for paper titles and authors
            paper_pattern = r'<li class="entry[^"]*"[^>]*>.*?</li>'
            title_pattern = r'<span class="title"[^>]*>(.*?)</span>'
            author_pattern = r'<span[^>]*itemprop="author"[^>]*><span[^>]*>(.*?)</span></span>'
            doi_pattern = r'href="https://doi\.org/([^"]+)"'
            dblp_key_pattern = r'<a href="/rec/([^"]+)\.html"'
            
            import re
            
            paper_matches = re.findall(paper_pattern, html, re.DOTALL | re.IGNORECASE)
            
            for paper_html in paper_matches:
                paper_data = {'year': year, 'source': 'dblp'}
                
                # Extract title
                title_match = re.search(title_pattern, paper_html, re.IGNORECASE)
                if title_match:
                    title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
                    paper_data['title'] = self._clean_text(title)
                
                # Extract authors
                author_matches = re.findall(author_pattern, paper_html, re.IGNORECASE)
                if author_matches:
                    authors = [self._clean_text(author) for author in author_matches]
                    paper_data['authors'] = authors
                
                # Extract DOI
                doi_match = re.search(doi_pattern, paper_html, re.IGNORECASE)
                if doi_match:
                    paper_data['doi'] = doi_match.group(1)
                
                # Extract DBLP key
                dblp_key_match = re.search(dblp_key_pattern, paper_html, re.IGNORECASE)
                if dblp_key_match:
                    paper_data['dblp_key'] = dblp_key_match.group(1)
                    paper_data['dblp_url'] = f"{self.base_url}/rec/{dblp_key_match.group(1)}.html"
                
                # Only include papers with title and authors
                if 'title' in paper_data and 'authors' in paper_data:
                    papers.append(paper_data)
            
        except Exception as e:
            self.logger.error(f"Error parsing proceedings HTML: {e}")
        
        return papers
    
    def _search_conference_papers(self, dblp_key: str, year: int, max_papers: int = 0) -> List[Dict]:
        """Search for conference papers using DBLP API"""
        papers = []
        
        try:
            # Build search query - use proper venue names for DBLP
            conf_name = dblp_key.split('/')[-1]
            
            # Map DBLP keys to proper venue search terms
            venue_search_map = {
                'nips': 'neurips',
                'aaai': 'aaai', 
                'icml': 'icml',
                'iclr': 'iclr',
                'acl': 'acl',
                'emnlp': 'emnlp',
                'cvpr': 'cvpr',
                'iccv': 'iccv',
                'eccv': 'eccv',
                'ijcai': 'ijcai',
                'uai': 'uai',
                'colt': 'colt',
                'corl': 'corl'
            }
            
            search_term = venue_search_map.get(conf_name, conf_name)
            query = f"venue:{search_term} year:{year}"
            
            # DBLP API parameters
            params = {
                'q': query,
                'format': 'xml',
                'h': min(max_papers, 1000) if max_papers > 0 else 1000,  # DBLP API limit
                'f': 0  # Start from first result
            }
            
            response = self.session.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            papers = self._parse_dblp_xml(response.text, year)
            
            # Handle pagination if needed
            if max_papers == 0 or len(papers) < max_papers:
                # Check if there are more results and fetch them
                # DBLP API supports pagination through 'f' parameter
                pass  # For now, limit to first 1000 results
            
        except Exception as e:
            self.logger.error(f"Error searching DBLP API: {e}")
        
        return papers
    
    def _parse_dblp_xml(self, xml_content: str, year: int) -> List[Dict]:
        """Parse DBLP XML API response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for hit in root.findall('.//hit'):
                info = hit.find('info')
                if info is None:
                    continue
                
                paper_data = {
                    'year': year,
                    'source': 'dblp'
                }
                
                # Extract title
                title_elem = info.find('title')
                if title_elem is not None:
                    paper_data['title'] = self._clean_text(title_elem.text or '')
                
                # Extract authors
                authors = []
                for author in info.findall('authors/author'):
                    if author.text:
                        authors.append(self._clean_text(author.text))
                paper_data['authors'] = authors
                
                # Extract venue
                venue_elem = info.find('venue')
                if venue_elem is not None:
                    paper_data['venue_dblp'] = self._clean_text(venue_elem.text or '')
                
                # Extract year (verify)
                year_elem = info.find('year')
                if year_elem is not None:
                    paper_data['year'] = int(year_elem.text) if year_elem.text else year
                
                # Extract DOI
                doi_elem = info.find('doi')
                if doi_elem is not None:
                    paper_data['doi'] = self._clean_text(doi_elem.text or '')
                
                # Extract DBLP key/URL
                key_elem = info.find('key')
                if key_elem is not None:
                    dblp_key = self._clean_text(key_elem.text or '')
                    paper_data['dblp_key'] = dblp_key
                    paper_data['dblp_url'] = f"{self.base_url}/rec/{dblp_key}.html"
                
                # Extract type (ensure it's a conference paper)
                type_elem = info.find('type')
                paper_type = type_elem.text if type_elem is not None else ''
                
                # Only include conference papers
                if paper_type in ['Conference and Workshop Papers', 'inproceedings']:
                    papers.append(paper_data)
            
        except ET.ParseError as e:
            self.logger.error(f"Error parsing DBLP XML: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing DBLP XML: {e}")
        
        return papers
    
    def _enhance_paper_metadata(self, paper: Dict, dblp_key: str, year: int) -> Dict:
        """Enhance paper with additional metadata from DBLP"""
        enhanced_paper = paper.copy()
        
        # Add conference information
        enhanced_paper['venue'] = self._extract_venue_acronym(dblp_key)
        enhanced_paper['conference_year'] = year
        
        # Build URLs if missing
        if 'dblp_url' not in enhanced_paper and 'dblp_key' in enhanced_paper:
            enhanced_paper['dblp_url'] = f"{self.base_url}/rec/{enhanced_paper['dblp_key']}.html"
        
        # Build official URLs if DOI exists
        if 'doi' in enhanced_paper:
            enhanced_paper['official_url'] = f"https://doi.org/{enhanced_paper['doi']}"
        
        # Add scraping metadata
        enhanced_paper['scraped_at'] = datetime.now().isoformat()
        enhanced_paper['scraping_source'] = 'dblp'
        
        return enhanced_paper
    
    def _extract_venue_acronym(self, dblp_key: str) -> str:
        """Extract venue acronym from DBLP key"""
        # DBLP keys like 'conf/nips' -> 'NIPS'
        venue_map = {
            'nips': 'NeurIPS',
            'icml': 'ICML',
            'iclr': 'ICLR',
            'aaai': 'AAAI',
            'ijcai': 'IJCAI',
            'acl': 'ACL',
            'emnlp': 'EMNLP',
            'naacl': 'NAACL',
            'cvpr': 'CVPR',
            'iccv': 'ICCV',
            'eccv': 'ECCV',
            'uai': 'UAI',
            'colt': 'COLT',
            'corl': 'CoRL',
            'interspeech': 'INTERSPEECH',
            'iwslt': 'IWSLT'
        }
        
        venue = dblp_key.split('/')[-1].lower()
        return venue_map.get(venue, venue.upper())
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Decode HTML entities
        import html
        text = html.unescape(text)
        
        return text.strip()
    
    def get_conference_years(self, dblp_key: str) -> List[int]:
        """Get available years for a conference in DBLP"""
        try:
            # Search for all papers from this conference
            conf_name = dblp_key.split('/')[-1]
            params = {
                'q': f"venue:{conf_name}",
                'format': 'xml',
                'h': 100,
                'f': 0
            }
            
            response = self.session.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse years from results
            root = ET.fromstring(response.text)
            years = set()
            
            for hit in root.findall('.//hit'):
                info = hit.find('info')
                if info is not None:
                    year_elem = info.find('year')
                    if year_elem is not None and year_elem.text:
                        years.add(int(year_elem.text))
            
            return sorted(list(years))
            
        except Exception as e:
            self.logger.error(f"Error getting conference years for {dblp_key}: {e}")
            return []

def main():
    """Demo usage of DBLP scraper"""
    scraper = DBLPScraper(rate_limit_delay=0.5)
    
    # Test with NeurIPS 2023
    print("=== DBLP Scraper Demo ===")
    print("Scraping NeurIPS 2023...")
    
    papers = scraper.scrape_conference('conf/nips', 2023, max_papers=10)
    
    print(f"\nFound {len(papers)} papers:")
    for i, paper in enumerate(papers[:5], 1):
        print(f"\n{i}. {paper.get('title', 'No title')}")
        print(f"   Authors: {', '.join(paper.get('authors', []))}")
        print(f"   DOI: {paper.get('doi', 'N/A')}")
        print(f"   DBLP: {paper.get('dblp_url', 'N/A')}")
    
    # Test getting available years
    print("\n=== Available Years ===")
    years = scraper.get_conference_years('conf/nips')
    print(f"NeurIPS years in DBLP: {years}")

if __name__ == "__main__":
    main()