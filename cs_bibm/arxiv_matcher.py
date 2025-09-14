"""
arXiv Matcher - Links conference papers with arXiv preprints
Provides full-text access and additional metadata for papers.
"""

import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import time
import logging
import re
from datetime import datetime
from urllib.parse import quote
import difflib

class ArxivMatcher:
    """Matches conference papers with arXiv preprints"""
    
    def __init__(self, rate_limit_delay: float = 1.0, similarity_threshold: float = 0.85):
        self.api_url = "http://export.arxiv.org/api/query"
        self.rate_limit_delay = rate_limit_delay
        self.similarity_threshold = similarity_threshold
        self.session = requests.Session()
        self.logger = self._setup_logging()
        
        # Cache for API results to avoid duplicate queries
        self._search_cache = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for arXiv matcher"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def match_paper_batch(self, papers: List[Dict]) -> List[Dict]:
        """
        Match a batch of papers with arXiv preprints
        
        Args:
            papers: List of paper dictionaries with 'title' and 'authors'
            
        Returns:
            List of enhanced papers with arXiv metadata
        """
        self.logger.info(f"Matching {len(papers)} papers with arXiv")
        
        enhanced_papers = []
        matched_count = 0
        
        for i, paper in enumerate(papers):
            self.logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.get('title', 'No title')[:60]}...")
            
            # Try to find arXiv match
            arxiv_data = self.find_arxiv_match(paper)
            
            # Enhance paper with arXiv data
            enhanced_paper = paper.copy()
            if arxiv_data:
                enhanced_paper.update(arxiv_data)
                matched_count += 1
                self.logger.info(f"  ✓ Matched with arXiv:{arxiv_data['arxiv_id']}")
            else:
                self.logger.debug(f"  ✗ No arXiv match found")
            
            enhanced_papers.append(enhanced_paper)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        match_rate = (matched_count / len(papers)) * 100 if papers else 0
        self.logger.info(f"arXiv matching completed: {matched_count}/{len(papers)} papers matched ({match_rate:.1f}%)")
        
        return enhanced_papers
    
    def find_arxiv_match(self, paper: Dict) -> Optional[Dict]:
        """
        Find arXiv match for a single paper
        
        Args:
            paper: Paper dictionary with 'title' and optionally 'authors'
            
        Returns:
            Dictionary with arXiv metadata if match found, None otherwise
        """
        title = paper.get('title', '').strip()
        authors = paper.get('authors', [])
        
        if not title:
            return None
        
        # Try multiple search strategies
        strategies = [
            self._search_by_title_exact,
            self._search_by_title_fuzzy,
            self._search_by_title_and_author
        ]
        
        for strategy in strategies:
            try:
                arxiv_data = strategy(title, authors)
                if arxiv_data:
                    return arxiv_data
            except Exception as e:
                self.logger.debug(f"Search strategy failed: {e}")
                continue
        
        return None
    
    def _search_by_title_exact(self, title: str, authors: List[str]) -> Optional[Dict]:
        """Search arXiv by exact title match"""
        # Clean title for search
        clean_title = self._clean_title_for_search(title)
        search_query = f'ti:"{clean_title}"'
        
        results = self._query_arxiv_api(search_query, max_results=5)
        
        for result in results:
            if self._is_title_match(title, result['title']):
                # Verify author match if available
                if authors and not self._has_author_overlap(authors, result.get('authors', [])):
                    continue
                
                return self._extract_arxiv_metadata(result)
        
        return None
    
    def _search_by_title_fuzzy(self, title: str, authors: List[str]) -> Optional[Dict]:
        """Search arXiv by fuzzy title matching"""
        # Extract key terms from title
        key_terms = self._extract_key_terms(title)
        if len(key_terms) < 2:
            return None
        
        search_query = ' AND '.join([f'ti:{term}' for term in key_terms[:4]])  # Limit to 4 terms
        
        results = self._query_arxiv_api(search_query, max_results=10)
        
        best_match = None
        best_similarity = 0
        
        for result in results:
            similarity = self._calculate_title_similarity(title, result['title'])
            
            if similarity > self.similarity_threshold and similarity > best_similarity:
                # Additional validation with authors if available
                if authors:
                    if not self._has_author_overlap(authors, result.get('authors', [])):
                        continue
                
                best_match = result
                best_similarity = similarity
        
        if best_match:
            arxiv_data = self._extract_arxiv_metadata(best_match)
            arxiv_data['title_similarity'] = best_similarity
            return arxiv_data
        
        return None
    
    def _search_by_title_and_author(self, title: str, authors: List[str]) -> Optional[Dict]:
        """Search arXiv by combining title terms and author names"""
        if not authors:
            return None
        
        # Use first author's last name
        first_author = authors[0].split()
        if len(first_author) < 2:
            return None
        
        author_lastname = first_author[-1]
        key_terms = self._extract_key_terms(title)[:3]
        
        if not key_terms:
            return None
        
        # Combine title terms and author
        title_part = ' AND '.join([f'ti:{term}' for term in key_terms])
        author_part = f'au:{author_lastname}'
        search_query = f'{title_part} AND {author_part}'
        
        results = self._query_arxiv_api(search_query, max_results=10)
        
        for result in results:
            # Check title similarity
            similarity = self._calculate_title_similarity(title, result['title'])
            if similarity > 0.7:  # Lower threshold when author matches
                # Verify author match
                if self._has_author_overlap(authors, result.get('authors', [])):
                    arxiv_data = self._extract_arxiv_metadata(result)
                    arxiv_data['title_similarity'] = similarity
                    return arxiv_data
        
        return None
    
    def _query_arxiv_api(self, query: str, max_results: int = 10) -> List[Dict]:
        """Query arXiv API and parse results"""
        # Check cache first
        cache_key = f"{query}:{max_results}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        
        try:
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = self.session.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            
            results = self._parse_arxiv_xml(response.text)
            
            # Cache results
            self._search_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying arXiv API: {e}")
            return []
    
    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict]:
        """Parse arXiv API XML response"""
        results = []
        
        try:
            # Register namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            root = ET.fromstring(xml_content)
            
            for entry in root.findall('atom:entry', namespaces):
                result = {}
                
                # Extract arXiv ID
                id_elem = entry.find('atom:id', namespaces)
                if id_elem is not None:
                    arxiv_url = id_elem.text
                    arxiv_id = arxiv_url.split('/')[-1]
                    result['arxiv_id'] = arxiv_id
                    result['arxiv_url'] = arxiv_url
                
                # Extract title
                title_elem = entry.find('atom:title', namespaces)
                if title_elem is not None:
                    result['title'] = self._clean_text(title_elem.text)
                
                # Extract authors
                authors = []
                for author in entry.findall('atom:author', namespaces):
                    name_elem = author.find('atom:name', namespaces)
                    if name_elem is not None:
                        authors.append(self._clean_text(name_elem.text))
                result['authors'] = authors
                
                # Extract abstract
                summary_elem = entry.find('atom:summary', namespaces)
                if summary_elem is not None:
                    result['abstract'] = self._clean_text(summary_elem.text)
                
                # Extract publication date
                published_elem = entry.find('atom:published', namespaces)
                if published_elem is not None:
                    result['published_date'] = published_elem.text
                
                # Extract categories
                categories = []
                for category in entry.findall('atom:category', namespaces):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                result['categories'] = categories
                
                # Extract DOI if present
                doi_elem = entry.find('arxiv:doi', namespaces)
                if doi_elem is not None:
                    result['doi'] = doi_elem.text
                
                # Extract PDF link
                for link in entry.findall('atom:link', namespaces):
                    if link.get('type') == 'application/pdf':
                        result['pdf_url'] = link.get('href')
                        break
                
                results.append(result)
                
        except ET.ParseError as e:
            self.logger.error(f"Error parsing arXiv XML: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing arXiv XML: {e}")
        
        return results
    
    def _extract_arxiv_metadata(self, arxiv_result: Dict) -> Dict:
        """Extract relevant metadata from arXiv result"""
        metadata = {
            'arxiv_id': arxiv_result.get('arxiv_id'),
            'arxiv_url': arxiv_result.get('arxiv_url'),
            'pdf_url': arxiv_result.get('pdf_url'),
            'abstract': arxiv_result.get('abstract'),
            'categories': arxiv_result.get('categories', []),
            'published_date': arxiv_result.get('published_date'),
            'arxiv_matched_at': datetime.now().isoformat()
        }
        
        # Use arXiv DOI if original paper doesn't have one
        if 'doi' in arxiv_result:
            metadata['arxiv_doi'] = arxiv_result['doi']
        
        return {k: v for k, v in metadata.items() if v is not None}
    
    def _clean_title_for_search(self, title: str) -> str:
        """Clean title for arXiv search"""
        # Remove common patterns that cause search issues
        title = re.sub(r'\s*\([^)]*\)\s*', ' ', title)  # Remove parentheses
        title = re.sub(r'[^\w\s]', ' ', title)  # Remove special characters
        title = ' '.join(title.split())  # Normalize whitespace
        return title.strip()
    
    def _extract_key_terms(self, title: str) -> List[str]:
        """Extract key terms from title for search"""
        # Common stop words to remove
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with'
        }
        
        # Clean and split title
        clean_title = re.sub(r'[^\w\s]', ' ', title.lower())
        words = clean_title.split()
        
        # Filter out stop words and short words
        key_terms = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return key_terms
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        # Normalize titles
        title1_norm = re.sub(r'[^\w\s]', '', title1.lower()).strip()
        title2_norm = re.sub(r'[^\w\s]', '', title2.lower()).strip()
        
        # Use difflib for sequence matching
        return difflib.SequenceMatcher(None, title1_norm, title2_norm).ratio()
    
    def _is_title_match(self, title1: str, title2: str) -> bool:
        """Check if two titles are essentially the same"""
        return self._calculate_title_similarity(title1, title2) > self.similarity_threshold
    
    def _has_author_overlap(self, authors1: List[str], authors2: List[str]) -> bool:
        """Check if there's significant author overlap between two author lists"""
        if not authors1 or not authors2:
            return True  # Can't verify, so don't reject
        
        # Normalize author names (last names only)
        def get_lastname(author):
            return author.split()[-1].lower() if author else ''
        
        lastnames1 = {get_lastname(author) for author in authors1}
        lastnames2 = {get_lastname(author) for author in authors2}
        
        # Check for any overlap
        overlap = lastnames1.intersection(lastnames2)
        return len(overlap) > 0
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def get_arxiv_categories(self) -> Dict[str, str]:
        """Get mapping of arXiv categories to descriptions"""
        return {
            'cs.AI': 'Artificial Intelligence',
            'cs.CL': 'Computation and Language (NLP)',
            'cs.CV': 'Computer Vision and Pattern Recognition',
            'cs.LG': 'Machine Learning',
            'cs.NE': 'Neural and Evolutionary Computing',
            'stat.ML': 'Machine Learning (Statistics)',
            'cs.IR': 'Information Retrieval',
            'cs.RO': 'Robotics',
            'cs.CR': 'Cryptography and Security',
            'cs.DC': 'Distributed, Parallel, and Cluster Computing',
            'cs.OS': 'Operating Systems'
        }

def main():
    """Demo usage of arXiv matcher"""
    matcher = ArxivMatcher(rate_limit_delay=0.5)
    
    # Test papers
    test_papers = [
        {
            'title': 'Attention Is All You Need',
            'authors': ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
            'year': 2017,
            'venue': 'NeurIPS'
        },
        {
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
            'authors': ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee'],
            'year': 2019,
            'venue': 'NAACL'
        }
    ]
    
    print("=== arXiv Matcher Demo ===")
    enhanced_papers = matcher.match_paper_batch(test_papers)
    
    for paper in enhanced_papers:
        print(f"\nTitle: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        if 'arxiv_id' in paper:
            print(f"✓ arXiv Match: {paper['arxiv_id']}")
            print(f"  PDF: {paper.get('pdf_url', 'N/A')}")
            print(f"  Categories: {', '.join(paper.get('categories', []))}")
            if 'title_similarity' in paper:
                print(f"  Similarity: {paper['title_similarity']:.3f}")
        else:
            print("✗ No arXiv match found")

if __name__ == "__main__":
    main()