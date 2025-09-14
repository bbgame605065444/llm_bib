import json
import time
import logging
from typing import Dict, List, Optional
from serpapi import GoogleSearch
import os
from datetime import datetime

class ScholarScraper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.results = []
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def search_scholar(self, keyword: str, max_results: int = 3000) -> List[Dict]:
        """
        Search Google Scholar for papers related to the keyword
        
        Args:
            keyword: Search term
            max_results: Maximum number of results to retrieve (default 3000)
        
        Returns:
            List of paper dictionaries
        """
        self.logger.info(f"Starting search for keyword: '{keyword}'")
        self.results = []
        
        # SerpApi allows 20 results per page for Google Scholar
        results_per_page = 20
        total_pages = min((max_results + results_per_page - 1) // results_per_page, 150)  # API limit
        
        for page in range(total_pages):
            start_index = page * results_per_page
            
            self.logger.info(f"Fetching page {page + 1}/{total_pages} (results {start_index + 1}-{min(start_index + results_per_page, max_results)})")
            
            params = {
                "engine": "google_scholar",
                "q": keyword,
                "api_key": self.api_key,
                "start": start_index,
                "num": results_per_page
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "organic_results" not in results:
                self.logger.warning(f"No organic results found on page {page + 1}")
                if "error" in results:
                    self.logger.error(f"API Error: {results['error']}")
                break
            
            organic_results = results["organic_results"]
            
            for result in organic_results:
                paper_data = self._extract_paper_data(result)
                if paper_data:
                    self.results.append(paper_data)
            
            # Check if we have enough results
            if len(self.results) >= max_results:
                self.logger.info(f"Reached target of {max_results} results")
                break
            
            # Rate limiting - respect API limits
            time.sleep(1)
        
        self.logger.info(f"Search completed. Total results collected: {len(self.results)}")
        return self.results
    
    def _extract_paper_data(self, result: Dict) -> Optional[Dict]:
        """Extract relevant data from a search result"""
        # Extract basic information
        paper_data = {
            "title": result.get("title", ""),
            "link": result.get("link", ""),
            "snippet": result.get("snippet", ""),
            "publication_info": result.get("publication_info", {})
        }
        
        # Extract authors
        if "publication_info" in result and "authors" in result["publication_info"]:
            paper_data["authors"] = [author["name"] for author in result["publication_info"]["authors"]]
        else:
            paper_data["authors"] = []
        
        # Extract year
        pub_info = result.get("publication_info", {})
        if "summary" in pub_info:
            summary = pub_info["summary"]
            # Try to extract year from publication info
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', summary)
            paper_data["year"] = int(year_match.group()) if year_match else None
        else:
            paper_data["year"] = None
        
        # Extract citation count
        if "inline_links" in result:
            cited_by_link = result["inline_links"].get("cited_by", {})
            if cited_by_link and "total" in cited_by_link:
                paper_data["citation_count"] = cited_by_link["total"]
            else:
                paper_data["citation_count"] = 0
        else:
            paper_data["citation_count"] = 0
        
        # Extract venue/journal
        if "publication_info" in result and "summary" in result["publication_info"]:
            summary = result["publication_info"]["summary"]
            # Try to extract venue from summary
            parts = summary.split(" - ")
            if len(parts) > 1:
                paper_data["venue"] = parts[0].strip()
            else:
                paper_data["venue"] = ""
        else:
            paper_data["venue"] = ""
        
        # Additional metadata
        paper_data["search_keyword"] = None  # Will be set when saving
        paper_data["scraped_at"] = datetime.now().isoformat()
        
        # Use snippet as abstract if no abstract is directly available
        paper_data["abstract"] = paper_data["snippet"]
        
        return paper_data
    
    def save_results(self, filename: str, keyword: str) -> str:
        """
        Save results to JSON file
        
        Args:
            filename: Output filename
            keyword: Search keyword used
        
        Returns:
            Path to saved file
        """
        # Add keyword to each result
        for result in self.results:
            result["search_keyword"] = keyword
        
        # Create metadata
        metadata = {
            "search_keyword": keyword,
            "total_results": len(self.results),
            "scraped_at": datetime.now().isoformat(),
            "scraper_version": "1.0"
        }
        
        # Combine metadata and results
        output_data = {
            "metadata": metadata,
            "results": self.results
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {filename}")
        return filename
    
    def get_statistics(self) -> Dict:
        """Get basic statistics about scraped results"""
        if not self.results:
            return {}
        
        years = [r["year"] for r in self.results if r["year"]]
        citations = [r["citation_count"] for r in self.results if r["citation_count"]]
        
        stats = {
            "total_papers": len(self.results),
            "papers_with_year": len(years),
            "year_range": f"{min(years)}-{max(years)}" if years else "N/A",
            "total_citations": sum(citations),
            "avg_citations": sum(citations) / len(citations) if citations else 0,
            "papers_with_citations": len([c for c in citations if c > 0])
        }
        
        return stats

def main():
    """Example usage"""
    # This would typically be called from the main application
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("Please set SERPAPI_KEY environment variable")
        return
    
    scraper = ScholarScraper(api_key)
    
    keyword = input("Enter search keyword: ")
    max_results = int(input("Enter maximum results (default 100): ") or "100")
    
    results = scraper.search_scholar(keyword, max_results)
    
    if results:
        filename = f"scholar_results_{keyword.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        scraper.save_results(filename, keyword)
        
        stats = scraper.get_statistics()
        print("\nScraping Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print("No results found")

if __name__ == "__main__":
    main()