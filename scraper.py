import requests
from bs4 import BeautifulSoup
import json
import time
import os
from urllib.parse import urljoin, urlparse


class PapersCoolScraper:
    def __init__(self, base_url="https://papers.cool", output_file="papers_data.json"):
        self.base_url = base_url
        self.output_file = output_file
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        
    def get_page(self, url):
        """Fetch a page and return BeautifulSoup object"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def get_conferences(self):
        """Get list of conferences from the main page"""
        soup = self.get_page(self.base_url)
        if not soup:
            return []
        
        # This will need to be adjusted based on the actual site structure
        # Looking for links to conference pages
        conference_links = []
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link['href']
            if '/conference/' in href or '/journal/' in href:
                full_url = urljoin(self.base_url, href)
                conference_links.append({
                    'name': link.get_text(strip=True),
                    'url': full_url
                })
        
        return conference_links
    
    def get_papers_from_conference(self, conf_url, conf_name):
        """Extract papers from a conference page"""
        soup = self.get_page(conf_url)
        if not soup:
            return []
        
        papers = []
        
        # This will need to be adjusted based on actual site structure
        # Looking for paper entries
        paper_elements = soup.find_all(['div', 'article'], class_=['paper', 'entry', 'item'])
        
        if not paper_elements:
            # Try a more general approach
            paper_elements = soup.find_all('div')
        
        for elem in paper_elements:
            title_elem = elem.find(['h1', 'h2', 'h3', 'h4', 'a'], class_=['title', 'paper-title'])
            if not title_elem:
                continue
                
            title = title_elem.get_text(strip=True)
            
            # Try to find authors
            authors_elem = elem.find(class_=['authors', 'author-list'])
            authors = []
            if authors_elem:
                authors = [a.get_text(strip=True) for a in authors_elem.find_all('a')]
            
            # Try to find category (oral, spotlight, poster, etc.)
            category_elem = elem.find(class_=['category', 'type'])
            category = category_elem.get_text(strip=True) if category_elem else "Unknown"
            
            papers.append({
                'conference': conf_name,
                'title': title,
                'authors': authors,
                'category': category
            })
        
        return papers
    
    def save_papers_batch(self, papers, batch_number):
        """Save a batch of papers to JSON file"""
        filename = f"papers_batch_{batch_number}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"Saved batch {batch_number} with {len(papers)} papers to {filename}")
    
    def append_to_main_file(self, papers):
        """Append papers to main JSON file"""
        data = []
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        
        data.extend(papers)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def scrape_all_papers(self, batch_size=1000):
        """Main scraping function"""
        print("Getting list of conferences...")
        conferences = self.get_conferences()
        print(f"Found {len(conferences)} conferences/journals")
        
        all_papers = []
        paper_count = 0
        batch_number = 1
        
        for conf in conferences:
            print(f"Scraping papers from {conf['name']}...")
            papers = self.get_papers_from_conference(conf['url'], conf['name'])
            print(f"Found {len(papers)} papers")
            
            for paper in papers:
                all_papers.append(paper)
                paper_count += 1
                
                # Save batch when we reach batch_size
                if len(all_papers) >= batch_size:
                    self.save_papers_batch(all_papers, batch_number)
                    self.append_to_main_file(all_papers)
                    all_papers = []  # Reset for next batch
                    batch_number += 1
                    
                # Be respectful to the server
                time.sleep(0.1)
        
        # Save any remaining papers
        if all_papers:
            self.save_papers_batch(all_papers, batch_number)
            self.append_to_main_file(all_papers)
        
        print(f"Scraping complete. Total papers collected: {paper_count}")


def main():
    scraper = PapersCoolScraper()
    scraper.scrape_all_papers(batch_size=1000)


if __name__ == "__main__":
    main()