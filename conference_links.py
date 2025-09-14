import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin, urlparse, parse_qs
import argparse


def get_conference_links(base_url="https://papers.cool", conf_csv_filename="conference_links.csv", all_csv_filename="all_links.csv"):
    """Get list of conferences from the main page and save to CSV"""
    # Set up session with headers
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    
    try:
        # Fetch the main page
        response = session.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links
        all_links_data = []
        conference_data = []
        links = soup.find_all('a', href=True)
        
        print(f"Found {len(links)} links on the page.")
        
        for link in links:
            href = link['href']
            full_url = urljoin(base_url, href)
            title = link.get_text(strip=True)
            
            # Add all links to all_links_data
            all_links_data.append({
                'title': title,
                'url': full_url
            })
            
            # Look for conference or journal links
            if '/conference/' in href or '/journal/' in href or '/venue/' in href:
                # Only add if we have a title
                if title:
                    conference_data.append({
                        'title': title,
                        'url': full_url
                    })
                    print(f"Found conference: {title} -> {full_url}")
        
        # If we're on a specific venue page and didn't find any conferences, 
        # treat the current page as a conference page
        if not conference_data and '/venue/' in base_url:
            # Extract venue name from URL
            parsed_url = urlparse(base_url)
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 2 and path_parts[0] == 'venue':
                venue_name = path_parts[1]
                # Get query parameters
                query_params = parse_qs(parsed_url.query)
                show_param = query_params.get('show', ['default'])[0]
                
                conference_data.append({
                    'title': f"{venue_name} (show={show_param})",
                    'url': base_url
                })
                print(f"Treating current page as conference: {venue_name} (show={show_param}) -> {base_url}")
        
        # Remove duplicates based on URL for conferences
        unique_conferences = []
        seen_conf_urls = set()
        
        for conf in conference_data:
            if conf['url'] not in seen_conf_urls:
                unique_conferences.append(conf)
                seen_conf_urls.add(conf['url'])
        
        # Remove duplicates based on URL for all links
        unique_all_links = []
        seen_all_urls = set()
        
        for link in all_links_data:
            if link['url'] not in seen_all_urls:
                unique_all_links.append(link)
                seen_all_urls.add(link['url'])
        
        print(f"\nFound {len(unique_conferences)} unique conferences/journals.")
        print(f"Found {len(unique_all_links)} unique links in total.")
        
        # Save conference links to CSV
        with open(conf_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for conf in unique_conferences:
                writer.writerow(conf)
        
        print(f"Conference links saved to {conf_csv_filename}")
        
        # Save all links to CSV
        with open(all_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for link in unique_all_links:
                writer.writerow(link)
        
        print(f"All links saved to {all_csv_filename}")
        
        return unique_conferences
        
    except requests.RequestException as e:
        print(f"Error fetching {base_url}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Scrape conference links from papers.cool")
    parser.add_argument("--base-url", default="https://papers.cool", help="Base URL to scrape (default: https://papers.cool)")
    parser.add_argument("--conf-csv", default="conference_links.csv", help="CSV file name for conference links (default: conference_links.csv)")
    parser.add_argument("--all-csv", default="all_links.csv", help="CSV file name for all links (default: all_links.csv)")
    
    args = parser.parse_args()
    
    print(f"Fetching links from {args.base_url}...")
    conferences = get_conference_links(args.base_url, args.conf_csv, args.all_csv)
    print("Done!")


if __name__ == "__main__":
    main()