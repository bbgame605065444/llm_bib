#!/usr/bin/env python3
"""
Bulk arXiv Paper Downloader
Downloads PDF files for all papers with arXiv IDs in the database.

Usage:
    python download_arxiv_papers.py --output-dir ./papers/
    python download_arxiv_papers.py --venue neurips --year 2023 --output-dir ./neurips_2023/
    python download_arxiv_papers.py --max-papers 50 --output-dir ./test_papers/
"""

import argparse
import requests
import sqlite3
from pathlib import Path
import time
import logging
from urllib.parse import urlparse
import json
import os
import sys
from datetime import datetime

def setup_logging(log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    return logging.getLogger(__name__)

def get_papers_with_arxiv(db_path, venue=None, year=None, max_papers=0):
    """Get all papers that have arXiv IDs from the database"""
    
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    query = """
        SELECT paper_id, title, authors, venue, year, arxiv_id, arxiv_url, pdf_url, doi
        FROM papers 
        WHERE arxiv_id IS NOT NULL AND arxiv_id != ''
    """
    params = []
    
    if venue:
        query += " AND UPPER(venue) = ?"
        params.append(venue.upper())
    
    if year:
        query += " AND year = ?"
        params.append(year)
    
    query += " ORDER BY venue, year, title"
    
    if max_papers > 0:
        query += f" LIMIT {max_papers}"
    
    cursor = conn.execute(query, params)
    papers = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return papers

def sanitize_filename(filename, max_length=200):
    """Clean filename for filesystem compatibility"""
    # Remove or replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove multiple spaces and strip
    filename = ' '.join(filename.split())
    
    # Limit length
    if len(filename) > max_length:
        filename = filename[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
    
    return filename.strip()

def download_arxiv_pdf(arxiv_id, output_dir, paper_info, logger, organize_by_venue=False):
    """Download a single arXiv PDF"""
    
    # Create directory structure if organizing by venue
    if organize_by_venue:
        venue_dir = Path(output_dir) / paper_info['venue'] / str(paper_info['year'])
        venue_dir.mkdir(parents=True, exist_ok=True)
        target_dir = venue_dir
    else:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create clean filename
    title = paper_info['title'][:80]  # Reasonable length limit
    clean_title = sanitize_filename(title)
    filename = f"{arxiv_id}_{clean_title}.pdf"
    output_path = target_dir / filename
    
    # Skip if already exists
    if output_path.exists():
        file_size = output_path.stat().st_size
        if file_size > 1000:  # At least 1KB - assume valid PDF
            logger.info(f"⏭️  Skipping {arxiv_id} - already downloaded ({file_size:,} bytes)")
            return True
        else:
            # Remove corrupted/empty file
            output_path.unlink()
    
    # arXiv PDF URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        logger.info(f"📥 Downloading {arxiv_id}: {paper_info['title'][:50]}...")
        
        # Use session for better connection handling
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Academic Research Tool) CS-BIBM arXiv Downloader'
        })
        
        response = session.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type:
            logger.warning(f"⚠️  {arxiv_id}: Unexpected content type: {content_type}")
        
        # Write file
        total_size = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        # Validate file size
        if total_size < 1000:  # Less than 1KB is suspicious
            logger.warning(f"⚠️  {arxiv_id}: Very small file size ({total_size} bytes)")
            return False
        
        logger.info(f"✅ Downloaded: {filename} ({total_size:,} bytes)")
        return True
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"❌ {arxiv_id}: Paper not found on arXiv (404)")
        else:
            logger.error(f"❌ {arxiv_id}: HTTP error {e.response.status_code}")
        return False
        
    except requests.exceptions.Timeout:
        logger.error(f"❌ {arxiv_id}: Download timeout")
        return False
        
    except Exception as e:
        logger.error(f"❌ {arxiv_id}: Download failed - {e}")
        return False

def generate_download_report(papers, successful, failed, output_dir, logger):
    """Generate a detailed download report"""
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_papers': len(papers),
            'successful_downloads': successful,
            'failed_downloads': failed,
            'success_rate': (successful / len(papers) * 100) if papers else 0
        },
        'venue_breakdown': {},
        'year_breakdown': {},
        'output_directory': str(output_dir)
    }
    
    # Analyze by venue and year
    for paper in papers:
        venue = paper['venue']
        year = str(paper['year'])
        
        if venue not in report_data['venue_breakdown']:
            report_data['venue_breakdown'][venue] = 0
        report_data['venue_breakdown'][venue] += 1
        
        if year not in report_data['year_breakdown']:
            report_data['year_breakdown'][year] = 0
        report_data['year_breakdown'][year] += 1
    
    # Save report
    report_file = Path(output_dir) / 'download_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📊 Download report saved: {report_file}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("📊 DOWNLOAD REPORT SUMMARY")
    print(f"{'='*50}")
    print(f"Total papers with arXiv IDs: {len(papers)}")
    print(f"Successful downloads: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Success rate: {successful / len(papers) * 100:.1f}%" if papers else "N/A")
    
    print(f"\n📚 BY VENUE:")
    for venue, count in sorted(report_data['venue_breakdown'].items()):
        print(f"  {venue:12}: {count} papers")
    
    print(f"\n📅 BY YEAR:")
    for year, count in sorted(report_data['year_breakdown'].items()):
        print(f"  {year}: {count} papers")

def main():
    parser = argparse.ArgumentParser(
        description="Bulk arXiv Paper Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all arXiv papers
  python download_arxiv_papers.py --output-dir ./all_papers/

  # Download NeurIPS 2023 papers only
  python download_arxiv_papers.py --venue neurips --year 2023 --output-dir ./neurips_2023/
  
  # Download with organized directory structure
  python download_arxiv_papers.py --output-dir ./papers/ --organize-by-venue
  
  # Test download with limited papers
  python download_arxiv_papers.py --max-papers 10 --output-dir ./test/
        """
    )
    
    parser.add_argument('--db-path', default='cs_bibm/data/conference_papers.db',
                       help='Path to conference papers database')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for downloaded PDFs')
    parser.add_argument('--venue', 
                       help='Filter by venue (e.g., NeurIPS, ICML, CVPR)')
    parser.add_argument('--year', type=int, 
                       help='Filter by year (e.g., 2023)')
    parser.add_argument('--max-papers', type=int, default=0,
                       help='Maximum papers to download (0 for all)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between downloads in seconds')
    parser.add_argument('--organize-by-venue', action='store_true',
                       help='Organize downloads in venue/year subdirectories')
    parser.add_argument('--resume', action='store_true',
                       help='Skip already downloaded files (default behavior)')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate detailed download report')
    parser.add_argument('--log-file',
                       help='Save logs to file (in addition to console)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file
    if log_file:
        log_path = Path(args.output_dir) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = None
    
    logger = setup_logging(log_path)
    
    try:
        # Get papers with arXiv IDs
        logger.info("🔍 Querying database for papers with arXiv IDs...")
        papers = get_papers_with_arxiv(
            args.db_path, 
            args.venue, 
            args.year, 
            args.max_papers
        )
        
        if not papers:
            logger.error("❌ No papers with arXiv IDs found matching your criteria")
            return
        
        logger.info(f"📋 Found {len(papers)} papers with arXiv IDs")
        
        if args.venue:
            logger.info(f"   Venue filter: {args.venue.upper()}")
        if args.year:
            logger.info(f"   Year filter: {args.year}")
        
        # Show venue breakdown
        venue_counts = {}
        for paper in papers:
            venue = paper['venue']
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
        
        logger.info("📊 Venue breakdown:")
        for venue, count in sorted(venue_counts.items()):
            logger.info(f"   {venue}: {count} papers")
        
        # Confirm with user for large downloads
        if len(papers) > 100:
            confirm = input(f"\n⚠️  About to download {len(papers)} papers. Continue? [y/N]: ")
            if confirm.lower() != 'y':
                logger.info("❌ Download cancelled by user")
                return
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir.absolute()}")
        
        # Start downloads
        logger.info(f"🚀 Starting downloads with {args.delay}s delay between requests...")
        
        successful = 0
        failed = 0
        start_time = datetime.now()
        
        for i, paper in enumerate(papers, 1):
            logger.info(f"\n[{i}/{len(papers)}] Processing: {paper['arxiv_id']}")
            
            success = download_arxiv_pdf(
                paper['arxiv_id'], 
                output_dir, 
                paper, 
                logger,
                args.organize_by_venue
            )
            
            if success:
                successful += 1
            else:
                failed += 1
            
            # Progress update every 10 papers
            if i % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = i / elapsed * 60  # papers per minute
                remaining = len(papers) - i
                eta = remaining / (rate / 60) if rate > 0 else 0
                
                logger.info(f"📈 Progress: {i}/{len(papers)} | Rate: {rate:.1f}/min | ETA: {eta/60:.1f}min")
            
            # Rate limiting (skip for last item)
            if i < len(papers) and args.delay > 0:
                time.sleep(args.delay)
        
        # Final summary
        elapsed_total = (datetime.now() - start_time).total_seconds()
        final_rate = len(papers) / elapsed_total * 60 if elapsed_total > 0 else 0
        
        print(f"\n{'='*60}")
        print("🎉 DOWNLOAD COMPLETED")
        print(f"{'='*60}")
        print(f"Total papers: {len(papers)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful / len(papers) * 100:.1f}%")
        print(f"Time elapsed: {elapsed_total/60:.1f} minutes")
        print(f"Download rate: {final_rate:.1f} papers/minute")
        print(f"Output directory: {output_dir.absolute()}")
        
        # Generate report if requested
        if args.generate_report:
            generate_download_report(papers, successful, failed, output_dir, logger)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Download interrupted by user")
        return
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()