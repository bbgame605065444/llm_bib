#!/usr/bin/env python3
"""
Full System Test - Demonstrates complete workflow
Tests scraping titles from conferences and saving to database.
"""

import time
from conference_scraper import ConferenceScraper
from conference_registry import ConferenceRegistry
from database_manager import DatabaseManager

def test_conference_scraping():
    """Test the complete conference scraping workflow"""
    
    print("🚀 FULL SYSTEM TEST - Conference Paper Scraping")
    print("=" * 60)
    
    # Step 1: Initialize registry
    print("📋 Step 1: Loading conference registry...")
    registry = ConferenceRegistry()
    print(f"✅ Loaded {len(registry.conferences)} conferences")
    
    # Show available conferences
    print("\n🏛️ Available conferences:")
    for conf_type in ['AI/ML', 'Computer Vision', 'Natural Language Processing']:
        confs = [c for c in registry.conferences if c.conference_type.value == conf_type]
        if confs:
            print(f"\n   {conf_type}:")
            for conf in confs[:3]:  # Show first 3 of each type
                print(f"     • {conf.acronym} - {conf.full_name}")
    
    # Step 2: Initialize scraper
    print(f"\n💾 Step 2: Setting up database and scraper...")
    db_path = 'full_system_test.db'
    scraper = ConferenceScraper(
        db_path=db_path,
        enable_arxiv=False,  # Disable for faster testing
        rate_limit_delay=0.5,
        batch_size=20
    )
    print("✅ Scraper initialized")
    
    # Step 3: Test scraping multiple conferences
    print(f"\n🔄 Step 3: Scraping conference papers...")
    
    test_conferences = ['NeurIPS', 'ICML', 'AAAI']
    test_year = 2023
    max_papers_per_conf = 10  # Limit for testing
    
    total_papers = 0
    
    for conf_name in test_conferences:
        print(f"\n   Scraping {conf_name} {test_year} (max {max_papers_per_conf} papers)...")
        
        try:
            start_time = time.time()
            results = scraper.scrape_conference(
                conf_name, 
                [test_year], 
                max_papers_per_conf
            )
            elapsed = time.time() - start_time
            
            papers_found = results.get('total_papers', 0)
            total_papers += papers_found
            
            print(f"   ✅ {conf_name}: {papers_found} papers in {elapsed:.1f}s")
            
            if results.get('errors'):
                print(f"   ⚠️  {len(results['errors'])} errors occurred")
        
        except Exception as e:
            print(f"   ❌ {conf_name}: Failed - {e}")
    
    # Step 4: Check database contents
    print(f"\n📊 Step 4: Verifying database contents...")
    db = DatabaseManager(db_path)
    stats = db.get_database_stats()
    
    print(f"✅ Database Statistics:")
    print(f"   Total papers: {stats.get('total_papers', 0)}")
    print(f"   Unique venues: {stats.get('unique_venues', 0)}")
    print(f"   Database size: {stats.get('database_size_mb', 0):.1f} MB")
    
    # Show papers by venue
    if stats.get('top_venues'):
        print(f"\n📚 Papers by venue:")
        for venue, count in stats['top_venues'].items():
            print(f"   {venue}: {count} papers")
    
    # Step 5: Show sample papers
    print(f"\n📄 Step 5: Sample papers in database...")
    papers = db.search_papers("", limit=5)  # Get any 5 papers
    
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'No title')
        authors = paper.get('authors', '[]')
        if isinstance(authors, str):
            import json
            authors = json.loads(authors) if authors != 'None' else []
        
        print(f"\n   {i}. {title[:60]}{'...' if len(title) > 60 else ''}")
        print(f"      Authors: {', '.join(authors[:2])}{'...' if len(authors) > 2 else ''}")
        print(f"      Venue: {paper.get('venue')} {paper.get('year')}")
        print(f"      Source: {paper.get('source')}")
    
    # Step 6: Test export functionality
    print(f"\n📤 Step 6: Testing data export...")
    export_file = 'test_export.csv'
    success = db.export_to_csv(export_file)
    
    if success:
        import os
        file_size = os.path.getsize(export_file) if os.path.exists(export_file) else 0
        print(f"✅ Exported to {export_file} ({file_size:,} bytes)")
    else:
        print(f"❌ Export failed")
    
    # Final summary
    print(f"\n🎉 SYSTEM TEST COMPLETED")
    print(f"=" * 60)
    print(f"Conferences tested: {len(test_conferences)}")
    print(f"Total papers scraped: {total_papers}")
    print(f"Database papers: {stats.get('total_papers', 0)}")
    print(f"System status: {'✅ WORKING' if total_papers > 0 else '❌ ISSUES'}")
    
    return total_papers > 0

def test_cli_interface():
    """Test the CLI interface"""
    
    print(f"\n🖥️  CLI INTERFACE TEST")
    print("=" * 40)
    
    from conference_cli import ConferenceCLI
    
    # Test CLI initialization
    try:
        cli = ConferenceCLI()
        print("✅ CLI initialized successfully")
        
        # Mock args for testing
        class MockArgs:
            db_path = 'test_cli.db'
            verbose = False
            no_arxiv = True
        
        # Test info command
        print("\n📋 Testing info command...")
        cli.cmd_info(MockArgs())
        
        return True
    
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    """Run all system tests"""
    
    print("🧪 RUNNING FULL SYSTEM TESTS")
    print("=" * 70)
    
    # Test 1: Conference scraping
    test1_success = test_conference_scraping()
    
    # Test 2: CLI interface  
    test2_success = test_cli_interface()
    
    # Final results
    print(f"\n🏁 FINAL RESULTS")
    print("=" * 30)
    print(f"Conference scraping: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"CLI interface: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    overall_success = test1_success and test2_success
    print(f"Overall system: {'✅ WORKING' if overall_success else '❌ NEEDS FIXES'}")
    
    if overall_success:
        print(f"\n🎉 System is ready for production use!")
        print(f"Next steps:")
        print(f"  • Run: python -c 'from conference_cli import ConferenceCLI; ConferenceCLI().main()' --help")
        print(f"  • Or use: python run_scraper.py")
        print(f"  • For bulk scraping: python -c 'from conference_cli import ConferenceCLI; cli = ConferenceCLI()' bulk-scrape")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)