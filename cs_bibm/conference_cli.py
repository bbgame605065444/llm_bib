"""
Conference CLI - Command-line interface for conference paper scraping
Provides user-friendly commands for scraping, monitoring, and exporting.
"""

import argparse
import sys
from typing import List, Optional
from pathlib import Path
import json
from datetime import datetime

from conference_scraper import ConferenceScraper
from conference_registry import ConferenceRegistry, ConferenceType
from database_manager import DatabaseManager

class ConferenceCLI:
    """Command-line interface for conference scraping operations"""
    
    def __init__(self):
        self.registry = ConferenceRegistry()
    
    def main(self):
        """Main CLI entry point"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        if not hasattr(args, 'func'):
            parser.print_help()
            return
        
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\\n⚠️  Operation interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive CLI parser"""
        parser = argparse.ArgumentParser(
            description="Conference Paper Scraper CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # List all conferences
  python conference_cli.py list-conferences
  
  # Scrape NeurIPS for recent years
  python conference_cli.py scrape --conferences neurips --years 2022 2023 2024
  
  # Scrape all AI/ML conferences for 2023
  python conference_cli.py scrape-by-type --type "AI/ML" --years 2023
  
  # Bulk scrape top 10 conferences
  python conference_cli.py bulk-scrape --max-conferences 10
  
  # Export all data to CSV
  python conference_cli.py export --format csv --output all_papers.csv
  
  # Check database status
  python conference_cli.py db-status
            """
        )
        
        # Global options
        parser.add_argument('--db-path', default='cs_bibm/data/conference_papers.db',
                           help='Database file path')
        parser.add_argument('--verbose', '-v', action='store_true',
                           help='Enable verbose logging')
        parser.add_argument('--no-arxiv', action='store_true',
                           help='Disable arXiv matching')
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # List conferences command
        list_parser = subparsers.add_parser('list-conferences', 
                                           help='List all registered conferences')
        list_parser.add_argument('--type', choices=[t.value for t in ConferenceType],
                                help='Filter by conference type')
        list_parser.add_argument('--year', type=int,
                                help='Show conferences available for specific year')
        list_parser.set_defaults(func=self.cmd_list_conferences)
        
        # Scrape specific conferences
        scrape_parser = subparsers.add_parser('scrape', 
                                             help='Scrape specific conferences')
        scrape_parser.add_argument('--conferences', nargs='+', required=True,
                                  help='Conference acronyms (e.g., neurips icml)')
        scrape_parser.add_argument('--years', nargs='+', type=int,
                                  help='Years to scrape (default: 2020-2024)')
        scrape_parser.add_argument('--max-papers', type=int, default=0,
                                  help='Maximum papers per venue (0 for unlimited)')
        scrape_parser.add_argument('--batch-size', type=int, default=50,
                                  help='Batch size for database operations')
        scrape_parser.set_defaults(func=self.cmd_scrape)
        
        # Scrape by conference type
        type_parser = subparsers.add_parser('scrape-by-type',
                                           help='Scrape all conferences of a specific type')
        type_parser.add_argument('--type', required=True, 
                                choices=[t.value for t in ConferenceType],
                                help='Conference type to scrape')
        type_parser.add_argument('--years', nargs='+', type=int,
                               help='Years to scrape (default: 2020-2024)')
        type_parser.add_argument('--max-papers', type=int, default=0,
                               help='Maximum papers per venue')
        type_parser.set_defaults(func=self.cmd_scrape_by_type)
        
        # Bulk scrape (all or top N conferences)
        bulk_parser = subparsers.add_parser('bulk-scrape',
                                           help='Bulk scrape multiple conferences')
        bulk_parser.add_argument('--max-conferences', type=int, default=0,
                                help='Maximum conferences to scrape (0 for all)')
        bulk_parser.add_argument('--years', nargs='+', type=int,
                                help='Years to scrape (default: 2020-2024)')
        bulk_parser.add_argument('--max-papers-per-venue', type=int, default=0,
                                help='Maximum papers per venue')
        bulk_parser.add_argument('--skip-types', nargs='+',
                                choices=[t.value for t in ConferenceType],
                                help='Conference types to skip')
        bulk_parser.set_defaults(func=self.cmd_bulk_scrape)
        
        # Database status
        status_parser = subparsers.add_parser('db-status',
                                             help='Show database statistics')
        status_parser.add_argument('--detailed', action='store_true',
                                  help='Show detailed statistics')
        status_parser.set_defaults(func=self.cmd_db_status)
        
        # Export data
        export_parser = subparsers.add_parser('export',
                                             help='Export scraped data')
        export_parser.add_argument('--format', choices=['csv', 'json', 'excel'],
                                  default='csv', help='Export format')
        export_parser.add_argument('--output', '-o', required=True,
                                  help='Output file path')
        export_parser.add_argument('--venue', 
                                  help='Export specific venue only')
        export_parser.add_argument('--year', type=int,
                                  help='Export specific year only')
        export_parser.set_defaults(func=self.cmd_export)
        
        # Search papers
        search_parser = subparsers.add_parser('search',
                                             help='Search papers in database')
        search_parser.add_argument('query', help='Search query')
        search_parser.add_argument('--limit', type=int, default=20,
                                  help='Maximum results to show')
        search_parser.add_argument('--venue',
                                  help='Filter by venue')
        search_parser.set_defaults(func=self.cmd_search)
        
        # Estimate papers
        estimate_parser = subparsers.add_parser('estimate',
                                               help='Estimate total papers for scraping')
        estimate_parser.add_argument('--conferences', nargs='+',
                                    help='Specific conferences (default: all)')
        estimate_parser.add_argument('--years', nargs='+', type=int,
                                    help='Years to estimate (default: 2020-2024)')
        estimate_parser.set_defaults(func=self.cmd_estimate)
        
        # System info
        info_parser = subparsers.add_parser('info',
                                           help='Show system information')
        info_parser.set_defaults(func=self.cmd_info)
        
        return parser
    
    def cmd_list_conferences(self, args):
        """List registered conferences"""
        print("📋 REGISTERED CONFERENCES")
        print("=" * 50)
        
        conferences = self.registry.conferences
        
        # Filter by type if specified
        if args.type:
            type_enum = next(t for t in ConferenceType if t.value == args.type)
            conferences = self.registry.get_conferences_by_type(type_enum)
        
        # Filter by year if specified
        if args.year:
            conferences = [c for c in conferences if args.year in c.years_available]
        
        # Group by type
        by_type = {}
        for conf in conferences:
            type_name = conf.conference_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(conf)
        
        for conf_type, confs in sorted(by_type.items()):
            print(f"\\n🏷️  {conf_type}")
            print("-" * 40)
            
            for conf in sorted(confs, key=lambda x: x.acronym):
                years_str = f"{min(conf.years_available)}-{max(conf.years_available)}"
                print(f"  {conf.acronym:12} | {conf.full_name[:50]:50} | {years_str}")
                print(f"  {'':12} | DBLP: {conf.dblp_key:20} | ~{conf.typical_paper_count:,} papers/year")
        
        print(f"\\n📊 Total: {len(conferences)} conferences")
    
    def cmd_scrape(self, args):
        """Scrape specific conferences"""
        print("🚀 SCRAPING SPECIFIC CONFERENCES")
        print("=" * 50)
        
        # Validate conferences
        invalid_confs = []
        for conf in args.conferences:
            if not self.registry.get_conference(conf.upper()):
                invalid_confs.append(conf)
        
        if invalid_confs:
            print(f"❌ Invalid conferences: {', '.join(invalid_confs)}")
            print("Use 'list-conferences' to see available options")
            return
        
        # Setup scraper
        scraper = self._create_scraper(args)
        
        # Default years
        years = args.years or list(range(2020, 2025))
        
        print(f"Conferences: {', '.join([c.upper() for c in args.conferences])}")
        print(f"Years: {years}")
        print(f"Max papers per venue: {'unlimited' if args.max_papers == 0 else args.max_papers}")
        
        # Scrape each conference
        total_papers = 0
        for conf_name in args.conferences:
            print(f"\\n🔄 Scraping {conf_name.upper()}...")
            
            try:
                results = scraper.scrape_conference(
                    conf_name.upper(), 
                    years, 
                    args.max_papers
                )
                
                total_papers += results['total_papers']
                print(f"✅ {conf_name.upper()}: {results['total_papers']} papers")
                
                if results['errors']:
                    print(f"⚠️  Errors: {len(results['errors'])}")
                
            except Exception as e:
                print(f"❌ {conf_name.upper()}: {e}")
        
        print(f"\\n🎉 Total papers scraped: {total_papers:,}")
    
    def cmd_scrape_by_type(self, args):
        """Scrape conferences by type"""
        print(f"🚀 SCRAPING {args.type.upper()} CONFERENCES")
        print("=" * 50)
        
        # Get conferences of specified type
        type_enum = next(t for t in ConferenceType if t.value == args.type)
        conferences = self.registry.get_conferences_by_type(type_enum)
        
        if not conferences:
            print(f"❌ No conferences found for type: {args.type}")
            return
        
        # Setup scraper
        scraper = self._create_scraper(args)
        years = args.years or list(range(2020, 2025))
        
        print(f"Conferences ({len(conferences)}): {', '.join([c.acronym for c in conferences])}")
        print(f"Years: {years}")
        
        # Scrape all conferences of this type
        results = scraper.scrape_all_conferences(
            years=years,
            max_papers_per_venue=args.max_papers,
            conference_types=[args.type]
        )
        
        self._print_scraping_summary(results)
    
    def cmd_bulk_scrape(self, args):
        """Bulk scrape conferences"""
        print("🚀 BULK CONFERENCE SCRAPING")
        print("=" * 50)
        
        # Setup scraper
        scraper = self._create_scraper(args)
        years = args.years or list(range(2020, 2025))
        
        # Filter conferences
        conference_types = None
        if args.skip_types:
            all_types = [t.value for t in ConferenceType]
            conference_types = [t for t in all_types if t not in args.skip_types]
        
        # Get estimate
        if args.max_conferences > 0:
            priority_confs = self.registry.get_scraping_priority()[:args.max_conferences]
            estimates = self.registry.estimate_total_papers(priority_confs, years)
        else:
            estimates = self.registry.estimate_total_papers(years=years)
        
        print(f"Estimated papers: {estimates['TOTAL']:,}")
        print(f"Years: {years}")
        
        if args.max_conferences > 0:
            print(f"Limited to top {args.max_conferences} conferences")
        
        # Confirm with user
        confirm = input("\\n🤔 Proceed with bulk scraping? [y/N]: ")
        if confirm.lower() != 'y':
            print("❌ Operation cancelled")
            return
        
        # Start bulk scraping
        print("\\n🚀 Starting bulk scraping...")
        results = scraper.scrape_all_conferences(
            years=years,
            max_papers_per_venue=args.max_papers_per_venue,
            conference_types=conference_types
        )
        
        self._print_scraping_summary(results)
    
    def cmd_db_status(self, args):
        """Show database status"""
        print("📊 DATABASE STATUS")
        print("=" * 50)
        
        try:
            db = DatabaseManager(args.db_path)
            stats = db.get_database_stats()
            
            # Basic stats
            print(f"Total papers: {stats.get('total_papers', 0):,}")
            print(f"Unique venues: {stats.get('unique_venues', 0)}")
            print(f"Years covered: {stats.get('years_covered', 0)}")
            print(f"Database size: {stats.get('database_size_mb', 0):.1f} MB")
            
            # Quality metrics
            print(f"\\n📈 QUALITY METRICS")
            print(f"Papers with DOI: {stats.get('papers_with_doi', 0):,}")
            print(f"Papers with arXiv: {stats.get('papers_with_arxiv', 0):,}")
            print(f"Papers with PDF: {stats.get('papers_with_pdf', 0):,}")
            print(f"Average citations: {stats.get('avg_citations', 0):.1f}")
            
            if args.detailed:
                # Year distribution
                if 'papers_by_year' in stats:
                    print(f"\\n📅 PAPERS BY YEAR")
                    for year, count in sorted(stats['papers_by_year'].items()):
                        print(f"  {year}: {count:,}")
                
                # Top venues
                if 'top_venues' in stats:
                    print(f"\\n🏛️  TOP VENUES")
                    for venue, count in list(stats['top_venues'].items())[:10]:
                        print(f"  {venue:12}: {count:,}")
                
                # Source distribution
                if 'papers_by_source' in stats:
                    print(f"\\n🔍 PAPERS BY SOURCE")
                    for source, count in stats['papers_by_source'].items():
                        print(f"  {source:12}: {count:,}")
                        
        except Exception as e:
            print(f"❌ Error accessing database: {e}")
    
    def cmd_export(self, args):
        """Export data"""
        print(f"📤 EXPORTING DATA TO {args.format.upper()}")
        print("=" * 50)
        
        try:
            db = DatabaseManager(args.db_path)
            
            if args.format == 'csv':
                success = db.export_to_csv(args.output, args.venue)
                if success:
                    print(f"✅ Data exported to: {args.output}")
                else:
                    print(f"❌ Export failed")
            
            elif args.format == 'json':
                # Custom JSON export
                self._export_to_json(db, args.output, args.venue, args.year)
            
            elif args.format == 'excel':
                # Custom Excel export
                self._export_to_excel(db, args.output, args.venue, args.year)
                
        except Exception as e:
            print(f"❌ Export error: {e}")
    
    def cmd_search(self, args):
        """Search papers"""
        print(f"🔍 SEARCHING: '{args.query}'")
        print("=" * 50)
        
        try:
            db = DatabaseManager(args.db_path)
            
            if args.venue:
                # Search within specific venue
                papers = db.get_papers_by_venue(args.venue)
                # Filter by query (simple text search)
                query_lower = args.query.lower()
                papers = [p for p in papers if query_lower in p.get('title', '').lower()]
            else:
                # Full database search
                papers = db.search_papers(args.query, args.limit)
            
            if not papers:
                print("❌ No papers found")
                return
            
            print(f"Found {len(papers)} papers:\\n")
            
            for i, paper in enumerate(papers[:args.limit], 1):
                print(f"{i:2d}. {paper.get('title', 'No title')}")
                authors = paper.get('authors')
                if isinstance(authors, str):
                    authors = json.loads(authors) if authors else []
                print(f"    Authors: {', '.join(authors[:3])}")
                print(f"    Venue: {paper.get('venue', 'N/A')} {paper.get('year', 'N/A')}")
                if paper.get('arxiv_id'):
                    print(f"    arXiv: {paper['arxiv_id']}")
                print()
                
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    def cmd_estimate(self, args):
        """Estimate paper counts"""
        print("📊 PAPER COUNT ESTIMATION")
        print("=" * 50)
        
        conferences = args.conferences
        years = args.years or list(range(2020, 2025))
        
        estimates = self.registry.estimate_total_papers(conferences, years)
        
        print(f"Years: {years}")
        if conferences:
            print(f"Conferences: {', '.join(conferences)}")
        else:
            print("Conferences: All registered")
        
        print(f"\\n📈 ESTIMATES")
        print("-" * 30)
        
        for conf, count in sorted(estimates.items()):
            if conf != 'TOTAL':
                print(f"{conf:12}: {count:,} papers")
        
        print("-" * 30)
        print(f"{'TOTAL':12}: {estimates['TOTAL']:,} papers")
    
    def cmd_info(self, args):
        """Show system information"""
        print("ℹ️  SYSTEM INFORMATION")
        print("=" * 50)
        
        print(f"Registered conferences: {len(self.registry.conferences)}")
        print(f"Database path: {args.db_path}")
        
        # Conference types
        print(f"\\n🏷️  CONFERENCE TYPES")
        for conf_type in ConferenceType:
            confs = self.registry.get_conferences_by_type(conf_type)
            print(f"  {conf_type.value}: {len(confs)} conferences")
        
        # Top conferences by paper count
        print(f"\\n🏆 TOP CONFERENCES (by estimated papers)")
        priority = self.registry.get_scraping_priority()[:10]
        for i, acronym in enumerate(priority, 1):
            conf = self.registry.get_conference(acronym)
            total_papers = len(conf.years_available) * conf.typical_paper_count
            print(f"  {i:2d}. {acronym:12} (~{total_papers:,} total papers)")
        
        # Check database
        if Path(args.db_path).exists():
            try:
                db = DatabaseManager(args.db_path)
                stats = db.get_database_stats()
                print(f"\\n💾 DATABASE STATUS")
                print(f"  Papers in database: {stats.get('total_papers', 0):,}")
                print(f"  Database size: {stats.get('database_size_mb', 0):.1f} MB")
            except:
                print(f"\\n💾 Database exists but unable to read statistics")
        else:
            print(f"\\n💾 Database not found (will be created on first scrape)")
    
    def _create_scraper(self, args) -> ConferenceScraper:
        """Create scraper instance with CLI arguments"""
        return ConferenceScraper(
            db_path=args.db_path,
            enable_arxiv=not args.no_arxiv,
            rate_limit_delay=1.0,
            batch_size=getattr(args, 'batch_size', 50)
        )
    
    def _print_scraping_summary(self, results: dict):
        """Print formatted scraping summary"""
        stats = results.get('statistics', {})
        
        print(f"\\n🎉 SCRAPING COMPLETED")
        print("=" * 50)
        print(f"Duration: {self._format_duration(stats)}")
        print(f"Conferences scraped: {stats.get('conferences_scraped', 0)}")
        print(f"Total papers found: {stats.get('papers_found', 0):,}")
        print(f"Papers inserted: {stats.get('papers_inserted', 0):,}")
        print(f"Papers updated: {stats.get('papers_updated', 0):,}")
        print(f"arXiv matches: {stats.get('arxiv_matches', 0):,}")
        
        if stats.get('errors', 0) > 0:
            print(f"⚠️  Errors: {stats['errors']}")
    
    def _format_duration(self, stats: dict) -> str:
        """Format duration from stats"""
        start = stats.get('start_time')
        end = stats.get('end_time')
        
        if not start or not end:
            return "Unknown"
        
        if isinstance(start, str):
            from datetime import datetime
            start = datetime.fromisoformat(start.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end.replace('Z', '+00:00'))
        
        duration = (end - start).total_seconds()
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def _export_to_json(self, db: DatabaseManager, output_path: str, 
                       venue: Optional[str], year: Optional[int]):
        """Export data to JSON format"""
        # This would be implemented based on specific JSON export requirements
        print("🚧 JSON export not yet implemented")
    
    def _export_to_excel(self, db: DatabaseManager, output_path: str, 
                        venue: Optional[str], year: Optional[int]):
        """Export data to Excel format"""
        # This would be implemented based on specific Excel export requirements  
        print("🚧 Excel export not yet implemented")

def main():
    """CLI entry point"""
    cli = ConferenceCLI()
    cli.main()

if __name__ == "__main__":
    main()