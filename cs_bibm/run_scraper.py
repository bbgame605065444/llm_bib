#!/usr/bin/env python3
"""
Quick launcher script for CS-BIBM Conference Scraper
Provides easy access to common scraping operations.
"""

import sys
import argparse
from pathlib import Path

# Add cs_bibm to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from conference_cli import ConferenceCLI
    from conference_registry import ConferenceRegistry
    from config import Config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the cs_bibm directory and dependencies are installed:")
    print("  cd cs_bibm")
    print("  pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Main launcher with quick options"""
    
    if len(sys.argv) == 1:
        # Interactive mode if no arguments
        show_interactive_menu()
    else:
        # Pass through to CLI
        cli = ConferenceCLI()
        cli.main()

def show_interactive_menu():
    """Show interactive menu for common operations"""
    
    print("🚀 CS-BIBM Conference Scraper")
    print("=" * 50)
    
    # Validate configuration
    print("Checking system configuration...")
    if not Config.validate_config():
        print("⚠️  Configuration issues detected. Please check your setup.")
        return
    
    # Show available conferences
    registry = ConferenceRegistry()
    print(f"✅ System ready! {len(registry.conferences)} conferences available.")
    
    while True:
        print("\\n📋 QUICK OPTIONS:")
        print("1. List all conferences")
        print("2. Scrape a single conference (test)")
        print("3. Scrape multiple conferences")
        print("4. Database status")
        print("5. Export data")
        print("6. System information")
        print("7. Advanced CLI (pass-through)")
        print("0. Exit")
        
        try:
            choice = input("\\nSelect option [0-7]: ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                list_conferences_interactive()
            elif choice == '2':
                scrape_single_interactive()
            elif choice == '3':
                scrape_multiple_interactive()
            elif choice == '4':
                show_database_status()
            elif choice == '5':
                export_data_interactive()
            elif choice == '6':
                show_system_info()
            elif choice == '7':
                show_cli_help()
            else:
                print("❌ Invalid option. Please try again.")
                
        except KeyboardInterrupt:
            print("\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def list_conferences_interactive():
    """Interactive conference listing"""
    from conference_cli import ConferenceCLI
    
    print("\\n📋 Listing all conferences...")
    cli = ConferenceCLI()
    
    # Create mock args for list command
    class Args:
        type = None
        year = None
    
    cli.cmd_list_conferences(Args())

def scrape_single_interactive():
    """Interactive single conference scraping"""
    registry = ConferenceRegistry()
    
    print("\\n🚀 Single Conference Scraping")
    print("Available conferences:", ", ".join(registry.get_all_acronyms()))
    
    conference = input("Enter conference acronym (e.g., 'neurips'): ").strip().upper()
    
    if not registry.get_conference(conference):
        print(f"❌ Conference '{conference}' not found.")
        return
    
    year = input("Enter year (e.g., '2023') or press Enter for 2023: ").strip() or "2023"
    max_papers = input("Max papers (press Enter for 50): ").strip() or "50"
    
    try:
        year = int(year)
        max_papers = int(max_papers)
    except ValueError:
        print("❌ Invalid year or paper count.")
        return
    
    print(f"\\n🔄 Scraping {conference} {year} (max {max_papers} papers)...")
    
    # Execute scraping
    cmd = [
        "python", "-m", "cs_bibm.conference_cli", "scrape",
        "--conferences", conference.lower(),
        "--years", str(year),
        "--max-papers", str(max_papers)
    ]
    
    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print("✅ Scraping completed!")
    except subprocess.CalledProcessError:
        print("❌ Scraping failed. Check logs for details.")

def scrape_multiple_interactive():
    """Interactive multiple conference scraping"""
    registry = ConferenceRegistry()
    
    print("\\n🚀 Multiple Conference Scraping")
    print("Available conferences:", ", ".join(registry.get_all_acronyms()))
    
    conferences = input("Enter conference acronyms (space-separated, e.g., 'neurips icml'): ").strip()
    
    if not conferences:
        print("❌ No conferences specified.")
        return
    
    years = input("Enter years (space-separated, e.g., '2022 2023') or press Enter for 2023: ").strip()
    if not years:
        years = "2023"
    
    max_papers = input("Max papers per venue (press Enter for unlimited): ").strip() or "0"
    
    print(f"\\n🔄 Scraping conferences: {conferences}")
    print(f"Years: {years}")
    print(f"Max papers per venue: {'unlimited' if max_papers == '0' else max_papers}")
    
    confirm = input("Proceed? [y/N]: ")
    if confirm.lower() != 'y':
        print("❌ Cancelled.")
        return
    
    # Build command
    cmd = [
        "python", "-m", "cs_bibm.conference_cli", "scrape",
        "--conferences"
    ] + conferences.lower().split() + [
        "--years"
    ] + years.split() + [
        "--max-papers", max_papers
    ]
    
    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print("✅ Scraping completed!")
    except subprocess.CalledProcessError:
        print("❌ Scraping failed. Check logs for details.")

def show_database_status():
    """Show database status"""
    print("\\n📊 Database Status")
    
    cmd = ["python", "-m", "cs_bibm.conference_cli", "db-status", "--detailed"]
    
    import subprocess
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ Unable to access database.")

def export_data_interactive():
    """Interactive data export"""
    print("\\n📤 Export Data")
    
    format_choice = input("Export format (csv/json) [csv]: ").strip() or "csv"
    output_file = input("Output filename (e.g., 'papers.csv'): ").strip()
    
    if not output_file:
        print("❌ No output filename specified.")
        return
    
    venue_filter = input("Filter by venue (optional, e.g., 'neurips'): ").strip() or None
    
    cmd = [
        "python", "-m", "cs_bibm.conference_cli", "export",
        "--format", format_choice,
        "--output", output_file
    ]
    
    if venue_filter:
        cmd.extend(["--venue", venue_filter])
    
    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Data exported to: {output_file}")
    except subprocess.CalledProcessError:
        print("❌ Export failed.")

def show_system_info():
    """Show system information"""
    print("\\n ℹ️  System Information")
    
    cmd = ["python", "-m", "cs_bibm.conference_cli", "info"]
    
    import subprocess
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ Unable to get system information.")

def show_cli_help():
    """Show CLI help and switch to advanced mode"""
    print("\\n🔧 Advanced CLI Mode")
    print("Available commands:")
    
    cmd = ["python", "-m", "cs_bibm.conference_cli", "--help"]
    
    import subprocess
    try:
        subprocess.run(cmd)
    except subprocess.CalledProcessError:
        print("❌ Unable to show CLI help.")
    
    print("\\nEnter CLI command (or 'back' to return):")
    while True:
        try:
            user_cmd = input("cs_bibm> ").strip()
            
            if user_cmd.lower() in ['back', 'exit', 'quit']:
                break
            
            if not user_cmd:
                continue
            
            # Parse and execute command
            full_cmd = ["python", "-m", "cs_bibm.conference_cli"] + user_cmd.split()
            subprocess.run(full_cmd)
            
        except KeyboardInterrupt:
            print("\\nReturning to main menu...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()