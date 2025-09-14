# 🚀 Quick Start Guide - Get All Conference Titles

This guide shows how to quickly scrape conference paper titles and save them to the database.

## ⚡ One-Line Commands

### Scrape Single Conference
```bash
# Scrape NeurIPS 2023 titles (max 100 papers)
python -c "from conference_scraper import ConferenceScraper; scraper = ConferenceScraper(); results = scraper.scrape_conference('NeurIPS', [2023], 100); print(f'✅ Scraped {results[\"total_papers\"]} papers')"
```

### Scrape Multiple Conferences
```bash
# Scrape top AI conferences for 2023
python -c "
from conference_scraper import ConferenceScraper
scraper = ConferenceScraper(enable_arxiv=False)  # Faster without arXiv
conferences = ['NeurIPS', 'ICML', 'ICLR', 'AAAI']
total = 0
for conf in conferences:
    results = scraper.scrape_conference(conf, [2023], 50)  # 50 papers each
    papers = results['total_papers']
    print(f'✅ {conf}: {papers} papers')
    total += papers
print(f'🎉 Total: {total} papers saved to database')
"
```

## 📋 Step-by-Step Commands

### 1. Basic Setup Test
```bash
cd cs_bibm

# Test system
python -c "from conference_registry import ConferenceRegistry; reg = ConferenceRegistry(); print(f'✅ {len(reg.conferences)} conferences available')"
```

### 2. Scrape Conference Titles
```bash
# Method 1: Using Python directly
python -c "
from conference_scraper import ConferenceScraper

# Initialize scraper
scraper = ConferenceScraper(
    db_path='conference_papers.db',  # Database file
    enable_arxiv=False,              # Skip arXiv for speed
    rate_limit_delay=1.0             # 1 second between requests
)

# Scrape NeurIPS 2023 - get all titles
print('🔄 Scraping NeurIPS 2023...')
results = scraper.scrape_conference('NeurIPS', [2023], max_papers=0)  # 0 = unlimited

print(f'✅ Results:')
print(f'  Papers found: {results[\"total_papers\"]}')
print(f'  Years: {results[\"years_scraped\"]}')
print(f'  Errors: {len(results[\"errors\"])}')
"

# Method 2: Using CLI
python run_scraper.py  # Interactive menu
# Or direct CLI:
python -m conference_cli scrape --conferences neurips --years 2023
```

### 3. Check Database
```bash
# View database contents
python -c "
from database_manager import DatabaseManager

db = DatabaseManager('conference_papers.db')
stats = db.get_database_stats()

print('📊 Database Statistics:')
print(f'  Total papers: {stats[\"total_papers\"]:,}')
print(f'  Venues: {stats[\"unique_venues\"]}')
print(f'  Years: {stats[\"years_covered\"]}')
print(f'  Size: {stats[\"database_size_mb\"]:.1f} MB')
"
```

### 4. Export Titles
```bash
# Export to CSV
python -c "
from database_manager import DatabaseManager

db = DatabaseManager('conference_papers.db')
db.export_to_csv('all_paper_titles.csv')
print('✅ Titles exported to all_paper_titles.csv')
"
```

## 🎯 Production Commands

### Scrape All Top Conferences (2020-2024)
```bash
# WARNING: This will take several hours and scrape thousands of papers
python -c "
from conference_scraper import ConferenceScraper

scraper = ConferenceScraper(
    db_path='full_conference_database.db',
    enable_arxiv=True,  # Include arXiv matching
    rate_limit_delay=1.0,
    batch_size=100
)

# Scrape all conferences for recent years
print('🚀 Starting full database scraping...')
print('⏰ This will take 8-12 hours for complete dataset')
print('📊 Expected: 50,000-150,000 papers')

results = scraper.scrape_all_conferences(
    years=[2020, 2021, 2022, 2023, 2024],
    max_papers_per_venue=0,  # Unlimited
    conference_types=None    # All types
)

print(f'🎉 Scraping completed!')
print(f'  Total papers: {results[\"statistics\"][\"papers_found\"]:,}')
print(f'  Conferences: {results[\"statistics\"][\"conferences_scraped\"]}')
"
```

### Scrape Specific Conference Types
```bash
# AI/ML conferences only
python -c "
from conference_scraper import ConferenceScraper

scraper = ConferenceScraper()
results = scraper.scrape_all_conferences(
    years=[2023, 2024],
    conference_types=['AI/ML']
)
print(f'✅ AI/ML papers: {results[\"statistics\"][\"papers_found\"]:,}')
"
```

## 📤 Export and Download Commands

### Export Database to CSV
```bash
python -c "
from database_manager import DatabaseManager
db = DatabaseManager('conference_papers.db')
db.export_to_csv('conference_papers.csv')
print('✅ Database exported to conference_papers.csv')
"
```

### Download arXiv PDFs
```bash
# First check how many papers have arXiv links
python -c "
import sqlite3
conn = sqlite3.connect('conference_papers.db')
cursor = conn.execute('SELECT COUNT(*) FROM papers WHERE arxiv_id IS NOT NULL')
count = cursor.fetchone()[0]
print(f'📄 Papers with arXiv PDFs available: {count:,}')
"

# Download all available PDFs
python download_arxiv_papers.py --output-dir ./downloaded_papers/ --max-papers 100
```

## 🔧 Troubleshooting Commands

### Test System Health
```bash
python test_full_system.py
```

### Check Individual Components
```bash
# Test DBLP scraper
python -c "
from dblp_scraper import DBLPScraper
scraper = DBLPScraper()
papers = scraper.scrape_conference('conf/nips', 2023, 3)
print(f'DBLP test: {len(papers)} papers found')
"

# Test database
python -c "
from database_manager import DatabaseManager
db = DatabaseManager('test.db')
print('Database test: OK')
"

# Test registry
python -c "
from conference_registry import ConferenceRegistry
reg = ConferenceRegistry()
print(f'Registry test: {len(reg.conferences)} conferences loaded')
"
```

## ⚙️ Configuration Options

### Environment Variables (.env file)
```bash
# Copy template
cp .env.example .env

# Edit .env file with your preferences:
RATE_LIMIT_DELAY=1.0
BATCH_SIZE=50
ENABLE_ARXIV_MATCHING=true
DATABASE_PATH=conference_papers.db
```

### Custom Scraping Parameters
```python
from conference_scraper import ConferenceScraper

# Custom configuration
scraper = ConferenceScraper(
    db_path='my_papers.db',           # Custom database
    enable_arxiv=True,                # Enable arXiv matching
    rate_limit_delay=0.5,             # Faster requests (be careful)
    batch_size=100                    # Larger batches
)

# Scrape with custom limits
results = scraper.scrape_conference(
    'NeurIPS',                        # Conference
    [2020, 2021, 2022, 2023, 2024],  # Multiple years
    max_papers=1000                   # Limit per year
)
```

## 📊 Expected Results

| Conference | Years | Expected Papers | Time Required |
|------------|-------|-----------------|---------------|
| NeurIPS | 2023 | ~2,000 | 15-20 minutes |
| ICML | 2023 | ~1,200 | 10-15 minutes |
| CVPR | 2023 | ~2,300 | 20-25 minutes |
| All AI/ML | 2023 | ~10,000 | 2-3 hours |
| All Conferences | 2020-2024 | ~150,000 | 8-12 hours |

## 🎉 Success Indicators

After running commands, you should see:
- ✅ Database file created (e.g., `conference_papers.db`)
- ✅ Papers count increasing in logs
- ✅ CSV exports with paper data
- ✅ No error messages in output
- ✅ Database size growing (MB/GB depending on scope)