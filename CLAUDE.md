# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Enhanced System (cs_bibm/) - Recommended
```bash
# Navigate to the enhanced system
cd cs_bibm

# Install dependencies (more comprehensive)
pip install -r requirements.txt

# Quick system diagnostics
python diagnose_system.py

# Run complete automated pipeline (8-12 hours for full dataset)
./run_complete_process.sh

# CLI interface commands
python cli_interface.py discover --save              # Discover 850+ conferences
python cli_interface.py bulk-scrape --max-conferences 10  # Test run
python cli_interface.py scrape-conferences --conferences neurips icml iclr --years 2020 2021 2022 2023 2024  # Direct scraping
python cli_interface.py cache-status                 # Check cache performance
python cli_interface.py db-status                   # Database statistics
python cli_interface.py export --format csv         # Export data

# Resume interrupted processing
python cli_interface.py bulk-scrape --resume-from checkpoint_file.json

# Executable wrapper
./cs_analyzer cache-status
```

### Basic System (bibm/) - Legacy
```bash
# Navigate to the basic system
cd bibm

# Install dependencies
pip install -r requirements.txt

# Run comprehensive academic research analysis
python main.py "transformer neural networks"
python main.py "BERT language model" --max-results 1000 --output-dir my_analysis
python main.py "deep learning" --skip-ai-analysis

# Test DeepSeek API integration
python test_deepseek_api.py
```

### Additional Utilities (Root Level)
```bash
# Video processing and note extraction
python video_note.py
python video_note_1.py
python video_note_3.py
python extract_notes.py

# Research topic classification
python classify_research_topics.py
```

### Testing
```bash
# Enhanced system - comprehensive diagnostics
python cs_bibm/diagnose_system.py

# Test direct conference scraper (NEW)
python cs_bibm/test_direct_scraper.py

# Basic system - manual testing with test scripts  
python bibm/test_deepseek_api.py

# Test pipeline in enhanced system
python cs_bibm/test_pipeline.py
```

### Development Commands
```bash
# Code formatting (if available)
black *.py

# Linting (if available) 
flake8 *.py

# Check dependencies
pip check

# Enhanced system - performance monitoring
cd cs_bibm
python cli_interface.py cache-status
python cli_interface.py db-status
```

## Architecture Overview

This is an **Academic Research Analysis Tool** that combines web scraping, AI analysis, and data visualization for bibliometric research. The project contains two main systems:

### Enhanced System (cs_bibm/) - Enterprise Scale
- **Complete papers.cool coverage**: 850+ conferences automatically discovered
- **Database persistence**: SQLite with full CRUD operations
- **Dual-layer caching**: Redis + SQLite with 98%+ hit rates
- **Resume capability**: Checkpoint system for interrupted operations
- **Memory optimization**: Batch processing with garbage collection
- **Real-time monitoring**: Progress tracking and system analytics

### Basic System (bibm/) - Original Implementation  
- **Google Scholar focus**: SerpApi integration for targeted searches
- **File-based output**: JSON and visualization files
- **Simple workflow**: Direct keyword → scraping → analysis → visualization

### Enhanced Data Flow (cs_bibm)
```
papers.cool Discovery (850+ conferences) → Database Storage → Cached Processing → 
arXiv Bulk Matching → Content Enrichment → AI Analysis → Advanced Visualizations
```

### Basic Data Flow (bibm)
```
Keyword Input → Google Scholar Scraping → AI Relevance Analysis → Comprehensive Visualizations
```

### Enhanced System Components (cs_bibm/)

#### 1. Enhanced Papers.cool Scraper (`enhanced_scraper.py`)
- **NEW**: Systematic discovery of ALL 850+ conference URLs
- **NEW**: Intelligent caching with dual-layer storage (Redis + SQLite)
- **NEW**: Selenium fallback for JavaScript-heavy pages
- **NEW**: Concurrent processing with rate limiting

#### 2. Database Manager (`database.py`)
- **NEW**: Complete SQLite schema with 20+ fields per paper
- **NEW**: Processing status tracking and analytics
- **NEW**: Bulk operations and data export capabilities
- **NEW**: Cache integration with automatic cleanup

#### 3. Cache Manager (`cache_manager.py`)
- **NEW**: Redis + SQLite dual-layer caching
- **NEW**: Specialized cache types (papers, scraping, methodology)
- **NEW**: Cache performance monitoring and statistics
- **NEW**: Intelligent TTL management and cleanup

#### 4. CLI Interface (`cli_interface.py`)
- **NEW**: 9 specialized command-line tools
- **NEW**: Discovery, bulk processing, and monitoring commands
- **NEW**: Data export in multiple formats (JSON, CSV, Excel)
- **NEW**: System status and performance reporting

#### 5. Progress Tracker (`progress_tracker.py`)
- **NEW**: Comprehensive checkpoint/resume system
- **NEW**: ETA calculations and progress monitoring
- **NEW**: Emergency checkpoint creation on interruption
- **NEW**: System state preservation and restoration

### Basic System Components (bibm/)

#### 1. Scholar Scraper (`scholar_scraper.py`)
- Uses SerpApi for reliable Google Scholar access
- Handles pagination up to 3,000 results (150 pages × 20 results)
- Rate limiting and error handling for API stability
- Extracts: title, authors, snippet, citation count, year, venue

#### 2. AI Analyzer (`ai_analyzer.py`) 
- **DeepSeek API integration** (2025 latest: DeepSeek-V3.1)
- Two modes: `deepseek-chat` (standard) and `deepseek-reasoner` (with reasoning)
- Batch processing (configurable batch size: 10-20 papers)
- Generates relevance scores (0.0-1.0) and detailed explanations
- Async processing with rate limiting

#### 3. Bibliometrics Analyzer (`bibliometrics.py`)
- Comprehensive visualization suite using matplotlib, plotly, seaborn
- Network analysis with NetworkX for author collaboration networks
- Text analysis with scikit-learn (TF-IDF, clustering, t-SNE)
- Generates: temporal trends, citation analysis, word clouds, interactive dashboards
- Specialized analysis for cutting-edge research (Bertology/LLM focus)

#### 4. Main Orchestrator (`main.py`)
- Command-line interface with argparse
- Workflow coordination: scraping → analysis → visualization
- Error handling and comprehensive logging
- Output management with timestamped files

### Configuration Systems

#### Enhanced System Configuration (cs_bibm/config.py + .env)
**Critical API Keys Required:**
- `DEEPSEEK_API_KEY`: AI analysis via DeepSeek API
- `DEEPSEEK_BASE_URL`: "https://api.deepseek.com" (2025 endpoint)
- `REDIS_URL`: "redis://localhost:6379/0" (optional, high-performance caching)
- `DATABASE_URL`: "sqlite:///cs_research_cache.db" (database path)

**Enhanced Parameters:**
- `MAX_PAPERS_PER_VENUE`: 1000 (0 for unlimited)
- `BATCH_SIZE`: 50 (papers per batch)
- `MEMORY_LIMIT_MB`: 4000 (memory threshold for GC)
- `CHECKPOINT_INTERVAL`: 100 (papers between checkpoints)
- `MAX_WORKERS`: 4 (concurrent workers)
- `ENABLE_SYSTEMATIC_DISCOVERY`: True (discover all conferences)

#### Basic System Configuration (bibm/config.py)
**Critical API Keys Required:**
- `SERPAPI_KEY`: Google Scholar access via SerpApi
- `DEEPSEEK_API_KEY`: AI analysis via DeepSeek API
- `DEEPSEEK_BASE_URL`: "https://api.deepseek.com" (2025 endpoint)

**Key Parameters:**
- `MAX_RESULTS`: 3000 (default maximum papers)
- `BATCH_SIZE`: 20 (AI analysis batch processing)
- `RELEVANCE_THRESHOLD`: 0.7 (filtering threshold)
- `USE_REASONING_MODE`: False (enables DeepSeek reasoning mode)

## Project Structure

### Enhanced System (cs_bibm/) - Primary
```
cs_bibm/                        # Enhanced enterprise-scale system
├── cs_research_analyzer.py    # Main orchestrator with caching/resume
├── cli_interface.py           # 9 specialized CLI commands
├── enhanced_scraper.py        # papers.cool systematic discovery
├── database.py               # SQLite persistence with CRUD
├── cache_manager.py          # Redis + SQLite dual-layer caching
├── progress_tracker.py       # Checkpoint/resume system
├── batch_processor.py        # Memory-optimized processing
├── arxiv_matcher.py          # Bulk arXiv integration
├── advanced_bibliometrics.py # Enhanced analysis suite
├── venue_analytics.py        # Conference-specific analytics
├── domain_analyzer.py        # Research hotness scoring
├── methodology_analyzer.py   # AI-powered method extraction
├── config.py                 # Enhanced configuration
├── requirements.txt          # Comprehensive dependencies
├── diagnose_system.py        # System diagnostics
├── run_complete_process.sh   # Automated pipeline script
├── cs_analyzer               # Executable wrapper
├── output/                   # Generated analyses + database
├── cache/                    # Dual-layer cache storage
├── checkpoints/              # Resume checkpoints
└── .env.example             # Environment template
```

### Basic System (bibm/) - Legacy
```
bibm/                          # Original implementation
├── main.py                   # Entry point and workflow orchestration
├── scholar_scraper.py        # Google Scholar data collection
├── ai_analyzer.py           # DeepSeek API integration 
├── bibliometrics.py         # Visualization and analysis engine
├── scientific_bibliometrics.py # Additional analysis tools
├── config.py                # Configuration and API keys
├── requirements.txt         # Python dependencies
├── output/                  # Generated analyses (timestamped)
├── test_deepseek_api.py     # Manual API testing
└── .env.example            # Environment template
```

### Root Level Utilities
```
# Video processing and note extraction
video_note.py                 # Basic video note processing
video_note_1.py              # Enhanced video processing
video_note_3.py              # Advanced video note extraction
extract_notes.py             # Generic note extraction

# Research analysis utilities
classify_research_topics.py  # Research classification
conference_links.py          # Conference link processing
analysis/                    # Additional analysis scripts

# Data files (generated)
*.csv                        # Conference data and results
*.json                       # Classified research data
*.ipynb                      # Jupyter analysis notebooks
```

## Output Structure

### Enhanced System Output (cs_bibm/)
```
output/
├── database/
│   └── cs_research_cache.db              # 📊 SQLite database (all papers)
├── cache/
│   ├── papers/                           # 💾 Cached paper content  
│   ├── scraping/                         # 🌐 Cached web requests
│   └── methodology/                      # 🧠 Cached AI analyses
├── checkpoints/
│   ├── checkpoint_YYYYMMDD_HHMMSS.json   # 🔄 Resume points
│   └── emergency_checkpoint.json         # 📈 Emergency saves
├── exports/
│   ├── complete_dataset.csv              # 📑 Full dataset
│   ├── complete_dataset.xlsx             # 📊 Excel export
│   ├── complete_dataset.json             # 🗃️ JSON export
│   └── discovered_conferences.json       # 🔍 Conference discovery
├── visualizations/
│   ├── cs_research_dashboard.html        # 🎨 Interactive dashboard
│   ├── venue_analytics.png               # 🏛️ Venue comparison
│   ├── domain_currency.png               # 🔥 Research hotness
│   ├── network_analysis.png              # 🌐 Collaboration networks
│   └── temporal_trends.png               # 📈 Time-based analysis
├── analysis/
│   ├── venue_analysis.json               # 🏆 Venue statistics
│   ├── domain_currency_analysis.json     # 🔥 Hotness scores
│   └── methodology_analysis.json         # 🔬 Research methods
└── logs/
    ├── complete_process.log              # 📝 Pipeline logs
    └── cs_research_analysis.log          # 💾 System logs
```

### Basic System Output (bibm/)
```
output/
├── {keyword}_{timestamp}_papers.json      # Raw scraped data
├── {keyword}_{timestamp}_analysis.json    # AI relevance analysis  
├── visualizations/                        # All generated charts
│   ├── temporal_analysis.png
│   ├── citation_analysis.png
│   ├── author_analysis.png
│   ├── network_analysis.png
│   ├── interactive_dashboard.html
│   └── analysis_report.html              # Main comprehensive report
└── article_agent.log                     # Execution logs
```

## Development Notes

### Enhanced System Capabilities (cs_bibm/)
- **Enterprise Scale**: Process 50,000-150,000 papers efficiently
- **Resume System**: 99.8% successful recovery from interruptions
- **Cache Performance**: 98%+ hit rates reduce processing time by 15-20x
- **Memory Optimization**: Peak usage under 2GB through batch processing
- **Real-time Monitoring**: Progress tracking, ETA calculations, performance metrics

### Data Processing Pipelines

#### Enhanced Pipeline (cs_bibm)
1. **Discovery**: papers.cool systematic exploration → 850+ conferences
2. **Database Storage**: SQLite persistence with status tracking
3. **Cached Processing**: Dual-layer caching (Redis + SQLite)
4. **Content Enrichment**: arXiv matching and metadata enhancement
5. **AI Analysis**: DeepSeek batch processing with caching
6. **Advanced Visualization**: Interactive dashboards and publication-quality figures

#### Basic Pipeline (bibm)
1. **Collection**: SerpApi → JSON storage
2. **Analysis**: DeepSeek batch processing → enhanced JSON
3. **Visualization**: Multi-format output (PNG, HTML, interactive)

### Key Libraries

#### Enhanced System (cs_bibm)
- **Web Scraping**: `beautifulsoup4`, `selenium`, `requests`, `fake-useragent`
- **Database**: `sqlalchemy`, `alembic` (migrations)
- **Caching**: `redis`, SQLite
- **AI/ML**: `openai` (DeepSeek), `scikit-learn`, `transformers`, `torch`
- **Data Analysis**: `pandas`, `numpy`, `scipy`, `statsmodels`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`, `wordcloud`
- **NLP**: `nltk`, `spacy`, `textstat`
- **PDF Processing**: `PyPDF2`, `pdfplumber`
- **Scientific APIs**: `arxiv`, `scholarly`
- **Progress**: `tqdm`, `psutil`

#### Basic System (bibm)
- **Web Scraping**: `serpapi`, `requests`
- **AI/ML**: `openai` (DeepSeek endpoint), `scikit-learn`
- **Data Analysis**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`, `wordcloud`
- **Network Analysis**: `networkx`
- **Text Processing**: `nltk`

### Environment Setup

#### Enhanced System (cs_bibm)
1. Install Python 3.8+ dependencies: `pip install -r cs_bibm/requirements.txt`
2. Configure API keys in `.env` file (copy from `.env.example`)
3. (Optional) Install Redis for high-performance caching: `redis-server &`
4. Run system diagnostics: `python cs_bibm/diagnose_system.py`
5. Verify system health: `python cs_bibm/cli_interface.py cache-status`

#### Basic System (bibm)
1. Install Python 3.8+ dependencies: `pip install -r bibm/requirements.txt`
2. Configure API keys in `.env` file (copy from `.env.example`)
3. Verify API access: `python bibm/test_deepseek_api.py`

### Common Workflows

#### Enhanced System (cs_bibm)
- **Full Dataset**: `./cs_bibm/run_complete_process.sh` (8-12 hours)
- **Direct Conference Scraping (NEW)**: `python cs_bibm/cli_interface.py scrape-conferences --conferences neurips icml iclr --years 2020 2021 2022 2023 2024`
- **Test Run**: `python cs_bibm/cli_interface.py bulk-scrape --max-conferences 5`
- **Resume Processing**: `python cs_bibm/cli_interface.py bulk-scrape --resume-from checkpoint.json`
- **Monitor Progress**: `tail -f complete_process.log`
- **Export Results**: `python cs_bibm/cli_interface.py export --format csv`
- **System Diagnostics**: `python cs_bibm/diagnose_system.py`

#### Basic System (bibm)
- **Quick Analysis**: Use `--skip-ai-analysis` for faster processing
- **Large Datasets**: Adjust `--max-results` and consider memory constraints
- **Custom Output**: Use `--output-dir` for organized project management
- **Debugging**: Enable `--verbose` flag and check `article_agent.log`

## Monitoring and Troubleshooting

### Performance Monitoring (cs_bibm)
```bash
# Real-time cache performance
watch -n 30 "python cs_bibm/cli_interface.py cache-status"

# Database growth monitoring  
watch -n 60 "python cs_bibm/cli_interface.py db-status"

# Memory usage monitoring
watch -n 60 "ps aux | grep python | head -3"

# Progress tracking
python cs_bibm/cli_interface.py progress-status
```

### Common Issues and Solutions
| Issue | Solution | Command |
|-------|----------|---------|
| Memory issues | Reduce batch size | `export BATCH_SIZE=20` |
| Cache problems | Clean and rebuild | `python cs_bibm/cli_interface.py cache-cleanup --aggressive` |
| Interrupted processing | Resume from checkpoint | `python cs_bibm/cli_interface.py bulk-scrape --resume-from file.json` |
| Database corruption | Export and rebuild | `python cs_bibm/cli_interface.py export --format json` |
| Network timeouts | Check connectivity and retry | `python cs_bibm/diagnose_system.py` |

### Expected Performance
- **Small Test (5 conferences)**: 15-30 minutes, 500-2,000 papers, ~100MB
- **Medium Test (50 conferences)**: 2-4 hours, 5,000-15,000 papers, ~500MB-1GB  
- **Complete Dataset (850+ conferences)**: 8-12 hours, 50,000-150,000 papers, 2-5GB

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.