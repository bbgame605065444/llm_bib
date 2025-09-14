# Article Agent - Comprehensive Academic Research Analysis Tool

A powerful tool for academic research analysis that combines Google Scholar scraping, AI-powered relevance analysis, and comprehensive bibliometric visualizations.

## Features

🔍 **Google Scholar Integration**: Scrape up to 3,000 research papers using SerpApi  
🤖 **AI-Powered Analysis**: Use DeepSeek API to analyze paper relevance to your keyword  
📊 **Rich Visualizations**: Generate comprehensive bibliometric analysis including:
- Temporal trends and publication patterns
- Citation analysis and impact metrics  
- Author collaboration networks
- Venue and domain distribution
- Text analysis and word clouds
- Cutting-edge research trends (Bertology/LLM focus)
- Interactive dashboards

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd bibm

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
SERPAPI_KEY=your_serpapi_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

**Get API Keys:**
- SerpApi: https://serpapi.com/ (Free tier: 100 searches/month)
- DeepSeek: https://platform.deepseek.com/ (Competitive pricing)

### 3. Run Analysis

```bash
# Basic usage
python main.py "transformer neural networks"

# Advanced usage with options
python main.py "BERT language model" --max-results 1000 --output-dir my_analysis

# Skip AI analysis for faster processing
python main.py "deep learning" --skip-ai-analysis
```

## Usage Examples

### Command Line Options

```bash
python main.py <keyword> [options]

Options:
  --max-results INT     Maximum papers to collect (default: 3000)
  --skip-ai-analysis    Skip AI relevance analysis (faster)
  --output-dir DIR      Output directory (default: output)
  --verbose, -v         Enable verbose logging
```

### Example Commands

```bash
# Comprehensive analysis of transformer research
python main.py "transformer attention mechanism" --max-results 2000

# Quick analysis without AI (faster)
python main.py "neural machine translation" --skip-ai-analysis

# Research on specific domain
python main.py "BERT fine-tuning" --max-results 500 --output-dir bert_analysis
```

## Output Structure

The tool generates a comprehensive analysis package:

```
output/
├── keyword_YYYYMMDD_HHMMSS_papers.json      # Raw scraped papers
├── keyword_YYYYMMDD_HHMMSS_analysis.json    # AI relevance analysis
├── visualizations/                           # All generated visualizations
│   ├── keyword_temporal_analysis.png
│   ├── keyword_citation_analysis.png
│   ├── keyword_author_analysis.png
│   ├── keyword_venue_analysis.png
│   ├── keyword_text_analysis.png
│   ├── keyword_relevance_analysis.png
│   ├── keyword_network_analysis.png
│   ├── keyword_cutting_edge_analysis.png
│   ├── keyword_interactive_dashboard.html
│   └── keyword_analysis_report.html         # Main comprehensive report
└── article_agent.log                        # Execution log
```

## Architecture

### Core Components

1. **Scholar Scraper** (`scholar_scraper.py`)
   - Uses SerpApi for reliable Google Scholar access
   - Handles pagination and rate limiting
   - Extracts comprehensive paper metadata

2. **AI Analyzer** (`ai_analyzer.py`)  
   - DeepSeek API integration for relevance analysis
   - Batch processing for efficiency
   - Generates relevance scores and explanations

3. **Bibliometrics Analyzer** (`bibliometrics.py`)
   - Comprehensive visualization suite
   - Network analysis capabilities
   - Interactive dashboards with Plotly

4. **Main Application** (`main.py`)
   - Orchestrates the complete workflow
   - Command-line interface
   - Error handling and logging

### Data Flow

```
Keyword Input → Google Scholar Search → Paper Collection → AI Analysis → Visualizations
```

## Advanced Features

### Bibliometric Analysis Includes:

- **Temporal Analysis**: Publication trends, cumulative growth
- **Citation Metrics**: Distribution, h-index, impact over time  
- **Author Analysis**: Top researchers, collaboration networks
- **Venue Analysis**: Journal distribution, domain classification
- **Text Mining**: Word clouds, topic evolution
- **Network Analysis**: Co-authorship and topic networks
- **Cutting-Edge Focus**: Recent trends, LLM/Bertology analysis
- **Interactive Dashboards**: Plotly-based exploration tools

### AI Relevance Analysis:

- Semantic understanding of paper content
- Relevance scoring (0.0 to 1.0)
- Detailed explanations for each paper
- Topic extraction and classification
- Confidence scoring

## Configuration

### Environment Variables

```bash
# Required
SERPAPI_KEY=your_serpapi_key
DEEPSEEK_API_KEY=your_deepseek_key

# Optional
MAX_RESULTS=3000           # Maximum papers to collect
BATCH_SIZE=20              # AI analysis batch size  
RELEVANCE_THRESHOLD=0.7    # Relevance threshold for filtering
```

### Config File Settings

Edit `config.py` to adjust:
- API endpoints and models
- Processing parameters
- Output configurations
- Analysis thresholds

## API Usage

Each component can be used independently:

```python
from scholar_scraper import ScholarScraper
from ai_analyzer import DeepSeekAnalyzer  
from bibliometrics import BibliometricsAnalyzer

# Scrape papers
scraper = ScholarScraper(api_key)
papers = scraper.search_scholar("keyword", 1000)

# Analyze relevance  
analyzer = DeepSeekAnalyzer(api_key, base_url)
analyzed = analyzer.analyze_relevance(papers, "keyword")

# Generate visualizations
visualizer = BibliometricsAnalyzer()
viz_paths = visualizer.generate_comprehensive_analysis(papers, "keyword", analyzed)
```

## Requirements

- Python 3.8+
- Internet connection for API access
- SerpApi account (free tier available)
- DeepSeek API account (competitive pricing)

## Limitations

- SerpApi rate limits apply (100 requests/month free tier)
- DeepSeek API costs apply for large-scale analysis
- Google Scholar may have anti-scraping measures
- Analysis quality depends on paper abstracts availability

## Troubleshooting

### Common Issues:

1. **API Key Errors**: Ensure keys are correctly set in `.env`
2. **Rate Limiting**: Reduce batch size or add delays
3. **Memory Issues**: Process smaller datasets for large analyses  
4. **Visualization Errors**: Ensure all dependencies are installed

### Logs and Debugging:

- Check `article_agent.log` for detailed execution logs
- Use `--verbose` flag for more detailed output
- Each component has independent error handling

## Contributing

This tool is designed for academic research and educational purposes. Feel free to extend functionality:

- Add new visualization types
- Integrate additional APIs
- Improve text analysis capabilities
- Enhance network analysis features

## License

Academic and research use. Please respect API terms of service and rate limits.

---

**Happy Researching! 📚🔬**