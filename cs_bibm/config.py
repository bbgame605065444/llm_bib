"""
Configuration for Conference Scraper System
Environment variables and system settings.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the conference scraper"""
    
    # Database settings
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'cs_bibm/data/conference_papers.db')
    
    # API Configuration
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    
    # Redis cache (optional)
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    ENABLE_REDIS = os.getenv('ENABLE_REDIS', 'false').lower() == 'true'
    
    # Scraping settings
    RATE_LIMIT_DELAY = float(os.getenv('RATE_LIMIT_DELAY', '1.0'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))
    MAX_PAPERS_PER_VENUE = int(os.getenv('MAX_PAPERS_PER_VENUE', '0'))  # 0 = unlimited
    
    # arXiv settings
    ENABLE_ARXIV_MATCHING = os.getenv('ENABLE_ARXIV_MATCHING', 'true').lower() == 'true'
    ARXIV_SIMILARITY_THRESHOLD = float(os.getenv('ARXIV_SIMILARITY_THRESHOLD', '0.85'))
    
    # Memory and performance
    MEMORY_LIMIT_MB = int(os.getenv('MEMORY_LIMIT_MB', '4000'))
    CHECKPOINT_INTERVAL = int(os.getenv('CHECKPOINT_INTERVAL', '100'))  # Papers between checkpoints
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    
    # Output directories
    OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'cs_bibm/output'))
    LOG_DIR = Path(os.getenv('LOG_DIR', 'cs_bibm/logs'))
    CACHE_DIR = Path(os.getenv('CACHE_DIR', 'cs_bibm/cache'))
    CHECKPOINT_DIR = Path(os.getenv('CHECKPOINT_DIR', 'cs_bibm/checkpoints'))
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # DBLP settings
    DBLP_API_BASE = 'https://dblp.org'
    DBLP_SEARCH_API = 'https://dblp.org/search/publ/api'
    
    # arXiv settings
    ARXIV_API_BASE = 'http://export.arxiv.org/api/query'
    
    # User agent for web requests
    USER_AGENT = 'CS-BIBM Conference Scraper (Academic Research Tool)'
    
    # Cache settings
    CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '86400'))  # 24 hours
    ENABLE_FILE_CACHE = os.getenv('ENABLE_FILE_CACHE', 'true').lower() == 'true'
    
    # Export settings
    DEFAULT_EXPORT_FORMAT = os.getenv('DEFAULT_EXPORT_FORMAT', 'csv')
    
    # Conference scraping priorities
    HIGH_PRIORITY_CONFERENCES = [
        'NeurIPS', 'ICML', 'ICLR', 'AAAI', 'CVPR', 'ICCV', 'ECCV', 
        'ACL', 'EMNLP', 'IJCAI'
    ]
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        issues = []
        
        # Check required directories
        required_dirs = [cls.OUTPUT_DIR, cls.LOG_DIR, cls.CACHE_DIR, cls.CHECKPOINT_DIR]
        for dir_path in required_dirs:
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {dir_path}: {e}")
        
        # Check API keys (optional but recommended)
        if not cls.DEEPSEEK_API_KEY:
            issues.append("DEEPSEEK_API_KEY not set (required for AI analysis)")
        
        # Check numeric settings
        if cls.RATE_LIMIT_DELAY < 0:
            issues.append("RATE_LIMIT_DELAY must be non-negative")
        
        if cls.BATCH_SIZE <= 0:
            issues.append("BATCH_SIZE must be positive")
        
        if cls.MEMORY_LIMIT_MB <= 0:
            issues.append("MEMORY_LIMIT_MB must be positive")
        
        # Check Redis connection if enabled
        if cls.ENABLE_REDIS:
            try:
                import redis
                r = redis.from_url(cls.REDIS_URL)
                r.ping()
            except Exception as e:
                issues.append(f"Redis connection failed: {e}")
        
        if issues:
            print("⚠️  Configuration Issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        return True
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database connection URL"""
        db_path = Path(cls.DATABASE_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.absolute()}"
    
    @classmethod
    def get_cache_config(cls) -> dict:
        """Get cache configuration"""
        return {
            'enable_redis': cls.ENABLE_REDIS,
            'redis_url': cls.REDIS_URL,
            'enable_file_cache': cls.ENABLE_FILE_CACHE,
            'cache_dir': cls.CACHE_DIR,
            'ttl_seconds': cls.CACHE_TTL_SECONDS
        }
    
    @classmethod
    def get_scraping_config(cls) -> dict:
        """Get scraping configuration"""
        return {
            'rate_limit_delay': cls.RATE_LIMIT_DELAY,
            'batch_size': cls.BATCH_SIZE,
            'max_papers_per_venue': cls.MAX_PAPERS_PER_VENUE,
            'enable_arxiv': cls.ENABLE_ARXIV_MATCHING,
            'arxiv_similarity_threshold': cls.ARXIV_SIMILARITY_THRESHOLD,
            'max_workers': cls.MAX_WORKERS,
            'memory_limit_mb': cls.MEMORY_LIMIT_MB,
            'checkpoint_interval': cls.CHECKPOINT_INTERVAL
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("🔧 SYSTEM CONFIGURATION")
        print("=" * 50)
        
        print(f"Database: {cls.DATABASE_PATH}")
        print(f"Output directory: {cls.OUTPUT_DIR}")
        print(f"Log directory: {cls.LOG_DIR}")
        print(f"Cache directory: {cls.CACHE_DIR}")
        
        print(f"\\n📡 API SETTINGS")
        print(f"DeepSeek API: {'✅ Configured' if cls.DEEPSEEK_API_KEY else '❌ Not configured'}")
        print(f"Redis cache: {'✅ Enabled' if cls.ENABLE_REDIS else '❌ Disabled'}")
        print(f"arXiv matching: {'✅ Enabled' if cls.ENABLE_ARXIV_MATCHING else '❌ Disabled'}")
        
        print(f"\\n⚙️  PERFORMANCE SETTINGS")
        print(f"Rate limit delay: {cls.RATE_LIMIT_DELAY}s")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Max workers: {cls.MAX_WORKERS}")
        print(f"Memory limit: {cls.MEMORY_LIMIT_MB} MB")
        print(f"Checkpoint interval: {cls.CHECKPOINT_INTERVAL} papers")
        
        print(f"\\n🎯 SCRAPING PRIORITIES")
        print(f"High priority conferences: {', '.join(cls.HIGH_PRIORITY_CONFERENCES)}")

def main():
    """Demo configuration usage"""
    Config.print_config()
    
    print(f"\\n🔍 VALIDATION")
    is_valid = Config.validate_config()
    print(f"Configuration valid: {'✅ Yes' if is_valid else '❌ No'}")

if __name__ == "__main__":
    main()