import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SERPAPI_KEY = "d24b1467b656421f008ce2eda9201655e9dc493c714e7d978ddcd896628d9d1f"
    DEEPSEEK_API_KEY = "sk-4398982fc85c40059ef5bc6a6767a8d5"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # Updated to latest 2025 endpoint
    
    # DeepSeek API Model Configuration (2025 Latest)
    DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek-V3.1 non-thinking mode
    DEEPSEEK_MODEL_REASONING = "deepseek-reasoner"  # DeepSeek-V3.1 thinking mode
    
    # Scholar search settings
    MAX_RESULTS = 3000
    RESULTS_PER_PAGE = 100
    
    # AI analysis settings (optimized for DeepSeek-V3.1)
    BATCH_SIZE = 20
    RELEVANCE_THRESHOLD = 0.7
    MAX_TOKENS = 4000  # Default max output for deepseek-chat
    TEMPERATURE = 0.1  # Low temperature for consistent analysis
    USE_REASONING_MODE = False  # Set to True to use deepseek-reasoner
    
    # Output settings
    OUTPUT_DIR = "output_v2"
    JSON_OUTPUT_FILE = "scholar_results.json"
    ANALYSIS_OUTPUT_FILE = "relevance_analysis.json"
    
    @classmethod
    def validate_config(cls):
        missing = []
        if not cls.SERPAPI_KEY:
            missing.append('SERPAPI_KEY')
        if not cls.DEEPSEEK_API_KEY:
            missing.append('DEEPSEEK_API_KEY')
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return True