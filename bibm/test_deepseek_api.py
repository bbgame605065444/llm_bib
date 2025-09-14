#!/usr/bin/env python3
"""
Test script to validate the updated DeepSeek API implementation
Tests both deepseek-chat and deepseek-reasoner models with latest 2025 specifications
"""

import json
from ai_analyzer import DeepSeekAnalyzer
from config import Config

def test_deepseek_api():
    """Test the DeepSeek API with sample paper data"""
    
    # Sample paper for testing
    test_paper = {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "year": 2017,
        "venue": "Advances in Neural Information Processing Systems",
        "citation_count": 50000
    }
    
    keyword = "transformer neural networks"
    
    print("=" * 60)
    print("TESTING DEEPSEEK API - 2025 UPDATED IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Test 1: Standard deepseek-chat model
        print("\n1. Testing deepseek-chat (non-thinking mode):")
        print("-" * 40)
        
        analyzer_chat = DeepSeekAnalyzer(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
            use_reasoning=False
        )
        
        print(f"Model: {analyzer_chat.model}")
        print(f"Max tokens: {analyzer_chat.max_tokens}")
        print(f"Temperature: {analyzer_chat.temperature}")
        
        # Analyze the test paper
        analyzed_papers = analyzer_chat.analyze_relevance([test_paper], keyword)
        
        if analyzed_papers:
            result = analyzed_papers[0]
            print(f"\nAnalysis Results:")
            print(f"  Relevance Score: {result.get('relevance_score', 'N/A')}")
            print(f"  Is Relevant: {result.get('is_relevant', 'N/A')}")
            print(f"  Explanation: {result.get('relevance_explanation', 'N/A')}")
            print(f"  Key Topics: {result.get('key_topics', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}")
            print(f"  Status: {result.get('analysis_status', 'N/A')}")
        
        print("\n✅ deepseek-chat test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error testing deepseek-chat: {str(e)}")
    
    try:
        # Test 2: deepseek-reasoner model (thinking mode)
        print("\n\n2. Testing deepseek-reasoner (thinking mode):")
        print("-" * 40)
        
        analyzer_reasoner = DeepSeekAnalyzer(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
            use_reasoning=True
        )
        
        print(f"Model: {analyzer_reasoner.model}")
        print(f"Max tokens: {analyzer_reasoner.max_tokens}")
        print(f"Temperature: {analyzer_reasoner.temperature}")
        
        # Analyze the test paper with reasoning mode
        analyzed_papers = analyzer_reasoner.analyze_relevance([test_paper], keyword)
        
        if analyzed_papers:
            result = analyzed_papers[0]
            print(f"\nAnalysis Results (Reasoning Mode):")
            print(f"  Relevance Score: {result.get('relevance_score', 'N/A')}")
            print(f"  Is Relevant: {result.get('is_relevant', 'N/A')}")
            print(f"  Explanation: {result.get('relevance_explanation', 'N/A')}")
            print(f"  Key Topics: {result.get('key_topics', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}")
            print(f"  Status: {result.get('analysis_status', 'N/A')}")
        
        print("\n✅ deepseek-reasoner test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error testing deepseek-reasoner: {str(e)}")
    
    print("\n" + "=" * 60)
    print("DEEPSEEK API TESTING COMPLETED")
    print("=" * 60)
    
    # Configuration summary
    print(f"\nConfiguration Summary:")
    print(f"  Base URL: {Config.DEEPSEEK_BASE_URL}")
    print(f"  Chat Model: {Config.DEEPSEEK_MODEL}")
    print(f"  Reasoning Model: {Config.DEEPSEEK_MODEL_REASONING}")
    print(f"  Default Max Tokens: {Config.MAX_TOKENS}")
    print(f"  Use Reasoning Mode: {Config.USE_REASONING_MODE}")

def test_api_connectivity():
    """Test basic API connectivity"""
    print("\nTesting API Connectivity...")
    print("-" * 30)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        
        # Simple connectivity test
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API connection successful' if you can read this."}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        print(f"✅ API Response: {response.choices[0].message.content}")
        print(f"✅ Model Used: {response.model}")
        print(f"✅ Connection successful!")
        
    except Exception as e:
        print(f"❌ API Connectivity Error: {str(e)}")

if __name__ == "__main__":
    # Run connectivity test first
    test_api_connectivity()
    
    # Run full API test
    test_deepseek_api()