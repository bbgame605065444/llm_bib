import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
import os
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

class DeepSeekAnalyzer:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", use_reasoning: bool = False):
        self.api_key = api_key
        self.base_url = base_url
        self.use_reasoning = use_reasoning
        
        # Initialize OpenAI client with DeepSeek endpoint (2025 compatible)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.setup_logging()
        
        # Configuration for DeepSeek-V3.1 (2025 Latest)
        self.model = "deepseek-reasoner" if use_reasoning else "deepseek-chat"
        self.max_tokens = 32000 if use_reasoning else 4000  # V3.1 specifications
        self.temperature = 0.1  # Low for consistent analysis
        self.max_retries = 3
        self.retry_delay = 1
        self.batch_size = 10  # Process papers in batches
        self.rate_limit_delay = 0.5  # Delay between API calls
        
        self.logger.info(f"Initialized DeepSeekAnalyzer with model: {self.model}")
        self.logger.info(f"Using reasoning mode: {use_reasoning}")
        self.logger.info(f"Max tokens: {self.max_tokens}")
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_relevance(self, papers: List[Dict], keyword: str) -> List[Dict]:
        """
        Analyze each paper's relevance to the given keyword using DeepSeek API
        
        Args:
            papers: List of paper dictionaries with title and abstract
            keyword: The search keyword to check relevance against
            
        Returns:
            List of papers with added relevance analysis
        """
        self.logger.info(f"Starting relevance analysis for {len(papers)} papers with keyword: '{keyword}'")
        
        analyzed_papers = []
        total_batches = (len(papers) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(papers), self.batch_size):
            batch = papers[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} papers)")
            
            batch_results = self._analyze_batch(batch, keyword)
            analyzed_papers.extend(batch_results)
            
            # Rate limiting
            if batch_num < total_batches:
                time.sleep(self.rate_limit_delay)
        
        self.logger.info(f"Analysis completed. {len(analyzed_papers)} papers analyzed.")
        return analyzed_papers
    
    def _analyze_batch(self, papers: List[Dict], keyword: str) -> List[Dict]:
        """Analyze a batch of papers"""
        analyzed_papers = []
        
        for paper in papers:
            try:
                analysis = self._analyze_single_paper(paper, keyword)
                paper_with_analysis = paper.copy()
                paper_with_analysis.update(analysis)
                analyzed_papers.append(paper_with_analysis)
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                self.logger.error(f"Error analyzing paper '{paper.get('title', 'Unknown')}': {str(e)}")
                # Add paper with error status
                paper_with_error = paper.copy()
                paper_with_error.update({
                    "relevance_score": 0.0,
                    "relevance_explanation": f"Analysis failed: {str(e)}",
                    "is_relevant": False,
                    "analysis_status": "error"
                })
                analyzed_papers.append(paper_with_error)
        
        return analyzed_papers
    
    def _analyze_single_paper(self, paper: Dict, keyword: str) -> Dict:
        """Analyze a single paper's relevance"""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        if not title and not abstract:
            return {
                "relevance_score": 0.0,
                "relevance_explanation": "No title or abstract available",
                "is_relevant": False,
                "analysis_status": "no_content"
            }
        
        # Create the prompt for analysis
        prompt = self._create_analysis_prompt(title, abstract, keyword)
        
        for attempt in range(self.max_retries):
            try:
                # Updated API call for DeepSeek-V3.1 (2025 specifications)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert academic researcher with deep expertise in analyzing the relevance of research papers. You provide precise, objective assessments with detailed explanations."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=min(self.max_tokens, 1000),  # Optimized for relevance analysis
                    stream=False  # Explicit non-streaming for batch processing
                )
                
                analysis_text = response.choices[0].message.content
                return self._parse_analysis_response(analysis_text)
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise e
    
    def _create_analysis_prompt(self, title: str, abstract: str, keyword: str) -> str:
        """Create optimized analysis prompt for DeepSeek-V3.1"""
        return f"""As an expert academic researcher, analyze how relevant this paper is to the research keyword "{keyword}".

**Paper Information:**
Title: {title}

Abstract: {abstract}

**Analysis Task:**
Determine the semantic and thematic relevance of this paper to "{keyword}". Consider:
1. Direct topic alignment
2. Methodological connections  
3. Application domain overlap
4. Conceptual relationships

**Required Output Format (JSON only):**
{{
    "relevance_score": <precise float 0.0-1.0>,
    "is_relevant": <boolean true/false>,
    "explanation": "<concise 1-2 sentence explanation>",
    "key_topics": ["<3-5 most relevant topics from the paper>"],
    "confidence": <assessment confidence 0.0-1.0>
}}

**Scoring Criteria:**
- 0.9-1.0: Direct focus on the keyword topic
- 0.7-0.8: Strong thematic connection with clear relevance
- 0.5-0.6: Moderate relevance with some connection
- 0.3-0.4: Weak connection, tangential relevance
- 0.0-0.2: No meaningful connection

Set is_relevant=true if relevance_score ≥ 0.5. Respond with valid JSON only."""
    
    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse the AI response and extract analysis data"""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                analysis_data = json.loads(json_str)
                
                return {
                    "relevance_score": float(analysis_data.get("relevance_score", 0.0)),
                    "relevance_explanation": analysis_data.get("explanation", ""),
                    "is_relevant": bool(analysis_data.get("is_relevant", False)),
                    "key_topics": analysis_data.get("key_topics", []),
                    "confidence": float(analysis_data.get("confidence", 0.0)),
                    "analysis_status": "success"
                }
            else:
                # Fallback: try to extract relevance from text
                score = self._extract_score_from_text(response_text)
                return {
                    "relevance_score": score,
                    "relevance_explanation": response_text[:200] + "...",
                    "is_relevant": score >= 0.5,
                    "key_topics": [],
                    "confidence": 0.5,
                    "analysis_status": "partial_parse"
                }
                
        except Exception as e:
            self.logger.error(f"Error parsing analysis response: {str(e)}")
            return {
                "relevance_score": 0.0,
                "relevance_explanation": f"Parse error: {str(e)}",
                "is_relevant": False,
                "key_topics": [],
                "confidence": 0.0,
                "analysis_status": "parse_error"
            }
    
    def _extract_score_from_text(self, text: str) -> float:
        """Extract relevance score from text when JSON parsing fails"""
        import re
        
        # Look for score patterns
        score_patterns = [
            r'relevance[_\s]*score[:\s]*([0-9]*\.?[0-9]+)',
            r'score[:\s]*([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)\s*out\s*of\s*[0-9]+',
            r'([0-9]*\.?[0-9]+)/[0-9]+',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize score to 0-1 range if needed
                    if score > 1.0 and score <= 10.0:
                        score = score / 10.0
                    elif score > 10.0:
                        score = score / 100.0
                    return min(1.0, max(0.0, score))
                except ValueError:
                    continue
        
        # If no score found, return neutral
        return 0.5
    
    def generate_analysis_report(self, analyzed_papers: List[Dict], keyword: str) -> Dict:
        """Generate a comprehensive analysis report"""
        if not analyzed_papers:
            return {"error": "No papers to analyze"}
        
        total_papers = len(analyzed_papers)
        relevant_papers = [p for p in analyzed_papers if p.get("is_relevant", False)]
        
        # Calculate statistics
        scores = [p.get("relevance_score", 0.0) for p in analyzed_papers]
        successful_analyses = [p for p in analyzed_papers if p.get("analysis_status") == "success"]
        
        # Topic analysis
        all_topics = []
        for paper in relevant_papers:
            all_topics.extend(paper.get("key_topics", []))
        
        # Count topic frequencies
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Sort topics by frequency
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        report = {
            "analysis_metadata": {
                "keyword": keyword,
                "total_papers": total_papers,
                "analysis_date": datetime.now().isoformat(),
                "analyzer_version": "1.0"
            },
            "relevance_statistics": {
                "relevant_papers": len(relevant_papers),
                "relevance_rate": len(relevant_papers) / total_papers if total_papers > 0 else 0,
                "average_relevance_score": sum(scores) / len(scores) if scores else 0,
                "successful_analyses": len(successful_analyses),
                "success_rate": len(successful_analyses) / total_papers if total_papers > 0 else 0
            },
            "top_topics": top_topics,
            "score_distribution": self._calculate_score_distribution(scores),
            "highly_relevant_papers": [
                {
                    "title": p["title"],
                    "relevance_score": p.get("relevance_score", 0),
                    "explanation": p.get("relevance_explanation", "")
                }
                for p in sorted(relevant_papers, key=lambda x: x.get("relevance_score", 0), reverse=True)[:10]
            ]
        }
        
        return report
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict:
        """Calculate distribution of relevance scores"""
        if not scores:
            return {}
        
        bins = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for score in scores:
            if score < 0.2:
                bins["0.0-0.2"] += 1
            elif score < 0.4:
                bins["0.2-0.4"] += 1
            elif score < 0.6:
                bins["0.4-0.6"] += 1
            elif score < 0.8:
                bins["0.6-0.8"] += 1
            else:
                bins["0.8-1.0"] += 1
        
        return bins
    
    def save_analyzed_results(self, analyzed_papers: List[Dict], filename: str, report: Dict = None) -> str:
        """Save analyzed results to JSON file"""
        output_data = {
            "analysis_report": report or {},
            "analyzed_papers": analyzed_papers
        }
        
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Analyzed results saved to: {filename}")
        return filename

def main():
    """Example usage"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Please set DEEPSEEK_API_KEY environment variable")
        return
    
    # Load papers from JSON file
    papers_file = input("Enter path to papers JSON file: ")
    keyword = input("Enter keyword for relevance analysis: ")
    
    try:
        with open(papers_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            papers = data.get("results", [])
        
        analyzer = DeepSeekAnalyzer(api_key)
        analyzed_papers = analyzer.analyze_relevance(papers[:10], keyword)  # Analyze first 10 for testing
        
        report = analyzer.generate_analysis_report(analyzed_papers, keyword)
        
        # Save results
        output_file = f"analyzed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analyzer.save_analyzed_results(analyzed_papers, output_file, report)
        
        print(f"\nAnalysis completed!")
        print(f"Relevant papers: {report['relevance_statistics']['relevant_papers']}")
        print(f"Relevance rate: {report['relevance_statistics']['relevance_rate']:.2%}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()