import csv
import json
import ollama
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed


def classify_research_topic(title: str) -> Dict[str, any]:
    """Classify a research title using the local Ollama qwen3:32b model"""
    # Create a prompt that asks the model to classify the research topic
    prompt = f"""Analyze the following research paper title and determine which of these research topics it belongs to: time-series, multimodal, RAG, KG.

If the paper belongs to one of these topics, respond ONLY with the topic name (time-series, multimodal, RAG, or KG).
If it doesn't belong to any of these topics, respond with "none".

Title: {title}

Topic:"""
    
    try:
        # Use the local Ollama qwen3:32b model
        response = ollama.generate(model='qwen3:32b', prompt=prompt, options={'temperature': 0.1})
        topic = response['response'].strip().lower()
        
        # Validate the response
        valid_topics = ['time-series', 'multimodal', 'rag', 'kg', 'none']
        if topic not in valid_topics:
            topic = 'none'
            
        return {
            'title': title,
            'topic': topic,
            'raw_response': response['response'].strip()
        }
    except Exception as e:
        print(f"Error classifying title '{title}': {e}")
        return {
            'title': title,
            'topic': 'error',
            'error': str(e)
        }


def process_csv_file(csv_file_path: str, max_workers=5) -> List[Dict[str, any]]:
    """Process the CSV file and classify each research title using concurrent processing"""
    results = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        titles = [(i, row['title'], row['url']) for i, row in enumerate(reader)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(classify_research_topic, title): (i, title, url) for i, title, url in titles}
        
        # Counter for saving results every 10 entries
        counter = 0
        
        for future in as_completed(future_to_index):
            i, title, url = future_to_index[future]
            try:
                result = future.result()
                result['url'] = url
                results.append((i, result))
                print(f"Processed {i+1}: {title}")
                
                # Increment counter
                counter += 1
                
                # Save results every 10 entries
                if counter % 10 == 0:
                    # Sort results by index to maintain order
                    results.sort(key=lambda x: x[0])
                    current_results = [result for _, result in results]
                    
                    # Append to file
                    with open('classified_research_topics_progress.json', 'a', encoding='utf-8') as f:
                        # For first batch, write opening bracket
                        if counter == 10:
                            f.write('[\n')
                        
                        # Write results
                        for idx, res in enumerate(current_results[-10:], start=len(current_results)-9):
                            json.dump(res, f, ensure_ascii=False)
                            if idx < len(current_results):  # Not the last item
                                f.write(',\n')
                            else:  # Last item
                                f.write('\n')
                        
                        # For last batch, write closing bracket
                        # This will be handled after the loop
                
            except Exception as e:
                print(f"Error processing title '{title}': {e}")
                results.append((i, {
                    'title': title,
                    'topic': 'error',
                    'error': str(e),
                    'url': url
                }))
                counter += 1
    
    # Sort results by index to maintain order
    results.sort(key=lambda x: x[0])
    
    # Handle the final batch and close the JSON array
    final_results = [result for _, result in results]
    remaining = len(final_results) % 10
    
    if remaining > 0 and len(final_results) > 0:
        with open('classified_research_topics_progress.json', 'a', encoding='utf-8') as f:
            for idx, res in enumerate(final_results[-remaining:], start=len(final_results)-remaining+1):
                json.dump(res, f, ensure_ascii=False)
                if idx < len(final_results):  # Not the last item
                    f.write(',\n')
                else:  # Last item
                    f.write('\n')
    
    # Close the JSON array
    with open('classified_research_topics_progress.json', 'a', encoding='utf-8') as f:
        f.write(']')
    
    return [result for _, result in results]


def filter_by_topic(results: List[Dict[str, any]], topic: str) -> List[Dict[str, any]]:
    """Filter results by a specific topic"""
    return [result for result in results if result['topic'] == topic]


def main():
    csv_file_path = "/home/campus.ncl.ac.uk/c1041562/Desktop/codes/article_agent/AAAI2025_conferences.csv"
    
    print("Starting to process research titles...")
    
    # Process the CSV file
    results = process_csv_file(csv_file_path)
    
    # Save all results to JSON
    with open('classified_research_topics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(results)} titles. Results saved to classified_research_topics.json")
    
    # Create filtered files for each topic
    topics = ['time-series', 'multimodal', 'rag', 'kg']
    for topic in topics:
        filtered_results = filter_by_topic(results, topic)
        with open(f'{topic}_papers.json', 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, indent=2, ensure_ascii=False)
        print(f"{len(filtered_results)} papers classified as {topic} saved to {topic}_papers.json")
    
    # Print summary
    print("\nSummary:")
    topic_counts = {}
    for result in results:
        topic = result['topic']
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()