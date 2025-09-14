#!/usr/bin/env python3
"""
Script to update all visualization method signatures and output paths
to use the new filename convention: {datetime}_{top3keywords}_{analysis_type}
"""

import re

def update_scientific_bibliometrics():
    """Update the scientific bibliometrics file with new naming convention"""
    
    file_path = 'scientific_bibliometrics.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Method signature updates
    methods_to_update = [
        '_create_citation_impact_analysis',
        '_create_collaboration_network',
        '_create_venue_distribution_analysis',
        '_create_topic_evolution_analysis',
        '_create_author_impact_analysis',
        '_create_knowledge_domain_map',
        '_create_research_trend_analysis',
        '_generate_scientific_report'
    ]
    
    # Update method signatures
    for method in methods_to_update:
        old_sig = f"def {method}(self, df: pd.DataFrame, keyword: str) -> str:"
        new_sig = f"def {method}(self, df: pd.DataFrame, keyword: str, filename_prefix: str) -> str:"
        content = content.replace(old_sig, new_sig)
    
    # Update output paths - find all instances and replace
    output_path_patterns = [
        (r"output_path = os\.path\.join\(self\.output_dir, f'\{keyword\.replace\(\" \", \"_\"\)\}_citation_impact\.png'\)",
         "output_path = os.path.join(self.output_dir, f'{filename_prefix}_citation_impact.png')"),
        (r"output_path = os\.path\.join\(self\.output_dir, f'\{keyword\.replace\(\" \", \"_\"\)\}_collaboration_network\.png'\)",
         "output_path = os.path.join(self.output_dir, f'{filename_prefix}_collaboration_network.png')"),
        (r"output_path = os\.path\.join\(self\.output_dir, f'\{keyword\.replace\(\" \", \"_\"\)\}_venue_analysis\.png'\)",
         "output_path = os.path.join(self.output_dir, f'{filename_prefix}_venue_analysis.png')"),
        (r"output_path = os\.path\.join\(self\.output_dir, f'\{keyword\.replace\(\" \", \"_\"\)\}_topic_evolution\.png'\)",
         "output_path = os.path.join(self.output_dir, f'{filename_prefix}_topic_evolution.png')"),
        (r"output_path = os\.path\.join\(self\.output_dir, f'\{keyword\.replace\(\" \", \"_\"\)\}_author_impact\.png'\)",
         "output_path = os.path.join(self.output_dir, f'{filename_prefix}_author_impact.png')"),
        (r"output_path = os\.path\.join\(self\.output_dir, f'\{keyword\.replace\(\" \", \"_\"\)\}_knowledge_map\.png'\)",
         "output_path = os.path.join(self.output_dir, f'{filename_prefix}_knowledge_map.png')"),
        (r"output_path = os\.path\.join\(self\.output_dir, f'\{keyword\.replace\(\" \", \"_\"\)\}_trend_analysis\.png'\)",
         "output_path = os.path.join(self.output_dir, f'{filename_prefix}_trend_analysis.png')"),
        (r"report_path = os\.path\.join\(self\.output_dir, f'\{keyword\.replace\(\" \", \"_\"\)\}_scientific_report\.html'\)",
         "report_path = os.path.join(self.output_dir, f'{filename_prefix}_scientific_report.html')")
    ]
    
    # Apply all replacements
    for pattern, replacement in output_path_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Update the _generate_scientific_report method signature
    content = content.replace(
        "def _generate_scientific_report(self, df: pd.DataFrame, keyword: str, viz_paths: Dict[str, str]) -> str:",
        "def _generate_scientific_report(self, df: pd.DataFrame, keyword: str, viz_paths: Dict[str, str], filename_prefix: str) -> str:"
    )
    
    # Write the updated file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated scientific_bibliometrics.py with new filename convention")
    print("📁 Files will now be saved as: {datetime}_{top3keywords}_{analysis_type}")
    print("📝 Example: 20250830_171530_neural_networks_machine_temporal_distribution.png")

if __name__ == "__main__":
    update_scientific_bibliometrics()