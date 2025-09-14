"""
Conference Registry System
Maps top-tier conferences to their metadata, DBLP keys, and scraping configurations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum

class ConferenceType(Enum):
    """Conference domain classification"""
    AI_ML = "AI/ML"
    NLP = "Natural Language Processing"
    COMPUTER_VISION = "Computer Vision"
    SECURITY = "Security"
    SYSTEMS = "Systems"
    GENERAL_CS = "General Computer Science"

@dataclass
class ConferenceInfo:
    """Complete conference information and scraping configuration"""
    acronym: str
    full_name: str
    conference_type: ConferenceType
    dblp_key: str
    official_url_pattern: str
    years_available: List[int]
    typical_paper_count: int
    has_workshops: bool = False
    arxiv_keywords: List[str] = None
    semantic_scholar_venue: str = None
    additional_sources: List[str] = None

class ConferenceRegistry:
    """Registry of top-tier conferences with scraping configurations"""
    
    def __init__(self):
        self.conferences = self._build_conference_registry()
        self.acronym_to_info = {conf.acronym.upper(): conf for conf in self.conferences}
        
    def _build_conference_registry(self) -> List[ConferenceInfo]:
        """Build comprehensive conference registry"""
        
        conferences = [
            # AI/ML Core Conferences
            ConferenceInfo(
                acronym="AAAI",
                full_name="AAAI Conference on Artificial Intelligence",
                conference_type=ConferenceType.AI_ML,
                dblp_key="conf/aaai",
                official_url_pattern="https://aaai.org/conference/aaai-{year}/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=1500,
                has_workshops=True,
                arxiv_keywords=["artificial intelligence", "AAAI"],
                semantic_scholar_venue="AAAI"
            ),
            
            ConferenceInfo(
                acronym="ICLR",
                full_name="International Conference on Learning Representations",
                conference_type=ConferenceType.AI_ML,
                dblp_key="conf/iclr",
                official_url_pattern="https://iclr.cc/Conferences/{year}",
                years_available=list(range(2020, 2026)),
                typical_paper_count=800,
                has_workshops=True,
                arxiv_keywords=["learning representations", "ICLR"],
                semantic_scholar_venue="ICLR"
            ),
            
            ConferenceInfo(
                acronym="ICML",
                full_name="International Conference on Machine Learning",
                conference_type=ConferenceType.AI_ML,
                dblp_key="conf/icml",
                official_url_pattern="https://icml.cc/{year}/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=1200,
                has_workshops=True,
                arxiv_keywords=["machine learning", "ICML"],
                semantic_scholar_venue="ICML"
            ),
            
            ConferenceInfo(
                acronym="NeurIPS",
                full_name="Neural Information Processing Systems",
                conference_type=ConferenceType.AI_ML,
                dblp_key="conf/nips",
                official_url_pattern="https://nips.cc/Conferences/{year}",
                years_available=list(range(2020, 2026)),
                typical_paper_count=2000,
                has_workshops=True,
                arxiv_keywords=["neural information processing", "NeurIPS", "NIPS"],
                semantic_scholar_venue="NeurIPS"
            ),
            
            ConferenceInfo(
                acronym="UAI",
                full_name="Conference on Uncertainty in Artificial Intelligence",
                conference_type=ConferenceType.AI_ML,
                dblp_key="conf/uai",
                official_url_pattern="https://www.auai.org/uai{year}/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=300,
                arxiv_keywords=["uncertainty", "UAI"],
                semantic_scholar_venue="UAI"
            ),
            
            ConferenceInfo(
                acronym="COLM",
                full_name="Conference on Language Modeling", 
                conference_type=ConferenceType.AI_ML,
                dblp_key="conf/colm",
                official_url_pattern="https://colmweb.org/{year}/",
                years_available=[2024, 2025],
                typical_paper_count=200,
                arxiv_keywords=["language modeling", "COLM"],
                semantic_scholar_venue="COLM"
            ),
            
            ConferenceInfo(
                acronym="COLT",
                full_name="Conference on Learning Theory",
                conference_type=ConferenceType.AI_ML,
                dblp_key="conf/colt",
                official_url_pattern="https://learningtheory.org/colt{year}/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=150,
                arxiv_keywords=["learning theory", "COLT"],
                semantic_scholar_venue="COLT"
            ),
            
            ConferenceInfo(
                acronym="CoRL",
                full_name="Conference on Robot Learning",
                conference_type=ConferenceType.AI_ML,
                dblp_key="conf/corl",
                official_url_pattern="https://www.robot-learning.org/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=200,
                arxiv_keywords=["robot learning", "CoRL"],
                semantic_scholar_venue="CoRL"
            ),
            
            ConferenceInfo(
                acronym="MLSYS",
                full_name="Conference on Machine Learning and Systems",
                conference_type=ConferenceType.AI_ML,
                dblp_key="conf/mlsys",
                official_url_pattern="https://mlsys.org/Conferences/{year}",
                years_available=list(range(2020, 2026)),
                typical_paper_count=150,
                arxiv_keywords=["machine learning systems", "MLSys"],
                semantic_scholar_venue="MLSys"
            ),
            
            # NLP Conferences
            ConferenceInfo(
                acronym="ACL",
                full_name="Annual Meeting of the Association for Computational Linguistics",
                conference_type=ConferenceType.NLP,
                dblp_key="conf/acl",
                official_url_pattern="https://{year}.aclweb.org/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=800,
                has_workshops=True,
                arxiv_keywords=["computational linguistics", "ACL"],
                semantic_scholar_venue="ACL"
            ),
            
            ConferenceInfo(
                acronym="EMNLP",
                full_name="Conference on Empirical Methods in Natural Language Processing",
                conference_type=ConferenceType.NLP,
                dblp_key="conf/emnlp",
                official_url_pattern="https://{year}.emnlp.org/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=700,
                has_workshops=True,
                arxiv_keywords=["empirical methods", "natural language processing", "EMNLP"],
                semantic_scholar_venue="EMNLP"
            ),
            
            ConferenceInfo(
                acronym="NAACL",
                full_name="North American Chapter of the Association for Computational Linguistics",
                conference_type=ConferenceType.NLP,
                dblp_key="conf/naacl",
                official_url_pattern="https://{year}.naacl.org/",
                years_available=list(range(2021, 2026, 3)),  # Every 3 years
                typical_paper_count=500,
                has_workshops=True,
                arxiv_keywords=["computational linguistics", "NAACL"],
                semantic_scholar_venue="NAACL"
            ),
            
            ConferenceInfo(
                acronym="INTERSPEECH",
                full_name="Annual Conference of the International Speech Communication Association",
                conference_type=ConferenceType.NLP,
                dblp_key="conf/interspeech",
                official_url_pattern="https://www.interspeech{year}.org/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=1200,
                arxiv_keywords=["speech processing", "INTERSPEECH"],
                semantic_scholar_venue="INTERSPEECH"
            ),
            
            ConferenceInfo(
                acronym="IWSLT",
                full_name="International Workshop on Spoken Language Translation",
                conference_type=ConferenceType.NLP,
                dblp_key="conf/iwslt",
                official_url_pattern="https://iwslt.org/{year}/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=100,
                arxiv_keywords=["spoken language translation", "IWSLT"],
                semantic_scholar_venue="IWSLT"
            ),
            
            # Computer Vision Conferences
            ConferenceInfo(
                acronym="CVPR",
                full_name="IEEE Conference on Computer Vision and Pattern Recognition",
                conference_type=ConferenceType.COMPUTER_VISION,
                dblp_key="conf/cvpr",
                official_url_pattern="https://cvpr{year}.thecvf.com/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=2300,
                has_workshops=True,
                arxiv_keywords=["computer vision", "pattern recognition", "CVPR"],
                semantic_scholar_venue="CVPR"
            ),
            
            ConferenceInfo(
                acronym="ECCV",
                full_name="European Conference on Computer Vision",
                conference_type=ConferenceType.COMPUTER_VISION,
                dblp_key="conf/eccv",
                official_url_pattern="https://eccv{year}.ecva.net/",
                years_available=list(range(2020, 2026, 2)),  # Every 2 years
                typical_paper_count=1500,
                has_workshops=True,
                arxiv_keywords=["computer vision", "ECCV"],
                semantic_scholar_venue="ECCV"
            ),
            
            ConferenceInfo(
                acronym="ICCV",
                full_name="IEEE International Conference on Computer Vision",
                conference_type=ConferenceType.COMPUTER_VISION,
                dblp_key="conf/iccv",
                official_url_pattern="https://iccv{year}.thecvf.com/",
                years_available=list(range(2021, 2026, 2)),  # Odd years
                typical_paper_count=1600,
                has_workshops=True,
                arxiv_keywords=["computer vision", "ICCV"],
                semantic_scholar_venue="ICCV"
            ),
            
            # General CS
            ConferenceInfo(
                acronym="IJCAI",
                full_name="International Joint Conference on Artificial Intelligence",
                conference_type=ConferenceType.GENERAL_CS,
                dblp_key="conf/ijcai",
                official_url_pattern="https://ijcai-{year}.org/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=1000,
                has_workshops=True,
                arxiv_keywords=["artificial intelligence", "IJCAI"],
                semantic_scholar_venue="IJCAI"
            ),
            
            # Security Conferences
            ConferenceInfo(
                acronym="NDSS",
                full_name="Network and Distributed System Security Symposium",
                conference_type=ConferenceType.SECURITY,
                dblp_key="conf/ndss",
                official_url_pattern="https://www.ndss-symposium.org/ndss{year}/",
                years_available=list(range(2020, 2026)),
                typical_paper_count=90,
                arxiv_keywords=["network security", "NDSS"],
                semantic_scholar_venue="NDSS"
            ),
            
            ConferenceInfo(
                acronym="USENIX-Sec",
                full_name="USENIX Security Symposium",
                conference_type=ConferenceType.SECURITY,
                dblp_key="conf/uss",
                official_url_pattern="https://www.usenix.org/conference/usenixsecurity{year}",
                years_available=list(range(2020, 2026)),
                typical_paper_count=150,
                arxiv_keywords=["security", "USENIX Security"],
                semantic_scholar_venue="USENIX Security"
            ),
            
            # Systems Conferences
            ConferenceInfo(
                acronym="OSDI",
                full_name="USENIX Symposium on Operating Systems Design and Implementation",
                conference_type=ConferenceType.SYSTEMS,
                dblp_key="conf/osdi",
                official_url_pattern="https://www.usenix.org/conference/osdi{year}",
                years_available=list(range(2020, 2026, 2)),  # Every 2 years
                typical_paper_count=40,
                arxiv_keywords=["operating systems", "OSDI"],
                semantic_scholar_venue="OSDI"
            ),
            
            ConferenceInfo(
                acronym="USENIX-Fast",
                full_name="USENIX Conference on File and Storage Technologies",
                conference_type=ConferenceType.SYSTEMS,
                dblp_key="conf/fast",
                official_url_pattern="https://www.usenix.org/conference/fast{year}",
                years_available=list(range(2020, 2026)),
                typical_paper_count=30,
                arxiv_keywords=["file systems", "storage", "FAST"],
                semantic_scholar_venue="FAST"
            )
        ]
        
        return conferences
    
    def get_conference(self, acronym: str) -> Optional[ConferenceInfo]:
        """Get conference info by acronym"""
        return self.acronym_to_info.get(acronym.upper())
    
    def get_conferences_by_type(self, conf_type: ConferenceType) -> List[ConferenceInfo]:
        """Get all conferences of a specific type"""
        return [conf for conf in self.conferences if conf.conference_type == conf_type]
    
    def get_all_acronyms(self) -> List[str]:
        """Get all conference acronyms"""
        return list(self.acronym_to_info.keys())
    
    def get_conferences_for_year(self, year: int) -> List[ConferenceInfo]:
        """Get conferences that have proceedings for a specific year"""
        return [conf for conf in self.conferences if year in conf.years_available]
    
    def estimate_total_papers(self, conferences: List[str] = None, years: List[int] = None) -> Dict[str, int]:
        """Estimate total papers for given conferences and years"""
        if conferences is None:
            conferences = self.get_all_acronyms()
        if years is None:
            years = list(range(2020, 2025))
            
        estimates = {}
        total = 0
        
        for acronym in conferences:
            conf = self.get_conference(acronym)
            if conf:
                available_years = [y for y in years if y in conf.years_available]
                conf_total = len(available_years) * conf.typical_paper_count
                estimates[acronym] = conf_total
                total += conf_total
        
        estimates['TOTAL'] = total
        return estimates
    
    def get_scraping_priority(self) -> List[str]:
        """Get conferences ordered by scraping priority (largest first)"""
        priority_order = sorted(
            self.conferences,
            key=lambda x: x.typical_paper_count * len(x.years_available),
            reverse=True
        )
        return [conf.acronym for conf in priority_order]

def main():
    """Demo usage of conference registry"""
    registry = ConferenceRegistry()
    
    print("=== Conference Registry Demo ===")
    print(f"Total conferences: {len(registry.conferences)}")
    
    # Show conference types
    for conf_type in ConferenceType:
        confs = registry.get_conferences_by_type(conf_type)
        print(f"\n{conf_type.value}: {len(confs)} conferences")
        for conf in confs:
            print(f"  - {conf.acronym}: {conf.full_name}")
    
    # Estimate paper counts
    print("\n=== Paper Count Estimates (2020-2024) ===")
    estimates = registry.estimate_total_papers(years=list(range(2020, 2025)))
    for acronym, count in estimates.items():
        if acronym != 'TOTAL':
            print(f"{acronym}: ~{count:,} papers")
    print(f"\nTOTAL ESTIMATED PAPERS: ~{estimates['TOTAL']:,}")
    
    # Show scraping priority
    print("\n=== Scraping Priority Order ===")
    priority = registry.get_scraping_priority()[:10]
    for i, acronym in enumerate(priority, 1):
        conf = registry.get_conference(acronym)
        print(f"{i:2d}. {acronym}: ~{conf.typical_paper_count:,} papers/year")

if __name__ == "__main__":
    main()