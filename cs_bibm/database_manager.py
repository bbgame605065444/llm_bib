"""
Database Manager for Conference Papers
SQLite-based storage with comprehensive schema for academic paper metadata.
"""

import sqlite3
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import logging

class DatabaseManager:
    """Manages SQLite database for conference papers"""
    
    def __init__(self, db_path: str = "cs_bibm/data/conference_papers.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()
        self.init_database()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for database operations"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def init_database(self):
        """Initialize database with comprehensive schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main papers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT UNIQUE NOT NULL,  -- Generated hash for deduplication
                    title TEXT NOT NULL,
                    authors TEXT,  -- JSON array of author names
                    abstract TEXT,
                    year INTEGER,
                    venue TEXT NOT NULL,
                    venue_full_name TEXT,
                    conference_type TEXT,
                    
                    -- External identifiers
                    doi TEXT,
                    arxiv_id TEXT,
                    dblp_key TEXT,
                    semantic_scholar_id TEXT,
                    
                    -- URLs and links
                    pdf_url TEXT,
                    official_url TEXT,
                    dblp_url TEXT,
                    arxiv_url TEXT,
                    
                    -- Citation and impact metrics
                    citation_count INTEGER DEFAULT 0,
                    influential_citation_count INTEGER DEFAULT 0,
                    h_index_contribution INTEGER DEFAULT 0,
                    
                    -- Content metadata
                    page_count INTEGER,
                    has_code BOOLEAN DEFAULT FALSE,
                    code_url TEXT,
                    dataset_urls TEXT,  -- JSON array
                    
                    -- Research categorization
                    research_area TEXT,
                    keywords TEXT,  -- JSON array
                    topics TEXT,    -- JSON array of extracted topics
                    
                    -- Scraping metadata
                    source TEXT NOT NULL,  -- dblp, arxiv, venue_site, semantic_scholar
                    scraping_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    scraping_success BOOLEAN DEFAULT TRUE,
                    
                    -- Additional metadata
                    bibtex TEXT,
                    notes TEXT,
                    quality_score REAL,  -- 0.0-1.0 quality assessment
                    
                    UNIQUE(title, year, venue)
                )
            ''')
            
            # Authors table for better normalization
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS authors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    normalized_name TEXT,  -- Standardized author name
                    affiliation TEXT,
                    orcid_id TEXT,
                    semantic_scholar_id TEXT,
                    h_index INTEGER DEFAULT 0,
                    citation_count INTEGER DEFAULT 0,
                    first_seen_year INTEGER,
                    last_seen_year INTEGER,
                    paper_count INTEGER DEFAULT 0,
                    
                    UNIQUE(normalized_name)
                )
            ''')
            
            # Paper-author relationship table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_authors (
                    paper_id TEXT,
                    author_id INTEGER,
                    position INTEGER,  -- Author position in paper
                    is_corresponding BOOLEAN DEFAULT FALSE,
                    
                    PRIMARY KEY (paper_id, author_id),
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
                    FOREIGN KEY (author_id) REFERENCES authors(id)
                )
            ''')
            
            # Venues table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS venues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    acronym TEXT UNIQUE NOT NULL,
                    full_name TEXT,
                    conference_type TEXT,
                    dblp_key TEXT,
                    official_url TEXT,
                    h5_index INTEGER DEFAULT 0,
                    h5_median INTEGER DEFAULT 0,
                    
                    -- Scraping statistics
                    total_papers_scraped INTEGER DEFAULT 0,
                    years_covered TEXT,  -- JSON array of years
                    last_scraping_date DATETIME,
                    scraping_success_rate REAL DEFAULT 1.0
                )
            ''')
            
            # Citations relationship table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS citations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    citing_paper_id TEXT,
                    cited_paper_id TEXT,
                    citation_context TEXT,  -- Context where citation appears
                    
                    FOREIGN KEY (citing_paper_id) REFERENCES papers(paper_id),
                    FOREIGN KEY (cited_paper_id) REFERENCES papers(paper_id),
                    UNIQUE(citing_paper_id, cited_paper_id)
                )
            ''')
            
            # Scraping log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scraping_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue TEXT,
                    year INTEGER,
                    source TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    papers_found INTEGER,
                    papers_inserted INTEGER,
                    papers_updated INTEGER,
                    errors_count INTEGER DEFAULT 0,
                    success BOOLEAN DEFAULT TRUE,
                    notes TEXT
                )
            ''')
            
            # Create indexes for better performance
            self._create_indexes(cursor)
            
            conn.commit()
            self.logger.info("Database initialized successfully")
    
    def _create_indexes(self, cursor: sqlite3.Cursor):
        """Create database indexes for improved query performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_papers_venue_year ON papers(venue, year)",
            "CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title)",
            "CREATE INDEX IF NOT EXISTS idx_papers_authors ON papers(authors)",
            "CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)",
            "CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id)",
            "CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source)",
            "CREATE INDEX IF NOT EXISTS idx_papers_scraping_date ON papers(scraping_date)",
            "CREATE INDEX IF NOT EXISTS idx_authors_name ON authors(normalized_name)",
            "CREATE INDEX IF NOT EXISTS idx_paper_authors_paper ON paper_authors(paper_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_authors_author ON paper_authors(author_id)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    def generate_paper_id(self, title: str, authors: List[str], year: int) -> str:
        """Generate unique paper ID based on title, authors, and year"""
        # Normalize inputs
        title_norm = title.lower().strip()
        authors_norm = sorted([author.lower().strip() for author in authors])
        
        # Create hash
        content = f"{title_norm}|{','.join(authors_norm)}|{year}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def insert_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Insert a new paper into the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Generate paper ID
                authors = paper_data.get('authors', [])
                if isinstance(authors, str):
                    authors = json.loads(authors) if authors else []
                
                paper_id = self.generate_paper_id(
                    paper_data['title'], 
                    authors, 
                    paper_data.get('year', 0)
                )
                
                # Prepare data for insertion
                insert_data = {
                    'paper_id': paper_id,
                    'title': paper_data['title'],
                    'authors': json.dumps(authors) if authors else None,
                    'abstract': paper_data.get('abstract'),
                    'year': paper_data.get('year'),
                    'venue': paper_data.get('venue'),
                    'venue_full_name': paper_data.get('venue_full_name'),
                    'conference_type': paper_data.get('conference_type'),
                    'doi': paper_data.get('doi'),
                    'arxiv_id': paper_data.get('arxiv_id'),
                    'dblp_key': paper_data.get('dblp_key'),
                    'semantic_scholar_id': paper_data.get('semantic_scholar_id'),
                    'pdf_url': paper_data.get('pdf_url'),
                    'official_url': paper_data.get('official_url'),
                    'dblp_url': paper_data.get('dblp_url'),
                    'arxiv_url': paper_data.get('arxiv_url'),
                    'citation_count': paper_data.get('citation_count', 0),
                    'influential_citation_count': paper_data.get('influential_citation_count', 0),
                    'page_count': paper_data.get('page_count'),
                    'has_code': paper_data.get('has_code', False),
                    'code_url': paper_data.get('code_url'),
                    'dataset_urls': json.dumps(paper_data.get('dataset_urls', [])),
                    'research_area': paper_data.get('research_area'),
                    'keywords': json.dumps(paper_data.get('keywords', [])),
                    'topics': json.dumps(paper_data.get('topics', [])),
                    'source': paper_data.get('source', 'unknown'),
                    'bibtex': paper_data.get('bibtex'),
                    'notes': paper_data.get('notes'),
                    'quality_score': paper_data.get('quality_score')
                }
                
                # Insert paper
                placeholders = ', '.join(['?' for _ in insert_data])
                columns = ', '.join(insert_data.keys())
                
                cursor.execute(f'''
                    INSERT OR REPLACE INTO papers ({columns})
                    VALUES ({placeholders})
                ''', list(insert_data.values()))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error inserting paper: {e}")
            return False
    
    def batch_insert_papers(self, papers: List[Dict[str, Any]], batch_size: int = 100) -> Tuple[int, int]:
        """Batch insert multiple papers"""
        inserted = 0
        errors = 0
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for paper_data in batch:
                    try:
                        if self.insert_paper(paper_data):
                            inserted += 1
                        else:
                            errors += 1
                    except Exception as e:
                        self.logger.error(f"Batch insert error: {e}")
                        errors += 1
        
        self.logger.info(f"Batch insert completed: {inserted} inserted, {errors} errors")
        return inserted, errors
    
    def paper_exists(self, title: str, authors: List[str], year: int) -> bool:
        """Check if paper already exists in database"""
        paper_id = self.generate_paper_id(title, authors, year)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,))
            return cursor.fetchone() is not None
    
    def get_papers_by_venue(self, venue: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all papers from a specific venue"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if year:
                cursor.execute("""
                    SELECT * FROM papers 
                    WHERE venue = ? AND year = ? 
                    ORDER BY title
                """, (venue, year))
            else:
                cursor.execute("""
                    SELECT * FROM papers 
                    WHERE venue = ? 
                    ORDER BY year DESC, title
                """, (venue,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM papers")
            stats['total_papers'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT venue) FROM papers")
            stats['unique_venues'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT year) FROM papers")
            stats['years_covered'] = cursor.fetchone()[0]
            
            # Year distribution
            cursor.execute("""
                SELECT year, COUNT(*) as count 
                FROM papers 
                WHERE year IS NOT NULL 
                GROUP BY year 
                ORDER BY year
            """)
            stats['papers_by_year'] = dict(cursor.fetchall())
            
            # Venue distribution
            cursor.execute("""
                SELECT venue, COUNT(*) as count 
                FROM papers 
                GROUP BY venue 
                ORDER BY count DESC 
                LIMIT 20
            """)
            stats['top_venues'] = dict(cursor.fetchall())
            
            # Source distribution
            cursor.execute("""
                SELECT source, COUNT(*) as count 
                FROM papers 
                GROUP BY source
            """)
            stats['papers_by_source'] = dict(cursor.fetchall())
            
            # Quality metrics
            cursor.execute("SELECT AVG(citation_count) FROM papers WHERE citation_count > 0")
            result = cursor.fetchone()
            stats['avg_citations'] = float(result[0]) if result[0] else 0
            
            cursor.execute("SELECT COUNT(*) FROM papers WHERE doi IS NOT NULL")
            stats['papers_with_doi'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM papers WHERE arxiv_id IS NOT NULL")
            stats['papers_with_arxiv'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM papers WHERE pdf_url IS NOT NULL")
            stats['papers_with_pdf'] = cursor.fetchone()[0]
            
            # Database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            result = cursor.fetchone()
            stats['database_size_bytes'] = result[0] if result else 0
            stats['database_size_mb'] = stats['database_size_bytes'] / (1024 * 1024)
            
            return stats
    
    def search_papers(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search papers by title, authors, or abstract"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            search_query = f"%{query}%"
            cursor.execute("""
                SELECT * FROM papers 
                WHERE title LIKE ? OR authors LIKE ? OR abstract LIKE ?
                ORDER BY 
                    CASE 
                        WHEN title LIKE ? THEN 1
                        WHEN authors LIKE ? THEN 2  
                        ELSE 3
                    END,
                    citation_count DESC
                LIMIT ?
            """, (search_query, search_query, search_query, search_query, search_query, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def export_to_csv(self, output_path: str, venue: Optional[str] = None) -> bool:
        """Export papers to CSV format"""
        try:
            import csv
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if venue:
                    cursor.execute("SELECT * FROM papers WHERE venue = ?", (venue,))
                else:
                    cursor.execute("SELECT * FROM papers")
                
                papers = cursor.fetchall()
                
                if not papers:
                    self.logger.warning("No papers to export")
                    return False
                
                # Get column names
                cursor.execute("PRAGMA table_info(papers)")
                columns = [column[1] for column in cursor.fetchall()]
                
                with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(columns)
                    writer.writerows(papers)
                
                self.logger.info(f"Exported {len(papers)} papers to {output_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        pass  # Using context managers, so no explicit close needed

def main():
    """Demo usage of database manager"""
    db = DatabaseManager("test_conference_papers.db")
    
    # Demo paper insertion
    sample_paper = {
        'title': 'Attention Is All You Need',
        'authors': ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
        'abstract': 'The dominant sequence transduction models...',
        'year': 2017,
        'venue': 'NeurIPS',
        'venue_full_name': 'Neural Information Processing Systems',
        'conference_type': 'AI/ML',
        'arxiv_id': '1706.03762',
        'citation_count': 50000,
        'source': 'dblp'
    }
    
    print("Inserting sample paper...")
    success = db.insert_paper(sample_paper)
    print(f"Insert successful: {success}")
    
    # Get stats
    print("\nDatabase Statistics:")
    stats = db.get_database_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}: {len(value)} entries")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()