import pandas as pd
from nltk.stem import SnowballStemmer
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re

class DomainDictionary:
    """
    Manages domain-specific phrase dictionaries for enhanced phrase mining.
    Supports loading from various formats and provides phrase matching utilities.
    """

    def __init__(self, language: str = 'english'):
        """Initializes the dictionary and the Snowball stemmer."""
        self.phrases: Dict[str, Dict] = {}  # phrase -> {category, priority, source}
        self.categories: Dict[str, List[str]] = {}  # category -> phrases
        self.stemmed_lookup: Dict[Tuple[str, ...], str] = {}  # stemmed -> original
        self.all_stemmed_words: set = set() # Stores all unique stemmed words
        self.stemmer = SnowballStemmer(language)

    def load_osha_codes(self, filepath: str, code_col: str = 'Code',
                        desc_col: str = 'Description') -> int:
        """
        Load OSHA event codes from CSV file.

        Args:
            filepath: Path to OSHA codes file
            code_col: Column name for event codes
            desc_col: Column name for descriptions
            
        Returns:
            Number of phrases loaded
        """
        df = pd.read_csv(filepath)

        count = 0
        for _, row in df.iterrows():
            if pd.notna(row.get(code_col)) and pd.notna(row.get(desc_col)):
                code = str(row[code_col])
                desc = str(row[desc_col]).strip()

                # Clean description
                cleaned = self._clean_phrase(desc)
                if cleaned and 3 <= len(cleaned.split()) <= 5:
                    self.add_phrase(
                        phrase=cleaned,
                        category=f"OSHA_{code[:2]}",  # Group by first 2 digits
                        priority=2,
                        source='OSHA'
                    )
                    count += 1

        return count

    def add_phrase(self, phrase: str, category: str = 'general',
                   priority: int = 1, source: str = 'unknown'):
        """Add a single phrase to the dictionary."""
        phrase_lower = phrase.lower()
        self.phrases[phrase_lower] = {
            'category': category,
            'priority': priority,
            'source': source,
            'original': phrase
        }

        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(phrase_lower)

    def build_stemmed_index(self):
        """Build index of stemmed phrases for matching using the class stemmer."""
        for phrase in self.phrases:
            words = phrase.split()
            stemmed = tuple(self.stemmer.stem(w) for w in words)
            self.stemmed_lookup[stemmed] = phrase
            self.all_stemmed_words.update(stemmed)

    def match_phrase(self, words: List[str]) -> Optional[Dict]:
        """
        Check if word sequence matches a domain phrase using direct and stemmed matching.

        Returns:
            Dict with match info or None
        """
        phrase = ' '.join(words).lower()

        # 1. Direct match
        if phrase in self.phrases:
            return {
                'phrase': phrase,
                'matched': True,
                'exact': True,
                **self.phrases[phrase]
            }

        # 2. Stemmed match
        if self.stemmed_lookup:
            stemmed = tuple(self.stemmer.stem(w) for w in words)
            if stemmed in self.stemmed_lookup:
                original = self.stemmed_lookup[stemmed]
                return {
                    'phrase': original,
                    'matched': True,
                    'exact': False,
                    **self.phrases[original]
                }

        return None

    def get_phrase_priority(self, phrase: str) -> int:
        """Get priority score for a phrase (0 if not in dictionary)."""
        phrase_lower = phrase.lower()
        if phrase_lower in self.phrases:
            return self.phrases[phrase_lower]['priority']
        return 0

    def get_category_phrases(self, category: str) -> List[str]:
        """Get all phrases in a category."""
        return self.categories.get(category, [])

    def _clean_phrase(self, phrase: str) -> str:
        """Clean and normalize a phrase."""
        # Remove special markers
        cleaned = re.sub(r'[—–-]\s*(unspecified|n\.e\.c\.)', '', phrase)
        cleaned = re.sub(r'[*†]', '', cleaned)
        cleaned = re.sub(r'[—–]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip().lower()

        # Skip generic terms
        if any(term in cleaned for term in ['unspecified', 'n.e.c.', 'not elsewhere']):
            return ''

        return cleaned

    def get_statistics(self) -> Dict:
        """Get dictionary statistics."""
        phrase_lengths = Counter(len(p.split()) for p in self.phrases)
        return {
            'total_phrases': len(self.phrases),
            'total_categories': len(self.categories),
            'phrases_by_length': dict(phrase_lengths),
            'sources': Counter(p['source'] for p in self.phrases.values())
        }