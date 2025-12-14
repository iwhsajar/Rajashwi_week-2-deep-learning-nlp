"""
Exercise 01 Solution: Tokenization Practice

Complete implementation of a custom tokenizer with vocabulary management.
"""

import re
import json
from collections import Counter
from typing import List, Dict, Optional

# =============================================================================
# PART 1: Basic Tokenizer
# =============================================================================

class SimpleTokenizer:
    """
    A simple word-level tokenizer with vocabulary management.
    
    Features:
    - Word-level tokenization
    - Lowercase normalization
    - Punctuation handling
    - Vocabulary building with frequency threshold
    - OOV (out-of-vocabulary) token handling
    """
    
    def __init__(self, 
                 num_words: Optional[int] = None,
                 oov_token: str = '<OOV>',
                 lower: bool = True,
                 filters: str = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
        """
        Initialize tokenizer.
        
        Args:
            num_words: Maximum vocabulary size (None = unlimited)
            oov_token: Token for out-of-vocabulary words
            lower: Whether to lowercase text
            filters: Characters to remove
        """
        self.num_words = num_words
        self.oov_token = oov_token
        self.lower = lower
        self.filters = filters
        
        # Vocabulary mappings
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        
        # Special tokens
        self.oov_index = 1  # OOV is always index 1
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if self.lower:
            text = text.lower()
        
        # Remove filtered characters
        for char in self.filters:
            text = text.replace(char, ' ')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        cleaned = self._clean_text(text)
        return cleaned.split()
    
    def fit_on_texts(self, texts: List[str]) -> None:
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of text strings
        """
        # Count all words
        for text in texts:
            tokens = self._tokenize(text)
            self.word_counts.update(tokens)
        
        # Build vocabulary (sorted by frequency)
        sorted_words = sorted(
            self.word_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Limit vocabulary size if specified
        if self.num_words:
            sorted_words = sorted_words[:self.num_words - 1]  # -1 for OOV
        
        # Create word index (OOV is 1, words start from 2)
        self.word_index = {self.oov_token: self.oov_index}
        for idx, (word, count) in enumerate(sorted_words, start=2):
            self.word_index[word] = idx
        
        # Create reverse mapping
        self.index_word = {v: k for k, v in self.word_index.items()}
        
        print(f"Vocabulary built: {len(self.word_index)} words")
        print(f"Total unique words seen: {len(self.word_counts)}")
        if self.num_words:
            print(f"Vocabulary limited to: {self.num_words} words")
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert texts to sequences of integers.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of integer sequences
        """
        sequences = []
        oov_count = 0
        total_tokens = 0
        
        for text in texts:
            tokens = self._tokenize(text)
            sequence = []
            
            for token in tokens:
                total_tokens += 1
                if token in self.word_index:
                    sequence.append(self.word_index[token])
                else:
                    sequence.append(self.oov_index)
                    oov_count += 1
            
            sequences.append(sequence)
        
        oov_rate = oov_count / total_tokens if total_tokens > 0 else 0
        print(f"OOV rate: {oov_rate:.2%} ({oov_count}/{total_tokens} tokens)")
        
        return sequences
    
    def sequences_to_texts(self, sequences: List[List[int]]) -> List[str]:
        """
        Convert sequences back to text.
        
        Args:
            sequences: List of integer sequences
            
        Returns:
            List of text strings
        """
        texts = []
        
        for sequence in sequences:
            words = []
            for idx in sequence:
                if idx in self.index_word:
                    words.append(self.index_word[idx])
                else:
                    words.append(self.oov_token)
            texts.append(' '.join(words))
        
        return texts
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.word_index)
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                'word_index': self.word_index,
                'config': {
                    'num_words': self.num_words,
                    'oov_token': self.oov_token,
                    'lower': self.lower
                }
            }, f, indent=2)
        print(f"Vocabulary saved to: {filepath}")
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.word_index = data['word_index']
        self.index_word = {int(v): k for k, v in self.word_index.items()}
        
        config = data.get('config', {})
        self.num_words = config.get('num_words')
        self.oov_token = config.get('oov_token', '<OOV>')
        self.lower = config.get('lower', True)
        
        print(f"Vocabulary loaded: {len(self.word_index)} words")


# =============================================================================
# PART 2: Advanced Tokenizer with Subword Support
# =============================================================================

class CharacterTokenizer:
    """Character-level tokenizer."""
    
    def __init__(self, oov_char: str = '<UNK>'):
        self.oov_char = oov_char
        self.char_index: Dict[str, int] = {}
        self.index_char: Dict[int, str] = {}
    
    def fit_on_texts(self, texts: List[str]) -> None:
        """Build character vocabulary."""
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Sort for reproducibility
        sorted_chars = sorted(chars)
        
        # Build index
        self.char_index = {self.oov_char: 0}
        for idx, char in enumerate(sorted_chars, start=1):
            self.char_index[char] = idx
        
        self.index_char = {v: k for k, v in self.char_index.items()}
        print(f"Character vocabulary: {len(self.char_index)} characters")
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert texts to character sequences."""
        sequences = []
        for text in texts:
            sequence = [
                self.char_index.get(char, 0) 
                for char in text
            ]
            sequences.append(sequence)
        return sequences
    
    def sequences_to_texts(self, sequences: List[List[int]]) -> List[str]:
        """Convert character sequences back to text."""
        texts = []
        for sequence in sequences:
            text = ''.join(
                self.index_char.get(idx, self.oov_char)
                for idx in sequence
            )
            texts.append(text)
        return texts


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 01 Solution: Tokenization Practice")
    print("=" * 60)
    
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of technology.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of data.",
        "The fox is quick and the dog is lazy!"
    ]
    
    # Test SimpleTokenizer
    print("\n" + "=" * 60)
    print("Part 1: Word-Level Tokenization")
    print("=" * 60)
    
    tokenizer = SimpleTokenizer(num_words=50, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    # Show vocabulary
    print("\nTop 10 words by frequency:")
    for word, count in tokenizer.word_counts.most_common(10):
        idx = tokenizer.word_index.get(word, '?')
        print(f"  '{word}': count={count}, index={idx}")
    
    # Convert to sequences
    print("\nConverting texts to sequences:")
    sequences = tokenizer.texts_to_sequences(texts[:2])
    for text, seq in zip(texts[:2], sequences):
        print(f"  Text: '{text[:50]}...'")
        print(f"  Sequence: {seq}")
    
    # Convert back
    print("\nConverting sequences back to text:")
    reconstructed = tokenizer.sequences_to_texts(sequences)
    for original, reconstructed_text in zip(texts[:2], reconstructed):
        print(f"  Original:     '{original[:50]}...'")
        print(f"  Reconstructed: '{reconstructed_text[:50]}...'")
    
    # Test OOV handling
    print("\n" + "=" * 60)
    print("Part 2: OOV Handling")
    print("=" * 60)
    
    new_texts = ["Artificial intelligence and quantum computing are amazing!"]
    new_sequences = tokenizer.texts_to_sequences(new_texts)
    print(f"New text: '{new_texts[0]}'")
    print(f"Sequence: {new_sequences[0]}")
    print(f"Decoded: '{tokenizer.sequences_to_texts(new_sequences)[0]}'")
    
    # Test CharacterTokenizer
    print("\n" + "=" * 60)
    print("Part 3: Character-Level Tokenization")
    print("=" * 60)
    
    char_tokenizer = CharacterTokenizer()
    char_tokenizer.fit_on_texts(texts)
    
    char_sequences = char_tokenizer.texts_to_sequences(["Hello!"])
    print(f"Text: 'Hello!'")
    print(f"Character sequence: {char_sequences[0]}")
    print(f"Decoded: '{char_tokenizer.sequences_to_texts(char_sequences)[0]}'")
    
    # Save vocabulary
    print("\n" + "=" * 60)
    print("Part 4: Saving Vocabulary")
    print("=" * 60)
    
    tokenizer.save_vocab('vocabulary.json')
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)

