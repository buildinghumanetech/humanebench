"""
Semantic similarity-based deduplication using sentence transformers.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Set, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticDeduplicator:
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2", # also try text-embedding-3-large
                 similarity_threshold: float = 0.60,
                 cache_dir: str = None):
        """
        Initialize semantic deduplicator.

        Args:
            model_name: Sentence transformer model to use
            similarity_threshold: Cosine similarity threshold for duplicates (0-1)
            cache_dir: Directory to cache embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "embeddings_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache file paths
        self.embeddings_cache_file = self.cache_dir / "existing_embeddings.pkl"
        self.texts_cache_file = self.cache_dir / "existing_texts.pkl"

        # Load existing embeddings and texts
        self.existing_embeddings, self.existing_texts = self._load_cache()

        # Track recent deduplication patterns for feedback
        self.recent_duplicates = []  # Store recent duplicate patterns
        self.session_stats = {
            "total_processed": 0,
            "total_duplicates": 0,
            "common_duplicate_patterns": {}
        }

    def _load_cache(self) -> Tuple[np.ndarray, List[str]]:
        """Load cached embeddings and texts."""
        try:
            if self.embeddings_cache_file.exists() and self.texts_cache_file.exists():
                with open(self.embeddings_cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                with open(self.texts_cache_file, 'rb') as f:
                    texts = pickle.load(f)
                print(f"Loaded {len(texts)} cached embeddings")
                return embeddings, texts
        except Exception as e:
            print(f"Warning: Could not load cached embeddings: {e}")

        return np.array([]), []

    def _save_cache(self):
        """Save embeddings and texts to cache."""
        try:
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(self.existing_embeddings, f)
            with open(self.texts_cache_file, 'wb') as f:
                pickle.dump(self.existing_texts, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def update_existing_texts(self, texts: List[str]):
        """Update the cache with new existing texts."""
        if not texts:
            return

        print(f"Computing embeddings for {len(texts)} existing texts...")
        new_embeddings = self.model.encode(texts, show_progress_bar=True)

        if len(self.existing_embeddings) == 0:
            self.existing_embeddings = new_embeddings
            self.existing_texts = texts
        else:
            # Append new embeddings
            self.existing_embeddings = np.vstack([self.existing_embeddings, new_embeddings])
            self.existing_texts.extend(texts)

        # Save to cache
        self._save_cache()
        print(f"Cache updated with {len(self.existing_texts)} total texts")

    def find_duplicates(self, new_texts: List[str]) -> Tuple[List[int], List[float]]:
        """
        Find duplicates among new texts compared to existing texts.

        Args:
            new_texts: List of new text scenarios to check

        Returns:
            Tuple of (duplicate_indices, similarity_scores)
        """
        if not new_texts or len(self.existing_embeddings) == 0:
            return [], []

        print(f"Computing embeddings for {len(new_texts)} new texts...")
        new_embeddings = self.model.encode(new_texts, show_progress_bar=True)

        print("Computing semantic similarities...")
        duplicate_indices = []
        similarity_scores = []

        # Update session stats
        self.session_stats["total_processed"] += len(new_texts)

        for i, new_embedding in enumerate(new_embeddings):
            # Compute cosine similarity with all existing embeddings
            similarities = cosine_similarity([new_embedding], self.existing_embeddings)[0]
            max_similarity = np.max(similarities)

            if max_similarity >= self.similarity_threshold:
                duplicate_indices.append(i)
                similarity_scores.append(max_similarity)

                # Find the most similar existing text for logging
                most_similar_idx = np.argmax(similarities)
                most_similar_text = self.existing_texts[most_similar_idx]
                print(f"  Duplicate found (similarity: {max_similarity:.3f}):")
                print(f"    New: {new_texts[i][:100]}...")
                print(f"    Existing: {most_similar_text[:100]}...")

                # Store duplicate pattern for feedback analysis
                self._analyze_duplicate_pattern(new_texts[i], most_similar_text, max_similarity)

        # Update duplicate count
        self.session_stats["total_duplicates"] += len(duplicate_indices)

        return duplicate_indices, similarity_scores

    def _analyze_duplicate_pattern(self, new_text: str, existing_text: str, similarity: float):
        """Analyze and store duplicate patterns for feedback generation."""
        # Extract pattern type
        new_words = set(new_text.lower().split())
        existing_words = set(existing_text.lower().split())

        # Find common patterns
        common_words = new_words & existing_words
        if len(common_words) > 3:  # Significant word overlap
            # Extract key pattern indicators
            pattern_type = "word_overlap"
            if any(word in new_text.lower() and word in existing_text.lower()
                   for word in ["how to", "what is", "help me", "should i"]):
                pattern_type = "question_structure"
            elif any(word in common_words for word in ["advice", "guidance", "recommend", "suggest"]):
                pattern_type = "advice_seeking"

            # Store for feedback
            self.recent_duplicates.append({
                "new_text": new_text[:100],
                "existing_text": existing_text[:100],
                "similarity": similarity,
                "pattern_type": pattern_type,
                "common_words": list(common_words)[:5]  # Top 5 common words
            })

            # Update pattern frequency
            self.session_stats["common_duplicate_patterns"][pattern_type] = \
                self.session_stats["common_duplicate_patterns"].get(pattern_type, 0) + 1

        # Keep only recent duplicates (last 50)
        if len(self.recent_duplicates) > 50:
            self.recent_duplicates = self.recent_duplicates[-50:]

    def filter_duplicates(self, new_texts: List[str],
                         new_data: List[dict] = None) -> Tuple[List[str], List[dict]]:
        """
        Filter out duplicate texts and optionally associated data.

        Args:
            new_texts: List of new text scenarios
            new_data: Optional list of data dictionaries corresponding to texts

        Returns:
            Tuple of (unique_texts, unique_data)
        """
        duplicate_indices, similarities = self.find_duplicates(new_texts)

        if not duplicate_indices:
            print("No semantic duplicates found!")
            return new_texts, new_data if new_data else []

        print(f"Found {len(duplicate_indices)} semantic duplicates")

        # Create set of duplicate indices for efficient lookup
        duplicate_set = set(duplicate_indices)

        # Filter texts
        unique_texts = [text for i, text in enumerate(new_texts)
                       if i not in duplicate_set]

        # Filter data if provided
        unique_data = []
        if new_data:
            unique_data = [data for i, data in enumerate(new_data)
                          if i not in duplicate_set]

        # Update cache with unique texts
        if unique_texts:
            self.update_existing_texts(unique_texts)

        print(f"Kept {len(unique_texts)} unique texts out of {len(new_texts)} total")
        return unique_texts, unique_data

    def get_statistics(self) -> dict:
        """Get statistics about the deduplication cache."""
        return {
            "total_cached_texts": len(self.existing_texts),
            "similarity_threshold": self.similarity_threshold,
            "model_name": self.model.get_sentence_embedding_dimension(),
            "cache_files_exist": {
                "embeddings": self.embeddings_cache_file.exists(),
                "texts": self.texts_cache_file.exists()
            },
            "session_stats": self.session_stats.copy(),
            "recent_duplicate_count": len(self.recent_duplicates)
        }

    def get_deduplication_feedback(self) -> str:
        """Generate feedback for improving scenario uniqueness."""
        if self.session_stats["total_processed"] == 0:
            return ""

        duplicate_rate = (self.session_stats["total_duplicates"] / self.session_stats["total_processed"]) * 100

        feedback_parts = []

        if duplicate_rate > 80:
            feedback_parts.append("CRITICAL: Very high duplication rate (80%+). Generate significantly more unique scenarios.")
        elif duplicate_rate > 50:
            feedback_parts.append("HIGH: High duplication rate (50%+). Focus on unique phrasing and contexts.")
        elif duplicate_rate > 20:
            feedback_parts.append("MODERATE: Some duplication detected. Vary sentence structures and topics.")

        # Analyze common patterns
        if self.session_stats["common_duplicate_patterns"]:
            patterns = self.session_stats["common_duplicate_patterns"]
            most_common = max(patterns.items(), key=lambda x: x[1])

            if most_common[0] == "question_structure":
                feedback_parts.append("Avoid repetitive question formats like 'How to...', 'What is...', 'Should I...'")
            elif most_common[0] == "advice_seeking":
                feedback_parts.append("Vary advice-seeking scenarios beyond 'recommend', 'suggest', 'guidance'")
            elif most_common[0] == "word_overlap":
                feedback_parts.append("Use more diverse vocabulary and contexts")

        # Recent duplicate analysis
        if len(self.recent_duplicates) > 5:
            # Find common words across recent duplicates
            all_common_words = []
            for dup in self.recent_duplicates[-10:]:  # Last 10 duplicates
                all_common_words.extend(dup.get("common_words", []))

            if all_common_words:
                word_freq = {}
                for word in all_common_words:
                    word_freq[word] = word_freq.get(word, 0) + 1

                overused_words = [word for word, count in word_freq.items() if count > 2]
                if overused_words:
                    feedback_parts.append(f"Avoid overused words: {', '.join(overused_words[:5])}")

        return " ".join(feedback_parts)

    def clear_cache(self):
        """Clear the embeddings cache."""
        if self.embeddings_cache_file.exists():
            self.embeddings_cache_file.unlink()
        if self.texts_cache_file.exists():
            self.texts_cache_file.unlink()

        self.existing_embeddings = np.array([])
        self.existing_texts = []
        print("Cache cleared")

    def adjust_threshold(self, new_threshold: float):
        """Adjust the similarity threshold."""
        self.similarity_threshold = new_threshold
        print(f"Similarity threshold updated to {new_threshold}")


class OpenAIDeduplicator:
    """
    Semantic deduplicator using OpenAI's text-embedding-3-large.
    Supports principle-specific deduplication for focused generation.
    """

    def __init__(self,
                 similarity_threshold: float = 0.50,
                 cache_dir: str = None,
                 principle: str = None,
                 dataset_path: str = None):
        """
        Initialize OpenAI-based deduplicator.

        Args:
            similarity_threshold: Cosine similarity threshold for duplicates (0-1)
            cache_dir: Directory to cache embeddings
            principle: Optional principle name for focused deduplication
            dataset_path: Path to dataset for loading existing principle scenarios
        """
        import os
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model_name = "text-embedding-3-large"
        self.similarity_threshold = similarity_threshold
        self.principle = principle

        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "embeddings_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache file paths - principle-specific if principle provided
        if principle:
            self.embeddings_cache_file = self.cache_dir / f"existing_embeddings_openai_{principle}.pkl"
            self.texts_cache_file = self.cache_dir / f"existing_texts_openai_{principle}.pkl"
        else:
            self.embeddings_cache_file = self.cache_dir / "existing_embeddings_openai.pkl"
            self.texts_cache_file = self.cache_dir / "existing_texts_openai.pkl"

        # Load existing embeddings and texts
        self.existing_embeddings, self.existing_texts = self._load_cache()

        # If principle-specific mode and no cache exists, load from dataset
        if principle and dataset_path and len(self.existing_embeddings) == 0:
            self._load_principle_scenarios_from_dataset(dataset_path)

        # Track recent deduplication patterns for feedback
        self.recent_duplicates = []
        self.session_stats = {
            "total_processed": 0,
            "total_duplicates": 0,
            "common_duplicate_patterns": {}
        }

    def _load_principle_scenarios_from_dataset(self, dataset_path: str):
        """Load and embed existing scenarios for the specified principle."""
        import json

        print(f"Loading existing scenarios for principle '{self.principle}' from dataset...")

        try:
            principle_scenarios = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Check metadata.principle (not top-level principle field)
                    if data.get('metadata', {}).get('principle') == self.principle:
                        principle_scenarios.append(data.get('input', ''))

            if principle_scenarios:
                print(f"Found {len(principle_scenarios)} existing scenarios for '{self.principle}'")
                self.update_existing_texts(principle_scenarios)
            else:
                print(f"No existing scenarios found for principle '{self.principle}'")
        except FileNotFoundError:
            print(f"Dataset not found at {dataset_path}, starting with empty cache")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def _load_cache(self) -> Tuple[np.ndarray, List[str]]:
        """Load cached embeddings and texts."""
        try:
            if self.embeddings_cache_file.exists() and self.texts_cache_file.exists():
                with open(self.embeddings_cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                with open(self.texts_cache_file, 'rb') as f:
                    texts = pickle.load(f)
                cache_type = f" for '{self.principle}'" if self.principle else ""
                print(f"Loaded {len(texts)} cached OpenAI embeddings{cache_type}")
                return embeddings, texts
        except Exception as e:
            print(f"Warning: Could not load cached embeddings: {e}")

        return np.array([]), []

    def _save_cache(self):
        """Save embeddings and texts to cache."""
        try:
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(self.existing_embeddings, f)
            with open(self.texts_cache_file, 'wb') as f:
                pickle.dump(self.existing_texts, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _get_openai_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Get embeddings from OpenAI API."""
        import time

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)

                # Rate limiting
                if i + batch_size < len(texts):
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
                raise

        return np.array(all_embeddings)

    def update_existing_texts(self, texts: List[str]):
        """Update the cache with new existing texts."""
        if not texts:
            return

        print(f"Computing OpenAI embeddings for {len(texts)} texts...")
        new_embeddings = self._get_openai_embeddings(texts)

        if len(self.existing_embeddings) == 0:
            self.existing_embeddings = new_embeddings
            self.existing_texts = texts
        else:
            # Append new embeddings
            self.existing_embeddings = np.vstack([self.existing_embeddings, new_embeddings])
            self.existing_texts.extend(texts)

        # Save to cache
        self._save_cache()
        cache_type = f" for '{self.principle}'" if self.principle else ""
        print(f"Cache updated with {len(self.existing_texts)} total texts{cache_type}")

    def find_duplicates(self, new_texts: List[str]) -> Tuple[List[int], List[float]]:
        """
        Find duplicates among new texts compared to existing texts.

        Args:
            new_texts: List of new text scenarios to check

        Returns:
            Tuple of (duplicate_indices, similarity_scores)
        """
        if not new_texts or len(self.existing_embeddings) == 0:
            return [], []

        print(f"Computing OpenAI embeddings for {len(new_texts)} new texts...")
        new_embeddings = self._get_openai_embeddings(new_texts)

        print("Computing semantic similarities...")
        duplicate_indices = []
        similarity_scores = []

        # Update session stats
        self.session_stats["total_processed"] += len(new_texts)

        for i, new_embedding in enumerate(new_embeddings):
            # Compute cosine similarity with all existing embeddings
            similarities = cosine_similarity([new_embedding], self.existing_embeddings)[0]
            max_similarity = np.max(similarities)

            if max_similarity >= self.similarity_threshold:
                duplicate_indices.append(i)
                similarity_scores.append(max_similarity)

                # Find the most similar existing text for logging
                most_similar_idx = np.argmax(similarities)
                most_similar_text = self.existing_texts[most_similar_idx]
                print(f"  Duplicate found (similarity: {max_similarity:.3f}):")
                print(f"    New: {new_texts[i][:100]}...")
                print(f"    Existing: {most_similar_text[:100]}...")

                # Store duplicate pattern for feedback analysis
                self._analyze_duplicate_pattern(new_texts[i], most_similar_text, max_similarity)

        # Update duplicate count
        self.session_stats["total_duplicates"] += len(duplicate_indices)

        return duplicate_indices, similarity_scores

    def _analyze_duplicate_pattern(self, new_text: str, existing_text: str, similarity: float):
        """Analyze and store duplicate patterns for feedback generation."""
        new_words = set(new_text.lower().split())
        existing_words = set(existing_text.lower().split())

        # Find common patterns
        common_words = new_words & existing_words
        if len(common_words) > 3:
            pattern_type = "word_overlap"
            if any(word in new_text.lower() and word in existing_text.lower()
                   for word in ["how to", "what is", "help me", "should i"]):
                pattern_type = "question_structure"
            elif any(word in common_words for word in ["advice", "guidance", "recommend", "suggest"]):
                pattern_type = "advice_seeking"

            self.recent_duplicates.append({
                "new_text": new_text[:100],
                "existing_text": existing_text[:100],
                "similarity": similarity,
                "pattern_type": pattern_type,
                "common_words": list(common_words)[:5]
            })

            self.session_stats["common_duplicate_patterns"][pattern_type] = \
                self.session_stats["common_duplicate_patterns"].get(pattern_type, 0) + 1

        # Keep only recent duplicates (last 50)
        if len(self.recent_duplicates) > 50:
            self.recent_duplicates = self.recent_duplicates[-50:]

    def filter_duplicates(self, new_texts: List[str],
                         new_data: List[dict] = None) -> Tuple[List[str], List[dict]]:
        """
        Filter out duplicate texts and optionally associated data.

        Args:
            new_texts: List of new text scenarios
            new_data: Optional list of data dictionaries corresponding to texts

        Returns:
            Tuple of (unique_texts, unique_data)
        """
        duplicate_indices, similarities = self.find_duplicates(new_texts)

        if not duplicate_indices:
            print("No semantic duplicates found!")
            return new_texts, new_data if new_data else []

        print(f"Found {len(duplicate_indices)} semantic duplicates")

        # Create set of duplicate indices for efficient lookup
        duplicate_set = set(duplicate_indices)

        # Filter texts
        unique_texts = [text for i, text in enumerate(new_texts)
                       if i not in duplicate_set]

        # Filter data if provided
        unique_data = []
        if new_data:
            unique_data = [data for i, data in enumerate(new_data)
                          if i not in duplicate_set]

        # Update cache with unique texts
        if unique_texts:
            self.update_existing_texts(unique_texts)

        print(f"Kept {len(unique_texts)} unique texts out of {len(new_texts)} total")
        return unique_texts, unique_data

    def get_statistics(self) -> dict:
        """Get statistics about the deduplication cache."""
        return {
            "total_cached_texts": len(self.existing_texts),
            "similarity_threshold": self.similarity_threshold,
            "model_name": self.model_name,
            "principle": self.principle,
            "cache_files_exist": {
                "embeddings": self.embeddings_cache_file.exists(),
                "texts": self.texts_cache_file.exists()
            },
            "session_stats": self.session_stats.copy(),
            "recent_duplicate_count": len(self.recent_duplicates)
        }

    def get_deduplication_feedback(self) -> str:
        """Generate feedback for improving scenario uniqueness."""
        if self.session_stats["total_processed"] == 0:
            return ""

        duplicate_rate = (self.session_stats["total_duplicates"] / self.session_stats["total_processed"]) * 100

        feedback_parts = []

        if duplicate_rate > 80:
            feedback_parts.append("CRITICAL: Very high duplication rate (80%+). Generate significantly more unique scenarios.")
        elif duplicate_rate > 50:
            feedback_parts.append("HIGH: High duplication rate (50%+). Focus on unique phrasing and contexts.")
        elif duplicate_rate > 20:
            feedback_parts.append("MODERATE: Some duplication detected. Vary sentence structures and topics.")

        # Analyze common patterns
        if self.session_stats["common_duplicate_patterns"]:
            patterns = self.session_stats["common_duplicate_patterns"]
            most_common = max(patterns.items(), key=lambda x: x[1])

            if most_common[0] == "question_structure":
                feedback_parts.append("Avoid repetitive question formats like 'How to...', 'What is...', 'Should I...'")
            elif most_common[0] == "advice_seeking":
                feedback_parts.append("Vary advice-seeking scenarios beyond 'recommend', 'suggest', 'guidance'")
            elif most_common[0] == "word_overlap":
                feedback_parts.append("Use more diverse vocabulary and contexts")

        # Recent duplicate analysis
        if len(self.recent_duplicates) > 5:
            all_common_words = []
            for dup in self.recent_duplicates[-10:]:
                all_common_words.extend(dup.get("common_words", []))

            if all_common_words:
                word_freq = {}
                for word in all_common_words:
                    word_freq[word] = word_freq.get(word, 0) + 1

                overused_words = [word for word, count in word_freq.items() if count > 2]
                if overused_words:
                    feedback_parts.append(f"Avoid overused words: {', '.join(overused_words[:5])}")

        return " ".join(feedback_parts)

    def clear_cache(self):
        """Clear the embeddings cache."""
        if self.embeddings_cache_file.exists():
            self.embeddings_cache_file.unlink()
        if self.texts_cache_file.exists():
            self.texts_cache_file.unlink()

        self.existing_embeddings = np.array([])
        self.existing_texts = []
        print("OpenAI cache cleared")

    def adjust_threshold(self, new_threshold: float):
        """Adjust the similarity threshold."""
        self.similarity_threshold = new_threshold
        print(f"Similarity threshold updated to {new_threshold}")