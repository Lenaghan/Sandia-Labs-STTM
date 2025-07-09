import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import warnings
from typing import Any, List, Dict

class ClusteringEvaluator:
    """
    A class to evaluate the performance of a trained clustering model.
    It calculates topic coherence (C_v, NPMI) and cluster separation (Silhouette Score).
    """

    def __init__(self, clusterer: Any, final_labels: np.ndarray):
        """
        Initializes the ClusteringEvaluator.

        Args:
            clusterer (Any): The fitted FrequencyBasedClustering instance.
            final_labels (np.ndarray): The final cluster labels for each document.
        """
        if clusterer is None:
            raise ValueError("The 'clusterer' object cannot be None.")
            
        self.clusterer = clusterer
        self.final_labels = final_labels
        # The preprocessed documents are required for both coherence and vectorization
        self.preprocessed_docs = self.clusterer.preprocessed_docs
        if self.preprocessed_docs is None:
            raise ValueError("The 'preprocessed_docs' attribute is missing from the clusterer object.")

    def calculate_topic_coherence(self, top_n: int = 15, measures: List[str] = ['c_v', 'npmi']) -> Dict[str, float]:
        """
        Calculates topic coherence scores (e.g., C_v, NPMI).

        Args:
            top_n (int): The number of top phrases to consider for each topic.
            measures (List[str]): A list of coherence measures to calculate.

        Returns:
            A dictionary with coherence scores, e.g., {'c_v': 0.55, 'npmi': -0.05}.
        """
        print(f"\n--- Calculating Topic Coherence (Top {top_n} Phrases) ---")
        
        # 1. Extract top N phrases for each topic
        topics = []
        for i in range(self.clusterer.topic_count):
            # Get probabilities for topic i and sort to find top phrases
            phrase_probs = self.clusterer.phrase_topic_dist[:, i]
            top_phrase_indices = np.argsort(phrase_probs)[-top_n:][::-1]
            topic_phrases = [self.clusterer.id_to_phrase[idx] for idx in top_phrase_indices]
            topics.append(topic_phrases)

        # 2. Prepare corpus and dictionary for Gensim, treating phrases as single tokens
        # We replace spaces with underscores so "fall from height" becomes "fall_from_height"
        phrase_tokens = [phrase.replace(' ', '_') for phrase in self.clusterer.significant_phrases]
        gensim_docs = []
        for doc in self.preprocessed_docs:
            doc_str = " ".join(doc)
            # Replace significant phrases in the doc with their underscore versions
            for phrase in self.clusterer.significant_phrases:
                if phrase in doc_str:
                    doc_str = doc_str.replace(phrase, phrase.replace(' ', '_'))
            gensim_docs.append(doc_str.split())
        
        dictionary = Dictionary(gensim_docs)
        corpus = [dictionary.doc2bow(doc) for doc in gensim_docs]

        # 3. Calculate coherence for each specified measure
        coherence_scores = {}
        for measure in measures:
            try:
                cm = CoherenceModel(
                    topics=[[p.replace(' ', '_') for p in topic] for topic in topics],
                    texts=gensim_docs,
                    corpus=corpus,
                    dictionary=dictionary,
                    coherence=measure
                )
                score = cm.get_coherence()
                coherence_scores[measure] = score
                print(f"Coherence ({measure.upper()}): {score:.4f}")
            except Exception as e:
                print(f"Could not calculate coherence for '{measure}'. Reason: {e}")
        
        return coherence_scores

    def calculate_silhouette_score(self, sample_size: int = 2500, metric: str = 'cosine') -> float:
        """
        Calculates the Silhouette Score on a random sample of documents.

        Args:
            sample_size (int): The number of documents to sample for the calculation.
            metric (str): The distance metric to use (e.g., 'cosine', 'euclidean').

        Returns:
            The mean Silhouette Score for the sample.
        """
        print(f"\n--- Calculating Silhouette Score (Sample Size: {sample_size}) ---")
        num_docs = len(self.preprocessed_docs)
        
        # Ensure there are enough documents and clusters to calculate a score
        if np.unique(self.final_labels).size < 2:
            warnings.warn("Cannot calculate Silhouette Score with fewer than 2 clusters.")
            return np.nan
        
        if num_docs < sample_size:
            sample_size = num_docs  # Use all docs if sample size is too large

        # 1. Take a random sample
        sample_indices = np.random.choice(num_docs, sample_size, replace=False)
        sampled_docs_as_tokens = [self.preprocessed_docs[i] for i in sample_indices]
        sampled_labels = self.final_labels[sample_indices]
        
        # Join tokens into strings for the vectorizer
        sampled_docs_as_str = [" ".join(tokens) for tokens in sampled_docs_as_tokens]

        # 2. Vectorize docs using TF-IDF, with a vocabulary of only significant phrases
        vectorizer = TfidfVectorizer(vocabulary=self.clusterer.significant_phrases.keys())
        
        doc_vectors = vectorizer.fit_transform(sampled_docs_as_str)

        # 3. Calculate and return the score
        try:
            score = silhouette_score(doc_vectors, sampled_labels, metric=metric)
            print(f"Mean Silhouette Score: {score:.4f}")
            return score
        except ValueError as e:
            print(f"Could not calculate Silhouette Score. Reason: {e}")
            return np.nan
    

    def generate_exclusive_phrase_report(self, top_n: int = 10):
        """
        Generates a simplified hierarchy report where each phrase is uniquely
        assigned to the single topic where it has the highest probability,
        with no replacement.

        Args:
            top_n (int): The number of top phrases to display per topic in the report.
        Returns:
            A dictionary mapping topic IDs to lists of exclusively assigned phrases.
        """
        print("\n--- Generating Exclusive Phrase Assignment Report ---")
        
        # 1. Perform the exclusive assignment using a greedy approach
        probs_matrix = self.clusterer.phrase_topic_dist.copy()
        exclusive_assignments = {i: [] for i in range(self.clusterer.topic_count)}
        num_phrases = probs_matrix.shape[0]

        for _ in range(num_phrases):
            # Find the absolute max probability in the entire matrix
            max_prob = probs_matrix.max()
            if max_prob == 0:
                break # All remaining probabilities are zero

            # Find the location (phrase_id, topic_id) of this max value
            phrase_idx, topic_idx = np.unravel_index(np.argmax(probs_matrix), probs_matrix.shape)
            
            # Get the phrase text
            phrase_text = self.clusterer.id_to_phrase.get(phrase_idx, "N/A")
            
            # Assign the phrase to the topic
            exclusive_assignments[topic_idx].append(phrase_text)
            
            # "Remove" this phrase from consideration by zeroing out its row
            probs_matrix[phrase_idx, :] = 0

        # 2. Build and print the hierarchy table based on the new assignments
        meta_cluster_to_topics = {}
        for topic_idx, meta_label in enumerate(self.clusterer.topic_to_meta_label_map):
            if meta_label != -1:
                if meta_label not in meta_cluster_to_topics:
                    meta_cluster_to_topics[meta_label] = []
                meta_cluster_to_topics[meta_label].append(topic_idx)
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', 120)

        for meta_label, topic_list in sorted(meta_cluster_to_topics.items()):
            print("\n" + "="*80)
            print(f" SIMPLIFIED META-CLUSTER: {meta_label}")
            print("="*80)

            table_data = []
            for topic_idx in sorted(topic_list):
                # Get the uniquely assigned phrases for this topic
                assigned_phrases = exclusive_assignments.get(topic_idx, [])
                for rank, phrase_text in enumerate(assigned_phrases[:top_n]):
                    freq = self.clusterer.frequent_phrases.get(phrase_text, 0)
                    score = self.clusterer.significant_phrases.get(phrase_text, 0.0)
                    
                    table_data.append({
                        'Topic ID': topic_idx,
                        'Rank': rank + 1,
                        'Exclusively Assigned Phrase': phrase_text,
                        'Frequency': freq,
                        'Significance': f"{score:.3f}"
                    })
            
            if not table_data:
                print("No phrases were uniquely assigned to this meta-cluster.")
                continue

            df = pd.DataFrame(table_data)
            print(df.to_string(index=False))
        
        return exclusive_assignments
