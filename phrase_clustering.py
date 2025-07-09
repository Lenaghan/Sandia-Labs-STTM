import numpy as np
import logging
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
from nltk.util import ngrams
from nltk.stem import SnowballStemmer

# --- Basic Configuration for Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Forward declaration for type hinting ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gazetteer import DomainDictionary

class FrequencyBasedClustering:
    """
    Implements a Frequency-Based Clustering Algorithm.
    """
    def __init__(self,
                 topic_count: int,
                 support_threshold: float = 0.01,
                 significance_threshold: float = 0.5,
                 alpha: float = 0.1,
                 beta: float = 0.01,
                 phrase_boost_factor: float = 1.5,
                 min_phrase_len: int = 3,
                 max_phrase_len: int = 5,
                 n_iterations: int = 1000,
                 language: str = 'english'):
        # --- Hyperparameters ---
        self.topic_count = topic_count
        self.support_threshold = support_threshold
        self.significance_threshold = significance_threshold
        self.alpha = alpha
        self.beta = beta
        self.phrase_boost_factor = phrase_boost_factor
        self.min_phrase_len = min_phrase_len
        self.max_phrase_len = max_phrase_len
        self.n_iterations = n_iterations

        # --- Internal State ---
        self.preprocessed_docs = None
        self.doc_count = 0
        self.frequent_phrases = None
        self.significant_phrases = None
        self.phrase_to_id = None
        self.id_to_phrase = None
        self.doc_topic_assignments = None
        self.phrase_topic_dist = None
        self.last_log_likelihood = None
        self.stemmer = SnowballStemmer(language)
        self.topic_to_meta_label_map = None

    def mine_phrases(self):
        """
        Extracts frequent phrases from pre-tokenized documents.
        """
        logging.info("Phase 1: Starting frequent phrase mining.")
        phrase_counts = defaultdict(int)

        for tokens in self.preprocessed_docs:
            seen_in_doc = set()
            # Use a sliding window to find candidate phrases
            for n in range(self.min_phrase_len, self.max_phrase_len + 1):
                for ngram_tuple in ngrams(tokens, n):
                    phrase = " ".join(ngram_tuple)
                    if phrase not in seen_in_doc:
                        phrase_counts[phrase] += 1
                        seen_in_doc.add(phrase)

        logging.info(f"Discovered {len(phrase_counts)} total unique phrases before filtering.")

        # Filter by support threshold
        min_support_count = self.doc_count * self.support_threshold
        self.frequent_phrases = {
            phrase: count for phrase, count in phrase_counts.items()
            if count >= min_support_count
        }
        logging.info(f"Found {len(self.frequent_phrases)} frequent phrases after applying support threshold.")

    def score_phrases(self, domain_dictionary: 'DomainDictionary' = None):
        """
        Calculates significance scores and filters phrases. Boosts scores for
        phrases matching the provided DomainDictionary instance.
        """
        logging.info("Phase 2: Scoring and filtering phrases.")
        if not self.frequent_phrases:
            logging.warning("No frequent phrases found. Skipping scoring.")
            self.significant_phrases = {}
            return

        max_count = max(self.frequent_phrases.values()) if self.frequent_phrases else 1
        scores = {p: c / max_count for p, c in self.frequent_phrases.items()}

        # Boost scores for domain-matched phrases using full and partial matching
        if domain_dictionary and hasattr(domain_dictionary, 'stemmed_lookup'):
            logging.info(f"Boosting scores with a factor of up to {self.phrase_boost_factor} using domain dictionary.")
            boosted_count = 0
            domain_stemmed_words = domain_dictionary.all_stemmed_words

            for phrase in scores:
                stemmed_phrase_tuple = tuple(self.stemmer.stem(word) for word in phrase.split())

                # Full match: provides the maximum boost
                if stemmed_phrase_tuple in domain_dictionary.stemmed_lookup:
                    scores[phrase] *= self.phrase_boost_factor
                    boosted_count += 1
                # Partial match: provides a smaller, proportional boost
                else:
                    match_count = sum(1 for word in stemmed_phrase_tuple if word in domain_stemmed_words)
                    if match_count > 0:
                        phrase_len = len(stemmed_phrase_tuple)
                        # Calculate a boost proportional to the number of matched words
                        partial_boost = 1.0 + (self.phrase_boost_factor - 1.0) * (match_count / phrase_len)
                        scores[phrase] *= partial_boost
                        boosted_count += 1

            logging.info(f"Applied a full or partial boost to {boosted_count} phrases.")

        # Filter by significance threshold
        self.significant_phrases = {
            phrase: score for phrase, score in scores.items()
            if score >= self.significance_threshold
        }
        logging.info(f"Retained {len(self.significant_phrases)} significant phrases after filtering.")

    def model_topics(self):
        """
        Implements PhraseLDA with a single topic per document constraint.
        """
        logging.info("Phase 3: Starting constrained topic modeling (PhraseLDA).")
        if not self.significant_phrases:
            logging.error("No significant phrases to model. Aborting.")
            return

        sorted_phrases = sorted(list(self.significant_phrases.keys()), key=len, reverse=True)
        self.phrase_to_id = {phrase: i for i, phrase in enumerate(sorted_phrases)}
        self.id_to_phrase = {i: phrase for phrase, i in self.phrase_to_id.items()}
        vocab_size = len(self.significant_phrases)

        doc_indices = []
        phrase_indices = []

        for i, doc_tokens in enumerate(self.preprocessed_docs):
            doc_str = " " + " ".join(doc_tokens) + " "
            for phrase in sorted_phrases:
                if f" {phrase} " in doc_str:
                    doc_indices.append(i)
                    phrase_indices.append(self.phrase_to_id[phrase])

        phrase_doc_matrix = csr_matrix((np.ones(len(doc_indices)), (doc_indices, phrase_indices)),
                                       shape=(self.doc_count, vocab_size))

        doc_topic_assignments = np.random.randint(0, self.topic_count, self.doc_count)
        doc_topic_counts = np.zeros((self.doc_count, self.topic_count))
        for i, topic in enumerate(doc_topic_assignments):
            doc_topic_counts[i, topic] = 1

        phrase_topic_counts = np.zeros((vocab_size, self.topic_count))
        topic_totals = np.zeros(self.topic_count)

        for doc_idx, phrase_idx in zip(*phrase_doc_matrix.nonzero()):
            topic = doc_topic_assignments[doc_idx]
            phrase_topic_counts[phrase_idx, topic] += 1
            topic_totals[topic] += 1

        logging.info(f"Running Gibbs sampling for {self.n_iterations} iterations.")

        for i in range(self.n_iterations):
            for d_idx in range(self.doc_count):
                old_topic = doc_topic_assignments[d_idx]
                doc_topic_counts[d_idx, old_topic] = 0
                phrases_in_doc = phrase_doc_matrix[d_idx, :].nonzero()[1]
                for p_idx in phrases_in_doc:
                    phrase_topic_counts[p_idx, old_topic] -= 1
                    topic_totals[old_topic] -= 1

                # Calculate phrase probabilities for all topics at once
                log_prob_phrases_vec = np.sum(np.log(phrase_topic_counts[phrases_in_doc, :] + self.beta), axis=0)

                # Calculate the full log probability vector for all topics
                log_probs = (np.log(doc_topic_counts.sum(axis=0) + self.alpha) +
                            log_prob_phrases_vec -
                            len(phrases_in_doc) * np.log(topic_totals + vocab_size * self.beta))

                probs = np.exp(log_probs - np.max(log_probs))
                new_topic = np.random.choice(self.topic_count, p=probs / probs.sum())

                doc_topic_assignments[d_idx] = new_topic
                doc_topic_counts[d_idx, new_topic] = 1
                for p_idx in phrases_in_doc:
                    phrase_topic_counts[p_idx, new_topic] += 1
                    topic_totals[new_topic] += 1

            # Convergence Monitoring
            if (i + 1) % 10 == 0:
                log_likelihood = self.calculate_log_likelihood(
                    phrase_doc_matrix,
                    phrase_topic_counts,
                    topic_totals,
                    doc_topic_assignments
                )
                logging.info(f"Iteration {i+1}/{self.n_iterations}, Log-Likelihood: {log_likelihood:.2f}")

        # Calculate and store the final log-likelihood after the loop
        self.last_log_likelihood = self.calculate_log_likelihood(
            phrase_doc_matrix,
            phrase_topic_counts,
            topic_totals,
            doc_topic_assignments
        )
        logging.info(f"Final Log-Likelihood: {self.last_log_likelihood:.2f}")

        # Store final assignments and distributions
        self.doc_topic_assignments = doc_topic_assignments
        self.phrase_topic_dist = (phrase_topic_counts + self.beta) / (topic_totals + vocab_size * self.beta)
        logging.info("Phase 3 complete.")

    def cluster_topics(self, n_final_clusters: int):
        """
        Applies AGNES clustering on the topic-phrase distributions to create a
        hierarchy of topics (meta-clusters). Dynamically handles unused topics.
        """
        logging.info("Phase 4: Applying AGNES to create topic hierarchy.")
        if  self.phrase_topic_dist is None or self.doc_topic_assignments is None:
            logging.error("Distributions not available. Cannot cluster topics.")
            return None

        # Identify which topics were actually used
        active_topic_indices = np.unique(self.doc_topic_assignments)
        num_active_topics = len(active_topic_indices)

        if num_active_topics == 0:
            logging.warning("No active topics found after modeling. Cannot create hierarchy.")
            return None

        logging.info(f"Found {num_active_topics} active topics out of {self.topic_count}.")

        # Adjust n_final_clusters if it's too high
        if n_final_clusters > num_active_topics:
            logging.warning(
                f"Requested {n_final_clusters} final clusters, but only {num_active_topics} topics are active. "
                f"Reducing n_final_clusters to {num_active_topics}."
            )
            n_final_clusters = num_active_topics

        # Cluster only the vectors of the active topics
        topic_vectors = self.phrase_topic_dist.T
        active_topic_vectors = topic_vectors[active_topic_indices, :]

        agnes = AgglomerativeClustering(
            n_clusters=n_final_clusters,
            metric='cosine',
            linkage='average'
        )
        # These labels correspond to the active topics
        active_topic_meta_labels = agnes.fit_predict(active_topic_vectors)

        # Create a map from original topic index to final meta-cluster label
        topic_to_meta_label_map = np.full(self.topic_count, -1, dtype=int) # Initialize with -1 for unused topics
        for i, original_topic_idx in enumerate(active_topic_indices):
            topic_to_meta_label_map[original_topic_idx] = active_topic_meta_labels[i]

        logging.info(f"Topic hierarchy created. Grouped {num_active_topics} active topics into {n_final_clusters} meta-clusters.")
        return topic_to_meta_label_map

    # --- Intrinsic Metrics & Helpers ---
    def calculate_log_likelihood(self, phrase_doc_matrix, phrase_topic_counts, topic_totals, doc_topic_assignments):
        vocab_size = len(self.significant_phrases)
        phi = (phrase_topic_counts + self.beta) / (topic_totals + vocab_size * self.beta)
        log_likelihood = 0
        for d, p in zip(*phrase_doc_matrix.nonzero()):
            topic = doc_topic_assignments[d]
            if phi[p, topic] > 0:
                log_likelihood += np.log(phi[p, topic])
        return log_likelihood

    def validate_against_dictionary(self, labels: np.ndarray, domain_dictionary: 'DomainDictionary') -> dict:
        """Validates cluster purity against a DomainDictionary instance."""
        validation_results = {}
        domain_phrases_set = set(domain_dictionary.phrases.keys())

        for cluster_id in np.unique(labels):
            doc_indices = np.where(labels == cluster_id)[0]
            cluster_phrases = set()
            for doc_idx in doc_indices:
                # Reconstruct phrases from tokens to check for presence
                for n in range(self.min_phrase_len, self.max_phrase_len + 1):
                    for ngram_tuple in ngrams(self.preprocessed_docs[doc_idx], n):
                        phrase = " ".join(ngram_tuple)
                        if phrase in self.significant_phrases:
                             cluster_phrases.add(phrase)

            matched_phrases = cluster_phrases.intersection(domain_phrases_set)
            purity = len(matched_phrases) / len(cluster_phrases) if cluster_phrases else 0
            validation_results[cluster_id] = {
                "purity": purity,
                "matched_count": len(matched_phrases),
                "total_phrases": len(cluster_phrases)
            }
        return validation_results

    def log_corpus_statistics(self):
        """Calculates and logs key statistics about the input corpus."""
        logging.info("--- Corpus Statistics ---")
        if not self.preprocessed_docs or self.doc_count == 0:
            logging.info("Corpus is empty or not provided.")
            logging.info("-------------------------")
            return
        doc_lengths = [len(doc) for doc in self.preprocessed_docs]
        total_word_count = sum(doc_lengths)
        mean_wc = np.mean(doc_lengths)
        median_wc = np.median(doc_lengths)
        min_wc = np.min(doc_lengths)
        max_wc = np.max(doc_lengths)

        try:
            # most_common returns a list of (element, count) tuples
            mode_wc = Counter(doc_lengths).most_common(1)[0][0]
        except IndexError:
            mode_wc = "N/A" # Handle case of empty doc_lengths

        logging.info(f"Total Documents: {self.doc_count}")
        logging.info(f"Total Words (Tokens): {total_word_count}")
        logging.info("Per-Document Word Count:")
        logging.info(f"  - Mean:   {mean_wc:.2f}")
        logging.info(f"  - Median: {int(median_wc)}")
        logging.info(f"  - Mode:   {mode_wc}")
        logging.info(f"  - Min:    {min_wc}")
        logging.info(f"  - Max:    {max_wc}")
        logging.info("-------------------------")

    def fit_predict(self,
                    preprocessed_docs: list[list[str]],
                    domain_dictionary: 'DomainDictionary' = None,
                    n_final_clusters: int = None):
        """
        Executes the full pipeline from phrase mining to hierarchical clustering of topics.

        Args:
            preprocessed_docs (list[list[str]]): The input corpus as a list of tokenized documents.
            domain_dictionary (DomainDictionary): Optional instance of a domain dictionary for boosting.
            n_final_clusters (int): The desired number of final clusters from AGNES.

        Returns:
            np.ndarray: An array of cluster labels for each document.
        """
        if n_final_clusters is None:
            n_final_clusters = self.topic_count

        # --- Data Setup & Analysis ---
        self.preprocessed_docs = preprocessed_docs
        self.doc_count = len(preprocessed_docs)
        self.log_corpus_statistics()

        # --- Execute Pipeline ---
        self.mine_phrases()
        self.score_phrases(domain_dictionary)
        self.model_topics()

        if self.doc_topic_assignments is None:
            logging.error("Pipeline failed during topic modeling phase. Returning None.")
            return None

        topic_to_meta_label_map = self.cluster_topics(n_final_clusters)
        if topic_to_meta_label_map is None:
            logging.error("Pipeline failed during topic clustering phase. Returning None.")
            return None

        # Map the topic assignment of each document to its final meta-cluster label
        self.topic_to_meta_label_map = topic_to_meta_label_map
        final_doc_labels = np.array([topic_to_meta_label_map[topic_idx] for topic_idx in self.doc_topic_assignments])

        return final_doc_labels