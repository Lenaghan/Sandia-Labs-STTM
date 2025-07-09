import os
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.cluster import hierarchy
from typing import Any, List, Tuple

# Set a consistent style for all plots
sns.set_theme(style="whitegrid")

class ClusteringVisualizer:
    """
    A class dedicated to visualizing the results of a FrequencyBasedClustering model.

    This class is decoupled from the clustering logic and operates on a fitted
    instance of a clustering object (e.g., FrequencyBasedClustering). It provides
    methods to generate various plots to analyze the clustering results.
    """

    def __init__(self, clusterer: Any, output_dir: str):
        """
        Initializes the ClusteringVisualizer.

        Args:
            clusterer (Any): A fitted instance of a clustering class, such as
                             FrequencyBasedClustering. This object is expected
                             to have specific attributes containing the clustering
                             results (e.g., preprocessed_docs, significant_phrases).
            output_dir (str): The path to the directory where all plots will be saved.

        Raises:
            ValueError: If the provided clusterer object is None.
        """
        if clusterer is None:
            raise ValueError("The 'clusterer' object cannot be None. Please provide a fitted instance.")
        self.clusterer = clusterer
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_map = {
            'preprocessed_docs': 'Document word counts',
            'significant_phrases': 'Significant Phrases',
            'frequent_phrases': 'Frequent Phrases',
            'phrase_topic_dist': 'Phrase-Topic Distribution',
            'id_to_phrase': 'Phrase ID Mapping',
            'doc_topic_assignments': 'Document-Topic Assignments'
        }

    def check_data_availability(self, attr_name: str) -> bool:
        """Helper method to check for data availability and log a warning if missing."""
        data = getattr(self.clusterer, attr_name, None)
        if data is None or (hasattr(data, '__len__') and len(data) == 0):
            display_name = self.data_map.get(attr_name, attr_name)
            warnings.warn(
                f"Warning: Required data '{display_name}' not found or is empty in the clusterer object."
                f"Skipping plot generation."
            )
            return False
        return True

    def plot_document_length_distribution(self, bins: int = 50) -> None:
        """
        Generates and displays a histogram of document lengths.

        This plot helps visualize the distribution of word counts across all
        documents in the corpus, which can inform text preprocessing decisions.

        Args:
            bins (int): The number of bins to use for the histogram.
        """
        if not self.check_data_availability('preprocessed_docs'):
            return

        doc_lengths = [len(doc) for doc in self.clusterer.preprocessed_docs]

        plt.figure(figsize=(12, 6))
        sns.histplot(doc_lengths, bins=bins, kde=True)
        plt.title('Document Length Distribution', fontsize=16)
        plt.xlabel('Word Count per Document', fontsize=12)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        filepath = os.path.join(self.output_dir, 'plot_document_length_distribution.png')
        plt.savefig(filepath)
        plt.show(block=False)
        plt.pause(5)
        plt.close('all')

    def plot_top_phrases(self, n_phrases: int = 20, use_significant: bool = True) -> None:
        """
        Generates a horizontal bar chart of the top N phrases.

        This chart shows the most important phrases based on their frequency or
        significance score, providing insight into the key terms of the corpus.

        Args:
            n_phrases (int): The number of top phrases to display.
            use_significant (bool): If True, plots significant phrases and their
                                    scores. If False, plots frequent phrases and
                                    their counts.
        """
        attr_name = 'significant_phrases' if use_significant else 'frequent_phrases'
        if not self.check_data_availability(attr_name):
            return

        phrases_data = getattr(self.clusterer, attr_name)

        # Sort by score and take the top N
        top_phrases = sorted(phrases_data.items(), key=lambda item: item[1], reverse=True)[:n_phrases]

        if not top_phrases:
            warnings.warn("Warning: No phrases data to plot.")
            return

        phrases, scores = zip(*top_phrases)

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=list(scores), y=list(phrases), palette='viridis', orient='h', legend=False, hue=list(phrases))

        title = f'Top {n_phrases} Significant Phrases' if use_significant else f'Top {n_phrases} Frequent Phrases'
        xlabel = 'Significance Score' if use_significant else 'Frequency Count'

        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Phrases', fontsize=12)
        # Invert y-axis to have the highest score on top
        ax.invert_yaxis()
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f'{attr_name}.png')
        plt.savefig(filepath)
        plt.show(block=False)
        plt.pause(5)
        plt.close('all')

    def plot_topic_word_clouds(self, n_cols: int = 4) -> None:
        """
        Generates a grid of word clouds, one for each discovered topic.

        Each word cloud provides an intuitive visual summary of the theme of a
        latent topic, with word size proportional to its importance in that topic.

        Args:
            n_cols (int): The number of columns in the grid of word clouds.
        """
        if not self.check_data_availability('phrase_topic_dist') or \
           not self.check_data_availability('id_to_phrase'):
            return

        topic_dist = self.clusterer.phrase_topic_dist
        id_to_phrase = self.clusterer.id_to_phrase
        num_topics = topic_dist.shape[1]

        n_rows = (num_topics + n_cols - 1) // n_cols  # Calculate rows needed

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten() # Flatten to 1D array for easy iteration

        for i in range(num_topics):
            # Create a frequency dictionary for the current topic
            # topic_dist[:, i] gives probabilities for all phrases in topic i
            topic_probabilities = topic_dist[:, i]
            word_freq = {id_to_phrase[phrase_id]: prob for phrase_id, prob in enumerate(topic_probabilities)}

            wordcloud = WordCloud(
                width=400, height=300, background_color='white',
                colormap='plasma', max_words=50
            ).generate_from_frequencies(word_freq)

            ax = axes[i]
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Topic {i+1}', fontsize=14)
            ax.axis('off')

        # Hide any unused subplots
        for j in range(num_topics, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=3.0)
        filepath = os.path.join(self.output_dir, 'plot_topic_word_clouds.png')
        plt.savefig(filepath)
        plt.show(block=False)
        plt.pause(5)
        plt.close('all')

    def plot_topic_distribution(self, top_n_phrases: int = 2) -> None:
        """
        Generates a horizontal bar chart showing the number of documents per topic
        for better label readability.
        """
        if not self.check_data_availability('doc_topic_assignments') or \
           not self.check_data_availability('id_to_phrase'):
            return

        plt.figure(figsize=(14, 10))

        # 1. Pre-calculate topic IDs and their document counts
        topic_ids, doc_counts = np.unique(self.clusterer.doc_topic_assignments, return_counts=True)
        
        # 2. Generate text labels for the y-axis
        topic_labels = []
        for topic_id in topic_ids:
            # Get top phrases for the current topic
            top_phrase_indices = np.argsort(self.clusterer.phrase_topic_dist[:, topic_id])[-top_n_phrases:][::-1]
            top_phrases = [self.clusterer.id_to_phrase.get(idx, "") for idx in top_phrase_indices]
            label_text = ", ".join(filter(None, top_phrases))
            topic_labels.append(f"Topic {topic_id}: {label_text}")

        # 3. Use a horizontal barplot for better readability
        ax = sns.barplot(
            x=doc_counts,
            y=topic_labels,
            orient='h',
            palette='magma',
            hue=topic_labels,
            legend=False
        )

        plt.title('Document Distribution Across Topics', fontsize=16)
        plt.xlabel('Number of Documents', fontsize=12)
        plt.ylabel('Topic (Represented by Top Phrases)', fontsize=12)
        plt.tight_layout(pad=1.0)
        
        filepath = os.path.join(self.output_dir, 'plot_topic_distribution.png')
        plt.savefig(filepath)
        plt.show(block=False)
        plt.pause(5)
        plt.close('all')

    def plot_topic_hierarchy_dendrogram(self, top_n_phrases: int = 3) -> None:
        """
        Generates a dendrogram to show the hierarchical merging of topics.

        This plot visually represents how the granular topics are related and can
        be merged into higher-level meta-clusters based on their similarity.
        """
        if not self.check_data_availability('phrase_topic_dist') or \
           not self.check_data_availability('id_to_phrase'):
            return

        # We cluster topics, which are columns in the phrase-topic distribution matrix.
        # Therefore, we need to transpose the matrix.
        topic_vectors = self.clusterer.phrase_topic_dist.T

        if topic_vectors.shape[0] < 2:
            warnings.warn("Warning: Cannot generate dendrogram with fewer than 2 topics.")
            return
        
        num_topics = topic_vectors.shape[0]
        topic_labels = []
        for i in range(num_topics):
            # Get the indices of the top N most probable phrases for topic i
            top_phrase_indices = np.argsort(self.clusterer.phrase_topic_dist[:, i])[-top_n_phrases:][::-1]
            # Convert indices to phrase text
            top_phrases = [self.clusterer.id_to_phrase.get(idx, "") for idx in top_phrase_indices]
            # Join phrases into a single string for the label
            label_text = ", ".join(filter(None, top_phrases))
            topic_labels.append(label_text)

        # Perform hierarchical clustering (AGNES)
        linkage_matrix = hierarchy.linkage(topic_vectors, method='ward')

        plt.figure(figsize=(15, 12))
        plt.title('Hierarchical Clustering of Topics (Dendrogram)', fontsize=16)
        plt.ylabel('Euclidean Distance', fontsize=12)

        hierarchy.dendrogram(
            linkage_matrix,
            labels=topic_labels,
            orientation='right',
            leaf_font_size=10.
        )
        plt.tight_layout(pad=1.0)

        filepath = os.path.join(self.output_dir, 'plot_topic_hierarchy_dendrogram.png')
        plt.savefig(filepath)
        plt.show(block=False)
        plt.pause(5)
        plt.close('all')

    def display_cluster_hierarchy_table(self, final_doc_labels: np.ndarray, top_n_phrases: int = 10):
        """
        Generates and prints a set of tables detailing the cluster hierarchy.

        For each final meta-cluster, it shows the constituent topics and the
        top N most representative phrases for each topic, along with their
        frequency counts and significance scores.

        Args:
            final_doc_labels (np.ndarray): The final cluster label for each document.
            top_n_phrases (int): The number of top phrases to display for each topic.
        """
        # Check for required data from the clusterer object
        required_attrs = [
            'topic_to_meta_label_map', 'phrase_topic_dist', 'id_to_phrase',
            'frequent_phrases', 'significant_phrases'
        ]
        for attr in required_attrs:
            if not self.check_data_availability(attr):
                return
    
        # --- Build the Hierarchy Data Structure ---
        # 1. Reverse the map to get {meta_cluster: [topic1, topic2, ...]}
        meta_cluster_to_topics = {}
        for topic_idx, meta_label in enumerate(self.clusterer.topic_to_meta_label_map):
            if meta_label != -1:  # Ignore unassigned topics
                if meta_label not in meta_cluster_to_topics:
                    meta_cluster_to_topics[meta_label] = []
                meta_cluster_to_topics[meta_label].append(topic_idx)

        # 2. Get document counts for each meta-cluster
        unique_labels, doc_counts = np.unique(final_doc_labels, return_counts=True)
        meta_cluster_doc_counts = dict(zip(unique_labels, doc_counts))

        # --- Generate and Print a Table for Each Meta-Cluster ---
        # Set pandas display options for better readability
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', 120)

        for meta_label, topic_list in sorted(meta_cluster_to_topics.items()):
            doc_count = meta_cluster_doc_counts.get(meta_label, 0)
            print("\n" + "="*80)
            print(f" META-CLUSTER: {meta_label}  (Documents: {doc_count})")
            print("="*80)

            table_data = []
            for topic_idx in sorted(topic_list):
                # Get top N phrases for this topic based on probability
                phrase_probs = self.clusterer.phrase_topic_dist[:, topic_idx]
                top_phrase_indices = np.argsort(phrase_probs)[-top_n_phrases:][::-1]

                for rank, phrase_id in enumerate(top_phrase_indices):
                    phrase_text = self.clusterer.id_to_phrase.get(phrase_id, "N/A")
                    freq = self.clusterer.frequent_phrases.get(phrase_text, 0)
                    score = self.clusterer.significant_phrases.get(phrase_text, 0.0)
                    
                    table_data.append({
                        'Topic ID': topic_idx,
                        'Rank': rank + 1,
                        'Phrase': phrase_text,
                        'Frequency': freq,
                        'Significance': f"{score:.3f}"
                    })
            
            if not table_data:
                print("No phrases found for this meta-cluster.")
                continue

            df = pd.DataFrame(table_data)
            print(df.to_string(index=False))

    def get_meta_cluster_labels(self, exclusive_assignments: dict) -> dict:
        """
        Determines the single best representative phrase for each meta-cluster
        based on the highest significance score.

        Args:
            exclusive_assignments (Dict): The output from the evaluator's report.

        Returns:
            A dictionary mapping meta-cluster IDs to a single phrase label.
        """
        meta_cluster_labels = {}
        # Reverse the topic->meta_cluster map to get {meta_label: [topic_ids]}
        meta_to_topics = {}
        for topic_idx, meta_label in enumerate(self.clusterer.topic_to_meta_label_map):
            if meta_label not in meta_to_topics:
                meta_to_topics[meta_label] = []
            meta_to_topics[meta_label].append(topic_idx)

        for meta_label, topic_list in meta_to_topics.items():
            best_phrase_for_cluster = ""
            max_score = -1.0

            # Find the most significant phrase among all phrases in this meta-cluster
            for topic_idx in topic_list:
                for phrase in exclusive_assignments.get(topic_idx, []):
                    score = self.clusterer.significant_phrases.get(phrase, 0.0)
                    if score > max_score:
                        max_score = score
                        best_phrase_for_cluster = phrase
            
            if best_phrase_for_cluster:
                meta_cluster_labels[meta_label] = best_phrase_for_cluster
                
        return meta_cluster_labels

    def plot_final_cluster_distribution(self, final_doc_labels: np.ndarray, custom_labels: dict = None) -> None:
        """
        Generates a pie chart showing the final distribution of documents.

        This chart displays the final count and proportion of documents assigned
        to each meta-cluster after the hierarchical clustering is cut.

        Args:
            final_doc_labels (np.ndarray): An array where each element is the
                                           final cluster label for a document.
        """
        unique_labels_from_data, counts = np.unique(final_doc_labels, return_counts=True)

        if custom_labels:
            # Filter data to only include clusters that have a custom label
            filtered_labels, filtered_counts = [], []
            chart_labels = []
            for i, cluster_id in enumerate(unique_labels_from_data):
                if cluster_id in custom_labels:
                    filtered_labels.append(cluster_id)
                    filtered_counts.append(counts[i])
                    chart_labels.append(custom_labels[cluster_id])
            counts = filtered_counts
        else:
            chart_labels = [f'Cluster {label}' for label in unique_labels_from_data]

        if not list(counts):
            warnings.warn("No data to plot in final cluster distribution after filtering.")
            return

        plt.figure(figsize=(12, 10))
        plt.pie(
            counts, 
            labels=chart_labels, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=sns.color_palette('pastel')
        )
        plt.title('Final Document Cluster Distribution', fontsize=16)
        plt.ylabel('')
        plt.axis('equal')
        
        filepath = os.path.join(self.output_dir, 'plot_final_cluster_distribution.png')
        plt.savefig(filepath)
        plt.show(block=False)
        plt.pause(5)
        plt.close('all')