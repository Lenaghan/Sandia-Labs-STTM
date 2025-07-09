# Domain-Aware Phrase-Based Text Clustering

This project implements an unsupervised machine learning pipeline to discover and group thematic topics from unstructured text documents. It is particularly well-suited for domain-specific corpora, such as incident reports, product reviews, or customer feedback, where identifying key, recurring concepts is crucial.

The core of the project is a frequency-based clustering algorithm that first mines significant multi-word phrases and then uses a topic modeling approach to group documents. It enhances the discovery process by integrating a domain-specific dictionary (a gazetteer) to boost the importance of known, relevant terms.

## Key Features

- **Advanced Preprocessing**: Includes tokenization, stemming, stop-word removal, and optional Named Entity Recognition (NER) to clean text data effectively.
- **Phrase Mining**: Identifies frequent and significant multi-word phrases (n-grams), which often carry more meaning than single words.
- **Domain-Specific Boosting**: Utilizes a `Gazetteer` (domain dictionary) to increase the significance score of known important phrases (e.g., OSHA injury codes), guiding the model toward more relevant topics.
- **Hierarchical Topic Modeling**:
    1.  Assigns documents to initial, granular topics using a PhraseLDA-like model.
    2.  Performs hierarchical clustering (AGNES) on the topic distributions to group similar topics into final, interpretable meta-clusters.
- **Comprehensive Visualization**: Generates a suite of plots and tables to help users understand the results, including:
    - Word clouds for each topic.
    - Dendrograms of the topic hierarchy.
    - Bar charts for phrase significance and topic distribution.
    - Detailed tables of cluster compositions.
- **Model Evaluation**: Calculates intrinsic evaluation metrics, including **Topic Coherence** (`c_v`, `npmi`) and **Silhouette Score**, to help assess the quality of the discovered topics and clusters.
- **Persistent Results**: Saves the trained model, all visualizations, and a final CSV file mapping each document to its assigned cluster for further analysis.

## The Pipeline: How It Works

The `main.py` script orchestrates the entire process, which can be broken down into the following stages:

1.  **Data Loading**: The pipeline starts by loading all `.csv` files from the `data/` directory.
2.  **Domain Dictionary Initialization**: It loads domain-specific terms from a source file (e.g., OSHA codes) into the `DomainDictionary` to be used for scoring.
3.  **Preprocessing**: The text data is thoroughly cleaned. This step is cached to a file in the `outputs/` directory to speed up subsequent runs.
4.  **Clustering Execution**:
    - **Phrase Mining**: Extracts frequent phrases from the corpus.
    - **Phrase Scoring**: Scores the phrases, applying a boost factor to any that match entries in the domain dictionary.
    - **Topic Modeling**: Runs a constrained Gibbs sampling model where each document is assigned to a single topic based on its significant phrase content.
    - **Hierarchical Clustering**: Groups the resulting topics into a specified number of final, high-level clusters.
5.  **Saving Results**: The trained clusterer object and the final document labels are saved to a `.pkl` file in a timestamped directory within `outputs/`.
6.  **Evaluation & Visualization**: If the clustering is successful, the `ClusteringEvaluator` and `ClusteringVisualizer` classes are used to generate metrics and a full suite of plots.
7.  **Export**: The final cluster assignments (both ID and a descriptive phrase label) are merged back with the original data and saved as `claims_with_cluster_assignments.csv`.

## File Structure
```
├── data/
│   └── your_data.csv         # <-- Place your input CSV files here
├── outputs/
│   ├── preprocessed_data.json  # <-- Cached preprocessed text
│   └── 20230101_120000/        # <-- Example output folder for a run
│       ├── clustering_results.pkl
│       ├── claims_with_cluster_assignments.csv
│       ├── plot_document_length_distribution.png
│       └── ... (all other plots)
├── main.py                     # Main execution script
├── phrase_clustering.py        # Core clustering and topic modeling algorithm
├── preprocess.py               # Text preprocessing logic
├── gazetteer.py                # Domain dictionary management
├── visualizer.py               # Generates all plots and visualizations
├── evaluator.py                # Calculates evaluation metrics
└── requirements.txt            # Project dependencies
```
    
## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Lenaghan/Sandia-Labs-STTM.git](https://github.com/Lenaghan/Sandia-Labs-STTM.git)
    cd Sandia-Labs-STTM
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. See the Dependencies section below).*

4.  **Download NLTK data:**
    The first time you run the preprocessor, it will automatically download the necessary NLTK packages (`punkt`, `stopwords`, etc.).

## Usage

1.  **Place your data:** Put one or more CSV files into the `data/` directory. Ensure your files have columns for a unique identifier and the text to be clustered. You will need to update the column names in `main.py`:
    ```python
    # in main.py
    text_col_name = 'your_text_column_name'
    index_col_name = 'your_id_column_name'
    ```

2.  **Configure the domain dictionary:** If you are using a domain dictionary, update the file path and column names in the `run_dictionary_process` function in `main.py`.

3.  **Run the full pipeline:**
    Execute the `main.py` script from your terminal. This will run the entire process from preprocessing to visualization.
    ```bash
    python main.py
    ```

4.  **Visualize existing results:**
    If you have already run the pipeline and want to re-generate the visualizations without re-running the clustering, you can use the `--visualize-only` flag with the path to your results file.
    ```bash
    python main.py --visualize-only outputs/20230101_120000/clustering_results.pkl
    ```

## Dependencies

Create a `requirements.txt` file with the following contents:

```pandas
numpy
scikit-learn
nltk
scipy
matplotlib
seaborn
wordcloud
gensim
```

## Tuning the Model (Hyperparameters)

The performance and quality of the clusters are highly dependent on the chosen hyperparameters. You can adjust them in the `run_clustering_pipeline` function within the **`main.py`** file.

Here are the key parameters to tune:

```python
# In main.py, inside run_clustering_pipeline()

clusterer = FrequencyBasedClustering(
    topic_count=20,                 # <-- Number of initial, granular topics to discover.
    support_threshold=0.0001,       # <-- Minimum % of docs a phrase must appear in.
    significance_threshold=0.001,   # <-- Minimum score for a phrase to be used in modeling.
    phrase_boost_factor=2.0,        # <-- Multiplier for domain-specific phrases.
    n_iterations=150,               # <-- Number of iterations for the topic model.
    min_phrase_len=3,               # <-- Minimum words in a phrase (n-gram).
    max_phrase_len=5                # <-- Maximum words in a phrase (n-gram).
)

labels = clusterer.fit_predict(
    preprocessed_docs=preprocessed_tokens,
    domain_dictionary=domain_dict,
    n_final_clusters=8              # <-- The final number of high-level clusters to create.
)
```

-   **`topic_count`**: The number of initial, fine-grained topics the model will try to find. This should generally be larger than `n_final_clusters`. A good starting point is 2-4 times the expected number of final clusters.
-   **`n_final_clusters`**: The number of final, interpretable meta-clusters you want. The model groups the initial `topic_count` topics into this many final clusters.
-   **`support_threshold`**: Controls how common a phrase must be to be considered.
    -   *Decrease* this value to include rarer phrases.
    -   *Increase* it to be more strict and only use very common phrases.
-   **`significance_threshold`**: Filters phrases based on their score (a combination of frequency and domain boost).
    -   *Decrease* this to allow less significant phrases into the model, potentially creating more diverse but less distinct topics.
    -   *Increase* it to be very selective, focusing only on the most important phrases.
-   **`phrase_boost_factor`**: How much to prioritize phrases from your domain dictionary. A value of `2.0` means a matched phrase's score is doubled. Increase this if your domain terms are critical for defining topics.
-   **`n_iterations`**: The number of training iterations for the topic model. More iterations can lead to better convergence but will take longer. 150-200 is often sufficient.
-   **`min_phrase_len` / `max_phrase_len`**: Defines the size of the phrases (n-grams) to look for. The default of 3-to-5 words is a good starting point for many text types.

## Interpreting the Output

After a successful run, a new directory will be created in `outputs/` named with the run's timestamp. Inside, you will find:

-   **`clustering_results.pkl`**: A Python pickle file containing the fitted `FrequencyBasedClustering` model object and the final labels. This can be reloaded for further analysis or visualization.
-   **`claims_with_cluster_assignments.csv`**: Your original data with new columns added for the assigned parent cluster ID, parent cluster phrase label, child topic ID, and child topic phrase label.
-   **A series of `.png` images**: These are the visualizations generated by the `ClusteringVisualizer`, providing multiple ways to explore and understand the resulting clusters.
