import pandas as pd
import numpy as np
import glob
import os
import json
import pickle
import argparse
from datetime import datetime
from gazetteer import DomainDictionary
from preprocess import DocumentPreprocessor
from phrase_clustering import FrequencyBasedClustering
from visualizer import ClusteringVisualizer
from evaluator import ClusteringEvaluator

def load_csv_data():
    """
    Loads CSV data from a specified folder and returns the text and index columns.
    Returns:
    --------
    text_data_column: pd.Series
    index_data_column: pd.Series
    """
    folder_path = 'data'

    # Get a list of all csv files in the folder
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # Read and concatenate all files into a single DataFrame
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    return df

def run_dictionary_process():
    """
    Initializes and processes the domain dictionary.
    """
    # Define the file path and column names for your specific data
    file_path = "oiics-manual-version-3-02 event codes.csv"
    code_column = "injury_cause_code"
    description_column = "injury_cause"

    # 1. Initialize the dictionary
    print("Initializing the domain dictionary...")
    domain_dict = DomainDictionary()

    # 2. Load phrases from the specified CSV file
    print(f"Loading and cleaning phrases from '{file_path}'...")
    loaded_count = domain_dict.load_osha_codes(
        filepath=file_path,
        code_col=code_column,
        desc_col=description_column
    )

    if loaded_count > 0:
        print(f"Successfully loaded {loaded_count} phrases.")

        # 3. Build the stemmed index for flexible matching
        print("Building the stemmed index...")
        domain_dict.build_stemmed_index()
        print("Index built successfully.")

        # 4. Display final dictionary statistics
        print("\n---Dictionary Statistics ---")
        stats = domain_dict.get_statistics()
        print(json.dumps(stats, indent=2))
    else:
        print("Could not load any phrases. Please check the file path and format.")
    return domain_dict

def get_preprocessed_data(text_data_column, index_data_column, output_filepath):
    """
    Preprocesses data while retaining alignment with index and original text:
    Filters out documents that become empty after preprocessing.

    Returns:
    --------
    indices: list[str/int]
    original texts: list[str]
    preprocessed tokens: list[list[str]]
    """
    if text_data_column is None or index_data_column is None:
        print("Could not retrieve data. Please check the configuration and file.")
        return None, None, None
    
    print("Successfully Retrieved Corpus Data")
    print(f"Total rows: {len(index_data_column)}\n")
    
    # Convert pandas Series to lists
    documents = text_data_column.tolist()
    indices = index_data_column.tolist()
    
    preprocessor = DocumentPreprocessor(language='english')
    preprocessed_docs = preprocessor.preprocess(
        documents, 
        perform_ner=True,
        output_filepath=output_filepath
    )

    filtered_indices = []
    filtered_original_texts = []
    filtered_preprocessed_tokens = []

    # Filter out documents with empty token lists while maintaining alignment
    for i, tokens in enumerate(preprocessed_docs):
        if len(tokens) > 0:
            filtered_indices.append(indices[i])
            filtered_original_texts.append(documents[i])
            filtered_preprocessed_tokens.append(tokens)

    print(f"Preprocessing complete:")
    print(f"Original documents: {len(documents)}")
    print(f"Documents with tokens after preprocessing: {len(filtered_indices)}")
    print(f"Documents filtered out (empty after preprocessing): {len(documents) - len(filtered_indices)}")

    return filtered_indices, filtered_original_texts, filtered_preprocessed_tokens

def run_clustering_pipeline(preprocessed_tokens, domain_dict):

    best_model = None
    best_log_likelihood = -np.inf  # Start with negative infinity
    best_labels = None
    num_runs = 1
    print(f"--- Starting {num_runs} clustering runs to find the best model ---")

    for i in range(num_runs):
        print(f"\n--- Run {i + 1}/{num_runs} ---")

        # Each run needs a new instance to have a different random start
        clusterer = FrequencyBasedClustering(
            topic_count=20,
            support_threshold=0.0005,
            significance_threshold=0.1,
            alpha=0.01,
            beta=0.1,
            phrase_boost_factor=2.0,
            n_iterations=150,
            min_phrase_len=3,
            max_phrase_len=5
        )

        # Fit the clusterer to the preprocessed documents and domain dictionary
        labels = clusterer.fit_predict(
            preprocessed_docs=preprocessed_tokens,
            domain_dictionary=domain_dict,
            n_final_clusters=8
        )

        # Get the final log-likelihood for comparison
        final_log_likelihood = clusterer.last_log_likelihood

        print(f"Run {i + 1} finished with Log-Likelihood: {final_log_likelihood}")

        if final_log_likelihood > best_log_likelihood:
            print(f"Found a new best model in Run {i + 1}!")
            best_log_likelihood = final_log_likelihood
            best_model = clusterer
            best_labels = labels

    print("\n--- Best Model Selection Complete ---")
    print(f"Highest Log-Likelihood Found: {best_log_likelihood}")
    print(f"Number of unique labels: {len(set(best_labels))}")
    print(f"Labels: {best_labels}")

    if best_model is not None:
        print("Best model's significant phrases:")
        for phrase in best_model.significant_phrases:
            print(phrase)
    else:
        print("No valid model was found.")

    return best_model, best_labels

def save_clustering_results(clusterer, labels, output_dir='outputs'):
    """
    Saves the fitted clusterer object and final labels to a .pkl file.

    Args:
        clusterer (FrequencyBasedClustering): The fitted clusterer instance.
        labels (np.ndarray): The final document labels.
        output_dir (str): The directory to save the output file.
    """
    # Create a filename
    filename = 'clustering_results.pkl'
    filepath = os.path.join(output_dir, filename)

    # Bundle the model and labels together for saving
    data_to_save = {
        'cluster_model': clusterer,
        'final_labels': labels
    }

    print(f"\n--- Saving clustering results to '{filepath}' ---")
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        print("Save complete.")
    except (IOError, pickle.PicklingError) as e:
        print(f"Error: Could not save results. Reason: {e}")

def load_clustering_results(filepath):
    """
    Loads a saved clusterer object and labels from a .pkl file.

    Args:
        filepath (str): The path to the .pkl file.

    Returns:
        A tuple of (cluster_model, final_labels) or (None, None) if loading fails.
    """
    print(f"\n--- Loading clustering results from '{filepath}' ---")
    try:
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)
        print("Load complete.")
        return loaded_data['cluster_model'], loaded_data['final_labels']
    except (FileNotFoundError, IOError, pickle.UnpicklingError, KeyError) as e:
        print(f"Error: Could not load results. Reason: {e}")
        return None, None

def export_assignments_to_csv(original_df, doc_topic_assignments, final_labels, filtered_indices, index_col_name, output_dir, child_topic_labels, parent_cluster_labels):
    """
    Merges cluster assignments back to the original dataframe and saves as a CSV.

    Args:
        original_df (pd.DataFrame): The original, unmodified dataframe.
        doc_topic_assignments (np.ndarray): The child topic ID for each document.
        final_labels (np.ndarray): The final parent cluster ID for each document.
        filtered_indices (list): The original indices of documents that were clustered.
        index_col_name (str): The name of the column used for merging (e.g., 'claim_number').
        output_dir (str): The directory to save the output file.
        child_topic_labels (dict): A map from child topic ID to a phrase label.
        parent_cluster_labels (dict): A map from parent cluster ID to a phrase label.
    """
    print(f"\n--- Appending phrase-based assignments to original dataset ---")
    
    # Create pandas Series from the raw assignment arrays
    child_series = pd.Series(doc_topic_assignments)
    parent_series = pd.Series(final_labels)

    # Map the numeric IDs to their corresponding phrase labels
    child_phrases = child_series.map(child_topic_labels).fillna('N/A')
    parent_phrases = parent_series.map(parent_cluster_labels).fillna('N/A')

    # Create a new dataframe with both IDs and phrase labels for clarity
    assignments_data = {
        index_col_name: filtered_indices,
        'child_topic_id': doc_topic_assignments,
        'child_topic_phrase': child_phrases,
        'parent_cluster_id': final_labels,
        'parent_cluster_phrase': parent_phrases
    }
    assignments_df = pd.DataFrame(assignments_data)

    # Merge with the original dataframe using a left join
    merged_df = pd.merge(original_df, assignments_df, on=index_col_name, how='left')

    # Save the final dataset to a CSV file
    filepath = os.path.join(output_dir, 'claims_with_cluster_assignments.csv')
    try:
        merged_df.to_csv(filepath, index=False)
        print(f"Successfully saved final dataset to '{filepath}'")
    except IOError as e:
        print(f"Error: Could not save final dataset. Reason: {e}")

if __name__ == "__main__":

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Run the frequency-based clustering pipeline or visualize existing results."
    )
    parser.add_argument(
        '--visualize-only',
        type=str,
        metavar='FILEPATH',
        help="Path to a .pkl file to load for visualization, skipping the clustering run."
    )
    args = parser.parse_args()

    # Define the main outputs directory and the static path for preprocessed data
    main_outputs_dir = 'outputs'
    os.makedirs(main_outputs_dir, exist_ok=True)
    preprocessed_filepath = os.path.join(main_outputs_dir, 'preprocessed_data.json')

    if args.visualize_only:
        cluster_model, final_labels = load_clustering_results(args.visualize_only)
        if not cluster_model:
            print("Aborting due to loading failure.")
            exit()
        output_dir = os.path.dirname(args.visualize_only)
    else:
        domain_dict = run_dictionary_process()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(main_outputs_dir, timestamp)
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"Created output directory for this run: {run_output_dir}")
        output_dir = run_output_dir

        # Define key column names for the dataset globally
        text_col_name = 'claim_description'
        index_col_name = 'claim_number'

        # Check for the centrally-located preprocessed file
        if os.path.exists(preprocessed_filepath):
            print(f"Found existing preprocessed file at '{preprocessed_filepath}'. Loading data...")
            with open(preprocessed_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            indices = data['indices']
            original_texts = data['original_texts']
            preprocessed_tokens = data['preprocessed_tokens']
            print("Data loaded successfully.")
            
            # Load the original dataframe for the final export step
            original_df = load_csv_data()
        else:
            print(f"No preprocessed file found at '{preprocessed_filepath}'. Running preprocessing...")
            original_df = load_csv_data()
            
            indices, original_texts, preprocessed_tokens = get_preprocessed_data(
                original_df[text_col_name], original_df[index_col_name], output_filepath=run_output_dir
        )
            # Save the aligned data for future runs
            print(f"Saving aligned preprocessed data to '{preprocessed_filepath}'...")
            data_to_save = {
                'indices': indices,
                'original_texts': original_texts,
                'preprocessed_tokens': preprocessed_tokens
            }
            try:
                with open(preprocessed_filepath, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f)
                    print("Save complete.")
            except (IOError, TypeError) as e:
                    print(f"Error: Could not save file. Reason: {e}")

            if indices is not None:
                print("\nSample preprocessed results:")
                for i in range(min(3, len(indices))):
                    print(f"\nIndex: {indices[i]}")
                    print(f"Original: {original_texts[i][:100]}...")
                    print(f"Preprocessed tokens: {preprocessed_tokens[i][:10]}...")

        cluster_model, final_labels = run_clustering_pipeline(preprocessed_tokens, domain_dict)

        if cluster_model and indices:
            # Generate Labels for Export
            print("\n--- Generating Phrase Labels for Export ---")
            evaluator = ClusteringEvaluator(cluster_model, final_labels)
            visualizer = ClusteringVisualizer(cluster_model, output_dir=output_dir)
            
            # Get exclusive phrase assignments from the evaluator
            exclusive_assignments = evaluator.generate_exclusive_phrase_report(top_n=1)
            
            # Create a label for each parent meta-cluster
            parent_cluster_labels = visualizer.get_meta_cluster_labels(exclusive_assignments)
            
            # Create a label for each child topic from the top exclusive phrase
            child_topic_labels = {
                topic_id: phrases[0] for topic_id, phrases in exclusive_assignments.items() if phrases
            }
            print("Labels generated.")

            # Save Model and Export CSV with Phrase Labels
            save_clustering_results(cluster_model, final_labels, output_dir=run_output_dir)
            export_assignments_to_csv(
                original_df=original_df,
                doc_topic_assignments=cluster_model.doc_topic_assignments,
                final_labels=final_labels,
                filtered_indices=indices,
                index_col_name=index_col_name,
                output_dir=run_output_dir,
                child_topic_labels=child_topic_labels,
                parent_cluster_labels=parent_cluster_labels
            )

    if cluster_model:
        # Re-initialize evaluator and visualizer if they weren't created in the 'else' block
        if 'visualizer' not in locals():
            visualizer = ClusteringVisualizer(cluster_model, output_dir=output_dir)
        if 'evaluator' not in locals():
            evaluator = ClusteringEvaluator(cluster_model, final_labels)
            
        print("\n--- Generating Visualizations ---")
        visualizer.plot_document_length_distribution()
        visualizer.plot_top_phrases(n_phrases=15, use_significant=False) # Frequent phrases
        visualizer.plot_top_phrases(n_phrases=15, use_significant=True) # Significant phrases
        visualizer.plot_topic_word_clouds(n_cols=4)
        visualizer.plot_topic_distribution()
        visualizer.plot_topic_hierarchy_dendrogram()
        visualizer.plot_final_cluster_distribution(final_doc_labels=final_labels)
        visualizer.display_cluster_hierarchy_table(final_doc_labels=final_labels, top_n_phrases=10)

        print("\n--- MODEL EVALUATION METRICS ---")
        evaluator.calculate_topic_coherence(top_n=15)
        evaluator.calculate_silhouette_score(sample_size=2500, metric='cosine')
        
        # We need the exclusive assignments and labels for the final pie chart
        if 'exclusive_assignments' not in locals():
            exclusive_assignments = evaluator.generate_exclusive_phrase_report(top_n=10)        
        if 'parent_cluster_labels' not in locals():
             parent_cluster_labels = visualizer.get_meta_cluster_labels(exclusive_assignments)

        print("\n--- Generating Final Cluster Distribution Chart ---")
        visualizer.plot_final_cluster_distribution(final_doc_labels=final_labels, custom_labels=parent_cluster_labels)
        print("\n" + "="*80)

    else:
        print("\nNo valid model to visualize.")