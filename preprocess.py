import time
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

class DocumentPreprocessor:
    def __init__(self, language='english'):
        """
        Initialize the DocumentPreprocessor with a specified language.

        Parameters:
        -----------
        language : str
            Language for stopwords and stemming (default is 'english').
        """
        self.language = language
        self.stemmer = SnowballStemmer(language)
        unstemmed_stopwords = set(stopwords.words(language))
        custom_stopwords = {'nec', 'data'}
        unstemmed_stopwords.update(custom_stopwords)
        self.stemmed_stop_words = {self.stemmer.stem(word) for word in unstemmed_stopwords}

        # Ensure NLTK resources are downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)

    def preprocess(self, documents, perform_ner: bool = False, output_filepath: str = None):
        """
        Preprocess the documents: tokenize, lowercase, remove named entities,
        remove stopwords/punctuation, and stem.

        Parameters:
        -----------
        documents : list[str]
            List of document strings.
        perform_ner : bool
            If True, performs expensive NER to remove entities. Defaults to False.
        output_filepath : str, optional
            If provided, saves the preprocessed documents to this JSON file path.

        Returns:
        --------
        list[list[str]]
            List of preprocessed documents, where each document is a list of tokens.
        """
        preprocessed_docs = []
        if not isinstance(documents, list):
             raise TypeError("Input 'documents' must be a list of strings.")

        print("Starting preprocessing...")
        start_time = time.time()

        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                print(f"Warning: Document at index {i} is not a string, skipping.")
                continue

            # Tokenize the original document to preserve case for NER
            tokens = word_tokenize(doc)

            # Identify and Collect Named Entity Tokens
            named_entity_tokens = set()
            if perform_ner:
                tagged_tokens = nltk.pos_tag(tokens)
                tree = nltk.ne_chunk(tagged_tokens, binary=True)
                for chunk in tree:
                    # If chunk is a named entity chunk
                    if hasattr(chunk, 'label') and chunk.label() == 'NE':
                        for leaf in chunk.leaves():
                            # Add the token part of the leaf (word, tag) to the set
                            named_entity_tokens.add(leaf[0].lower())

            # Filter tokens: lowercase, stem, remove NE's, keep only alphabetic and remove stopwords
            filtered_tokens = []

            for token in tokens:
                token_lower = token.lower()

                # Remove if it's a named entity
                if token_lower in named_entity_tokens:
                    continue

                # Keep only alphabetic tokens
                if not token_lower.isalpha():
                    continue

                # Stem the token and check against stemmed stopwords
                stemmed_token = self.stemmer.stem(token_lower)
                if stemmed_token in self.stemmed_stop_words:
                    continue

                filtered_tokens.append(stemmed_token)

            preprocessed_docs.append(filtered_tokens)

        end_time = time.time()
        print(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")

        # --- Save the output if a filepath is provided ---
        if output_filepath:
            print(f"Saving preprocessed data to {output_filepath}...")
            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(preprocessed_docs, f)
                print("Save complete.")
            except (IOError, TypeError) as e:
                print(f"Error: Could not save file. Reason: {e}")

        return preprocessed_docs