import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings

# Suppress FutureWarning from transformers library
warnings.filterwarnings("ignore", category=FutureWarning)

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class KeywordExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.stop_words = set(stopwords.words('english'))

    def extract(self, text):
        # Tokenize and remove stop words
        words = word_tokenize(text.lower())  # Removed language parameter
        words = [word for word in words if word.isalpha() and word not in self.stop_words]

        # Return empty list if no valid words are found
        if not words:
            return []

        # Get embeddings for each word
        embeddings = []
        for word in words:
            inputs = self.tokenizer(word, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            # Take the mean of the token embeddings
            embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
            embeddings.append(embedding)

        # Convert list of embeddings to a 2D numpy array
        embeddings = np.vstack(embeddings)

        # Calculate similarity matrix
        similarity_matrix = np.inner(embeddings, embeddings)

        # Extract keywords based on centrality
        centrality = similarity_matrix.sum(axis=0)
        # Ensure we don't select more keywords than words available
        num_keywords = min(2, len(words))
        keywords = [words[idx] for idx in centrality.argsort()[-num_keywords:][::-1]]

        return keywords

if __name__ == '__main__':
    extractor = KeywordExtractor()
    test_title = "Oklahoma State Cowboys Score, Stats, and Highlights"
    keywords = extractor.extract(test_title)
    print(f"Extracted Keywords: {keywords}")
