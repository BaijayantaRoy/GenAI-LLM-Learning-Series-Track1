# Step 1: Download Work of Rabindranath Tagore and Preprocess Text
# Libraries for downloading and preprocessing
import os
import requests
import zipfile
from bs4 import BeautifulSoup

def download_and_preprocess_text(url, local_file):
    """
    Download text from a URL, save it locally, and preprocess it by extracting plain text from HTML.
    Args:
        url (str): URL to download the text from.
        local_file (str): Local file path to save the downloaded content.
    Returns:
        str: Preprocessed plain text.
    """
    # Check if the file already exists locally to avoid redundant downloads
    if not os.path.exists(local_file):
        print("Downloading text file...")
        response = requests.get(url)
        with open(local_file, "wb") as file:
            file.write(response.content)
        print(f"Text file downloaded and saved as {local_file}.")
    else:
        print(f"Text file {local_file} already exists.")

    # Read the HTML file and extract plain text using BeautifulSoup
    with open(local_file, "r", encoding="utf-8", errors="replace") as file:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text()
    return text

    # Sample sentence for demonstration
    sample_sentence = "She plays the piano well."
    print(f"\nSample Sentence: {sample_sentence}")

    # Step 1: Download and preprocess text dataset
    print("\n=== Step 1: Downloading and Preprocessing Text ===")
    url = "https://www.gutenberg.org/files/33525/33525-h/33525-h.htm"
    local_file = "gutenberg_text.html"
    text = download_and_preprocess_text(url, local_file)
    print(f"Sample of Preprocessed Text:\n{text[:1000]}...")

# Step 2: Train BPE Tokenizer and Tokenize Text
# Libraries for tokenization
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(text):
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the given text and tokenize the text.
    BPE is a subword tokenization method that merges frequent character pairs.
    Args:
        text (str): Input text to train the tokenizer on.
    Returns:
        tuple: (tokenizer, tokens) - Trained tokenizer and list of tokens.
    """
    # Initialize a BPE tokenizer with an unknown token for out-of-vocabulary words
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=500, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()

    # Split text into sentences for training
    sentences = text.splitlines()
    tokenizer.train_from_iterator(sentences, trainer)

    # Tokenize the full text
    tokens = tokenizer.encode(text).tokens
    print(f"\nTotal Tokens Processed from Dataset: {len(tokens)}")
    return tokenizer, tokens

    # Execute Step 2: Train BPE tokenizer and tokenize text
    print("\n=== Step 2: Training BPE Tokenizer and Tokenizing Text ===")
    tokenizer, tokens = train_tokenizer(text)

    # Execute Step 2: demonstrate with the sample sentence
    sample_tokens = tokenizer.encode(sample_sentence).tokens
    print(f"Sample Sentence Tokens: {sample_tokens}")

# Step 3: Download and Load GloVe Embeddings for Static Token Embeddings
def download_glove_embeddings():
    """
    Download GloVe embeddings if not already present locally.
    GloVe embeddings are pre-trained static embeddings used for token representation.
    """
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    glove_file = "glove.6B.50d.txt"

    if not os.path.exists(glove_file):
        print("GloVe file not found locally, downloading...")
        glove_zip = "glove.6B.zip"
        response = requests.get(glove_url)
        with open(glove_zip, "wb") as file:
            file.write(response.content)
        with zipfile.ZipFile(glove_zip, "r") as zip_ref:
            zip_ref.extract("glove.6B.50d.txt")
        print("GloVe file downloaded and extracted.")
    else:
        print("GloVe file already exists locally.")

# Import numpy for numerical operations (used in embedding loading)
import numpy as np

def load_glove_embeddings(glove_file):
    """
    Load GloVe embeddings from a file into a dictionary mapping tokens to their embeddings.
    Args:
        glove_file (str): Path to the GloVe embedding file.
    Returns:
        dict: Dictionary mapping tokens to their GloVe embeddings.
    """
    glove_embeddings = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]  # First element is the token
            vector = np.array(values[1:], dtype=np.float32)  # Rest are embedding values
            glove_embeddings[word] = vector
    return glove_embeddings

    # Step 3: Download and load GloVe embeddings for static token embeddings
    print("\n=== Step 3: Downloading and Loading GloVe Embeddings ===")
    download_glove_embeddings()
    glove_embeddings = load_glove_embeddings("glove.6B.50d.txt")
    print(f"Total GloVe Embeddings Loaded: {len(glove_embeddings)}")

# Step 4: Convert Tokens to Static GloVe Embeddings
def get_embedding(token, glove_embeddings, embedding_dim):
    """
    Fetch the GloVe embedding for a token, or return a random embedding if not found.
    Args:
        token (str): Token to look up.
        glove_embeddings (dict): Dictionary of GloVe embeddings.
        embedding_dim (int): Dimensionality of the embeddings.
    Returns:
        np.ndarray: Embedding vector for the token.
    """
    return glove_embeddings.get(token.lower(), np.random.rand(embedding_dim))

def convert_tokens_to_embeddings(tokens, glove_embeddings, embedding_dim):
    """
    Convert a list of tokens to their corresponding GloVe embeddings.
    Args:
        tokens (list): List of tokens.
        glove_embeddings (dict): Dictionary of GloVe embeddings.
        embedding_dim (int): Dimensionality of the embeddings.
    Returns:
        list: List of embeddings for the tokens.
    """
    embeddings = [get_embedding(token, glove_embeddings, embedding_dim) for token in tokens]
    return embeddings

    # Execution Step 4: Convert tokens to static GloVe embeddings
    print("\n=== Step 4: Converting Tokens to Static GloVe Embeddings ===")

    # Define constants for training
    CONTEXT_LENGTH = 10  # Number of tokens per training sequence
    BATCH_SIZE = 4  # Mini-batch size for processing
    EMBEDDING_DIM = 50  # GloVe 50-dimensional embeddings

    # First, demonstrate with the sample sentence
    sample_tokens = tokenizer.encode(sample_sentence).tokens
    print(f"Sample Sentence Tokens: {sample_tokens}")

    static_sample_embeddings = convert_tokens_to_embeddings(sample_tokens, glove_embeddings, EMBEDDING_DIM)
    print("Static Embeddings for Sample Sentence:")
    for token, emb in zip(sample_tokens, static_sample_embeddings):
        print(f"Token: {token}, Static Embedding: {emb[:6]} ...")  # Display first 6 values in the 50 dimension

    # Execution Step 4:  Apply to the dataset
    static_token_embeddings = convert_tokens_to_embeddings(tokens, glove_embeddings, EMBEDDING_DIM)
    print(f"\nStatic Embeddings for Dataset (First 5 Tokens):")
    for token, emb in zip(tokens[:5], static_token_embeddings[:5]):
        print(f"Token: {token}, Static Embedding: {emb[:6]} ...")

# Step 5: Add Positional Embeddings
def positional_encoding(max_len, d_model):
    """
    Compute positional encodings using sine and cosine functions for each position in a sequence.
    Positional encodings add order information to embeddings.
    Args:
        max_len (int): Maximum sequence length.
        d_model (int): Embedding dimensionality.
    Returns:
        np.ndarray: Positional encoding matrix of shape (max_len, d_model).
    """
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

    # Execution Step 5: Add positional embeddings to create combined embeddings
    print("\n=== Step 5: Adding Positional Embeddings ===")
    # First, demonstrate with the sample sentence
    position_embeddings_sample = positional_encoding(len(sample_tokens), EMBEDDING_DIM)
    print("Positional Embeddings for Sample Sentence:")
    for pos, emb in enumerate(position_embeddings_sample):
        print(f"Position {pos}: {emb[:5]} ...")

# Step 6: Add Positional Embeddings with Static embedding to create Combined Embedding
def add_positional_embeddings(token_embeddings, embedding_dim):
    """
    Add positional embeddings to token embeddings to create combined embeddings.
    Args:
        token_embeddings (list): List of token embeddings.
        embedding_dim (int): Dimensionality of the embeddings.
    Returns:
        list: List of combined embeddings (token + positional).
    """
    position_embeddings = positional_encoding(len(token_embeddings), embedding_dim)
    return [token_embeddings[i] + position_embeddings[i] for i in range(len(token_embeddings))]

    # Execution Step 6: Add Positional Embeddings with Static embedding to create Combined Embedding
    combined_sample_embeddings = add_positional_embeddings(static_sample_embeddings, EMBEDDING_DIM)
    print("\nCombined Embeddings (Static + Positional) for Sample Sentence:")
    for token, emb in zip(sample_tokens, combined_sample_embeddings):
        print(f"Token: {token}, Combined Embedding: {emb[:5]} ...")

    # Apply to the dataset (Combined Embeddings)
    combined_embeddings = add_positional_embeddings(static_token_embeddings, EMBEDDING_DIM)
    print(f"\nCombined Embeddings for Dataset (First 5 Tokens):")
    for token, emb in zip(tokens[:5], combined_embeddings[:5]):
        print(f"Token: {token}, Combined Embedding: {emb[:5]} ...")



# In[1]:


# Step 1: Download Work of Rabindranath Tagore and Preprocess Text
# Libraries for downloading and preprocessing
import os
import requests
import zipfile
from bs4 import BeautifulSoup

def download_and_preprocess_text(url, local_file):
    """
    Download text from a URL, save it locally, and preprocess it by extracting plain text from HTML.
    This step prepares the raw text dataset for further processing like tokenization.
    Args:
        url (str): URL to download the text from (e.g., Project Gutenberg).
        local_file (str): Local file path to save the downloaded content.
    Returns:
        str: Preprocessed plain text extracted from the HTML.
    """
    # Check if the file already exists locally to avoid redundant downloads
    if not os.path.exists(local_file):
        print("Downloading text file...")
        response = requests.get(url)
        with open(local_file, "wb") as file:
            file.write(response.content)
        print(f"Text file downloaded and saved as {local_file}.")
    else:
        print(f"Text file {local_file} already exists.")

    # Read the HTML file and extract plain text using BeautifulSoup
    with open(local_file, "r", encoding="utf-8", errors="replace") as file:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text()
    return text

# Step 2: Train BPE Tokenizer and Tokenize Text
# Libraries for tokenization
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(text):
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the given text and tokenize the text.
    BPE is a subword tokenization method that merges frequent character pairs to create a vocabulary.
    This step converts raw text into a list of tokens for embedding generation.
    Args:
        text (str): Input text to train the tokenizer on.
    Returns:
        tuple: (tokenizer, tokens) - Trained tokenizer and list of tokens.
    """
    # Initialize a BPE tokenizer with an unknown token for out-of-vocabulary words
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=500, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()

    # Split text into sentences for training the tokenizer
    sentences = text.splitlines()
    tokenizer.train_from_iterator(sentences, trainer)

    # Tokenize the full text into a list of tokens
    tokens = tokenizer.encode(text).tokens
    print(f"\nTotal Tokens Processed from Dataset: {len(tokens)}")
    return tokenizer, tokens

# Step 3: Download and Load GloVe Embeddings for Static Token Embeddings
# Import numpy for numerical operations (used in embedding loading)
import numpy as np

def download_glove_embeddings():
    """
    Download GloVe embeddings if not already present locally.
    GloVe embeddings are pre-trained static embeddings used for token representation.
    These embeddings map words to fixed vectors in a high-dimensional space capturing semantic meaning.
    """
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    glove_file = "glove.6B.50d.txt"

    if not os.path.exists(glove_file):
        print("GloVe file not found locally, downloading...")
        glove_zip = "glove.6B.zip"
        response = requests.get(glove_url)
        with open(glove_zip, "wb") as file:
            file.write(response.content)
        with zipfile.ZipFile(glove_zip, "r") as zip_ref:
            zip_ref.extract("glove.6B.50d.txt")
        print("GloVe file downloaded and extracted.")
    else:
        print("GloVe file already exists locally.")

def load_glove_embeddings(glove_file):
    """
    Load GloVe embeddings from a file into a dictionary mapping tokens to their embeddings.
    Args:
        glove_file (str): Path to the GloVe embedding file (e.g., glove.6B.50d.txt).
    Returns:
        dict: Dictionary mapping tokens to their GloVe embeddings (e.g., "cat" → [0.123, -0.456, ...]).
    """
    glove_embeddings = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]  # First element is the token
            vector = np.array(values[1:], dtype=np.float32)  # Rest are embedding values
            glove_embeddings[word] = vector
    return glove_embeddings

# Step 4: Convert Tokens to Static GloVe Embeddings
def get_embedding(token, glove_embeddings, embedding_dim):
    """
    Fetch the GloVe embedding for a token, or return a random embedding if not found.
    Args:
        token (str): Token to look up (e.g., "she").
        glove_embeddings (dict): Dictionary of GloVe embeddings.
        embedding_dim (int): Dimensionality of the embeddings (e.g., 50 for GloVe 50D).
    Returns:
        np.ndarray: Embedding vector for the token (e.g., [0.123, -0.456, ...]).
    """
    # Return the GloVe embedding if the token exists, otherwise generate a random embedding
    return glove_embeddings.get(token.lower(), np.random.rand(embedding_dim))

def convert_tokens_to_embeddings(tokens, glove_embeddings, embedding_dim):
    """
    Convert a list of tokens to their corresponding GloVe embeddings.
    Args:
        tokens (list): List of tokens (e.g., ["She", "plays", ...]).
        glove_embeddings (dict): Dictionary of GloVe embeddings.
        embedding_dim (int): Dimensionality of the embeddings.
    Returns:
        list: List of embeddings for the tokens (e.g., [[0.123, -0.456, ...], ...]).
    """
    embeddings = [get_embedding(token, glove_embeddings, embedding_dim) for token in tokens]
    return embeddings

# Step 5: Generate Positional Embeddings
def positional_encoding(max_len, d_model):
    """
    Compute positional encodings using sine and cosine functions for each position in a sequence.
    Positional encodings add order information to embeddings, ensuring the model understands token positions.
    Args:
        max_len (int): Maximum sequence length (e.g., number of tokens in the sequence).
        d_model (int): Embedding dimensionality (e.g., 50 to match GloVe).
    Returns:
        np.ndarray: Positional encoding matrix of shape (max_len, d_model).
    """
    # Initialize a matrix of zeros for positional encodings
    PE = np.zeros((max_len, d_model))
    # Compute sine and cosine values for each position and dimension
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

# Step 6: Add Positional Embeddings to Static Embeddings to Create Combined Embeddings
def add_positional_embeddings(token_embeddings, embedding_dim):
    """
    Add positional embeddings to token embeddings to create combined embeddings.
    Combined embeddings encode both semantic meaning (static embeddings) and positional information.
    Args:
        token_embeddings (list): List of token embeddings (e.g., GloVe embeddings).
        embedding_dim (int): Dimensionality of the embeddings.
    Returns:
        list: List of combined embeddings (token + positional).
    """
    # Generate positional embeddings for the sequence length
    position_embeddings = positional_encoding(len(token_embeddings), embedding_dim)
    # Add positional embeddings to static embeddings element-wise
    return [token_embeddings[i] + position_embeddings[i] for i in range(len(token_embeddings))]

# Main execution with detailed steps
if __name__ == "__main__":
    # Define constants for training
    CONTEXT_LENGTH = 10  # Number of tokens per training sequence (used for context window in later steps)
    BATCH_SIZE = 4  # Mini-batch size for processing (used in batch processing)
    EMBEDDING_DIM = 50  # GloVe 50-dimensional embeddings

    # Sample sentence for demonstration
    sample_sentence = "She plays the piano well."
    print(f"\nSample Sentence: {sample_sentence}")

    # Step 1: Download and preprocess text dataset
    print("\n=== Step 1: Downloading and Preprocessing Text ===")
    # Define the URL and local file path for downloading Rabindranath Tagore's work
    url = "https://www.gutenberg.org/files/33525/33525-h/33525-h.htm"
    local_file = "gutenberg_text.html"
    # Download and preprocess the text to extract plain text
    text = download_and_preprocess_text(url, local_file)
    print(f"Sample of Preprocessed Text:\n{text[:1000]}...")

    # Step 2: Train BPE tokenizer and tokenize text
    print("\n=== Step 2: Training BPE Tokenizer and Tokenizing Text ===")
    # Train the tokenizer on the dataset and tokenize the entire text
    tokenizer, tokens = train_tokenizer(text)

    # Demonstrate tokenization with the sample sentence
    sample_tokens = tokenizer.encode(sample_sentence).tokens
    print(f"Sample Sentence Tokens: {sample_tokens}")

    # Step 3: Download and load GloVe embeddings for static token embeddings
    print("\n=== Step 3: Downloading and Loading GloVe Embeddings ===")
    # Download GloVe embeddings if not present and load them into a dictionary
    download_glove_embeddings()
    glove_embeddings = load_glove_embeddings("glove.6B.50d.txt")
    print(f"Total GloVe Embeddings Loaded: {len(glove_embeddings)}")

    # Step 4: Convert tokens to static GloVe embeddings
    print("\n=== Step 4: Converting Tokens to Static GloVe Embeddings ===")
    # First, demonstrate with the sample sentence
    static_sample_embeddings = convert_tokens_to_embeddings(sample_tokens, glove_embeddings, EMBEDDING_DIM)
    print("Static Embeddings for Sample Sentence:")
    for token, emb in zip(sample_tokens, static_sample_embeddings):
        print(f"Token: {token}, Static Embedding: {emb[:6]} ...")  # Display first 6 values

    # Apply to the dataset
    static_token_embeddings = convert_tokens_to_embeddings(tokens, glove_embeddings, EMBEDDING_DIM)
    print(f"\nStatic Embeddings for Dataset (First 5 Tokens):")
    for token, emb in zip(tokens[:5], static_token_embeddings[:5]):
        print(f"Token: {token}, Static Embedding: {emb[:6]} ...")

    # Step 5: Generate positional embeddings
    print("\n=== Step 5: Generating Positional Embeddings ===")
    # First, demonstrate with the sample sentence
    position_embeddings_sample = positional_encoding(len(sample_tokens), EMBEDDING_DIM)
    print("Positional Embeddings for Sample Sentence:")
    for pos, emb in enumerate(position_embeddings_sample):
        print(f"Position {pos}: {emb[:5]} ...")

    # Apply to the dataset (positional embeddings for the first few tokens)
    position_embeddings_dataset = positional_encoding(len(tokens), EMBEDDING_DIM)
    print(f"\nPositional Embeddings for Dataset (First 5 Positions):")
    for pos, emb in enumerate(position_embeddings_dataset[:5]):
        print(f"Position {pos}: {emb[:5]} ...")

    # Step 6: Add positional embeddings to static embeddings to create combined embeddings
    print("\n=== Step 6: Adding Positional Embeddings to Create Combined Embeddings ===")
    # First, demonstrate with the sample sentence
    combined_sample_embeddings = add_positional_embeddings(static_sample_embeddings, EMBEDDING_DIM)
    print("\nCombined Embeddings (Static + Positional) for Sample Sentence:")
    for token, emb in zip(sample_tokens, combined_sample_embeddings):
        print(f"Token: {token}, Combined Embedding: {emb[:5]} ...")

    # Apply to the dataset
    combined_embeddings = add_positional_embeddings(static_token_embeddings, EMBEDDING_DIM)
    print(f"\nCombined Embeddings for Dataset (First 5 Tokens):")
    for token, emb in zip(tokens[:5], combined_embeddings[:5]):
        print(f"Token: {token}, Combined Embedding: {emb[:5]} ...")
