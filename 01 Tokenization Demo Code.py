# Sample sentence
sentence = "We Love Large Language Model (LLM)."

# ==================================================
# 1. Whitespace Tokenization
# Splits text based on whitespace (spaces, tabs, newlines)
# ==================================================
def whitespace_tokenization(text):
    return text.split()

print("Whitespace Tokenization:", whitespace_tokenization(sentence))

# ==================================================
# 2. Character Tokenization
# Splits text into individual characters
# ==================================================
def character_tokenization(text):
    return list(text)

print("Character Tokenization:", character_tokenization(sentence))

# ==================================================
# 3. Word Tokenization
# Splits text into words using a simple regex-based tokenizer
# ==================================================
import re

def word_tokenization(text):
    return re.findall(r'\b\w+\b', text)

print("Word Tokenization:", word_tokenization(sentence))

# ==================================================
# 4. Sentence Tokenization
# Splits text into sentences using NLTK
# ==================================================
from nltk.tokenize import sent_tokenize

def sentence_tokenization(text):
    return sent_tokenize(text)

print("Sentence Tokenization:", sentence_tokenization(sentence))

# ==================================================
# 5. Byte-Pair Encoding (BPE)
# Uses Hugging Face's `tokenizers` library for BPE
# ==================================================
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize and train a BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator([sentence], trainer)

def bpe_tokenization(text):
    return tokenizer.encode(text).tokens

print("BPE Tokenization:", bpe_tokenization(sentence))

# ==================================================
# 6. WordPiece Tokenization
# Uses Hugging Face's `transformers` library for WordPiece
# ==================================================
from transformers import BertTokenizer

# Load a pre-trained WordPiece tokenizer
wordpiece_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def wordpiece_tokenization(text):
    return wordpiece_tokenizer.tokenize(text)

print("WordPiece Tokenization:", wordpiece_tokenization(sentence))

# ==================================================
# 7. SentencePiece Tokenization
# Uses the `sentencepiece` library
# ==================================================
import sentencepiece as spm
import io

# Sample sentence
sentence = "We Love Large Language Model (LLM)."

# Create an in-memory file-like object with the sample sentence
text_data = io.StringIO(sentence)

# Train SentencePiece model in-memory
spm.SentencePieceTrainer.train(
    sentence_iterator=text_data,  # Use the in-memory data
    model_prefix="spm_model",     # Prefix for the model files
    vocab_size=23,                # Vocabulary size
    model_type="bpe",         # Model type (unigram or bpe)
    input_sentence_size=-1,       # Number of sentences to use for training
    pad_id=0,                     # Padding token ID
    unk_id=1,                     # Unknown token ID
    bos_id=2,                     # Beginning-of-sentence token ID
    eos_id=3                      # End-of-sentence token ID
)

# Load the trained model
sp = spm.SentencePieceProcessor()
sp.load("spm_model.model")

# Tokenize the sentence
def sentencepiece_tokenization(text):
    return sp.encode_as_pieces(text)

print("SentencePiece Tokenization:", sentencepiece_tokenization(sentence))

# ==================================================
# 8. Unigram Tokenization
# Uses Hugging Face's `tokenizers` library for Unigram
# ==================================================
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize and train a Unigram tokenizer
tokenizer = Tokenizer(Unigram())
trainer = UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator([sentence], trainer)

def unigram_tokenization(text):
    return tokenizer.encode(text).tokens

print("Unigram Tokenization:", unigram_tokenization(sentence))

# ==================================================
# 9. Byte-Level BPE (BBPE)
# Uses Hugging Face's `tokenizers` library for BBPE
# ==================================================
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# Initialize and train a Byte-Level BPE tokenizer
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.train_from_iterator([sentence], trainer)

def bbpe_tokenization(text):
    return tokenizer.encode(text).tokens

print("Byte-Level BPE Tokenization:", bbpe_tokenization(sentence))





