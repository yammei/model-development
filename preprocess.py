import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# Read the text file
with open('./datasets/calling/calling-training-dataset.txt', 'r') as file:
    data = file.readlines()

# Convert text to lowercase and tokenize
tokenizer = Tokenizer()
texts = []
for line in data:
    # Check if the line contains a colon
    if ":" in line:
        # Split the line at the first colon
        parts = line.split(":", 1)
        # Check if the split resulted in two parts
        if len(parts) == 2:
            # Extract text after the colon
            text = parts[1].strip()
            # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)
            # Convert to lowercase
            text = text.lower()
            texts.append(text)

# Fit tokenizer on texts
tokenizer.fit_on_texts(texts)

# Remove stopwords and infrequent/rare words
stop_words = set(stopwords.words('english'))
word_counts = tokenizer.word_counts
for word in list(word_counts):
    if word in stop_words or word_counts[word] < 2:
        del word_counts[word]
tokenizer.word_index = {word: index for index, word in enumerate(word_counts, 1)}

# Encode text
encoded_texts = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_sequence_length = max(len(seq) for seq in encoded_texts)
padded_sequences = pad_sequences(encoded_texts, maxlen=max_sequence_length, padding='post')

# Convert to numpy array
padded_sequences = np.array(padded_sequences)

# Save preprocessed data
np.save('preprocessed_data.npy', padded_sequences)
