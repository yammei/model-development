import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model
model = tf.keras.models.load_model("./models/next_word_prediction_model.keras")

def predict_next_words(input_text, model, max_len=20, num_words=1):
    # Define a new tokenizer and fit on the input text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([input_text])
    
    # Tokenize the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])[0]
    
    # Pad the input sequence
    input_sequence = pad_sequences([input_sequence], maxlen=max_len-1, padding='pre')
    
    # Predict the next word
    predicted_word_indices = model.predict(input_sequence)
    
    # Get the indices of the top predicted words and their probabilities
    top_indices = np.argsort(predicted_word_indices[0])[-num_words:][::-1]
    top_probabilities = np.sort(predicted_word_indices[0])[-num_words:][::-1]
    
    # Convert the indices back to words
    predicted_words = []
    for index, probability in zip(top_indices, top_probabilities):
        print(tokenizer.index_word.get(index))
        word = tokenizer.index_word.get(index)
        if word is not None:
            predicted_words.append((word, probability))
    return predicted_words

print("kore2")
input_text = "hey"
predicted_words = predict_next_words(input_text, model)
for word, probability in predicted_words:
    print("Predicted word:", word, "Probability:", probability)
