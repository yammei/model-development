import numpy as np
import heapq
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from keras.optimizers import RMSprop

# Current dataset name extensions available: template, calling
dataset_name_ext = 'calling'

def data_preprocessing():
    print(f"Preprocessing data.")
    path = f'./datasets/preprocessed/preprocessed_data_{dataset_name_ext}.txt'
    text = open(path).read().lower()
    print('Corpus length: :', len(text))

    tokenizer = RegexpTokenizer(r"[\w']+")
    words = tokenizer.tokenize(text)

    print("Total words:", len(words))
    print("Unique words:", len(set(words)))

    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(words)
    unique_word_index = tokenizer.word_index
    indices_char = {index: word for word, index in unique_word_index.items()}

    print("Total unique words in tokenizer:", len(unique_word_index))

    SEQUENCE_LENGTH = 5
    prev_words = []
    next_words = []
    for i in range(len(words) - SEQUENCE_LENGTH):
        prev_words.append(words[i:i + SEQUENCE_LENGTH])
        next_words.append(words[i + SEQUENCE_LENGTH])

    X = np.zeros((len(prev_words), SEQUENCE_LENGTH, len(unique_word_index) + 1), dtype=bool)
    Y = np.zeros((len(next_words), len(unique_word_index) + 1), dtype=bool)
    for i, each_words in enumerate(prev_words):
        for j, each_word in enumerate(each_words):
            X[i, j, unique_word_index.get(each_word, 0)] = 1
        Y[i, unique_word_index.get(next_words[i], 0)] = 1

    return unique_word_index, SEQUENCE_LENGTH, X, Y, indices_char

def model_training(unique_word_index, SEQUENCE_LENGTH, X, Y):
    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(unique_word_index) + 1)))
    model.add(Dense(len(unique_word_index) + 1))
    model.add(Activation('softmax'))
    model.summary()

    optimizer = RMSprop(learning_rate=0.01)  # Corrected argument name
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

    model.save(f'./models/trained/word_prediction_model_{dataset_name_ext}.h5')
    pickle.dump(history, open(f"./models/metrics/metrics_{dataset_name_ext}.p", "wb"))
    model = load_model(f'./models/trained/word_prediction_model_{dataset_name_ext}.h5')
    history = pickle.load(open(f"./models/metrics/metrics_{dataset_name_ext}.p", "rb"))

    return model, history

def model_metrics(history):
    print(f"Metric  data.")
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

def word_prediction(model, unique_word_index, SEQUENCE_LENGTH, indices_char):
    def prepare_input(text):
        x = np.zeros((1, SEQUENCE_LENGTH, len(unique_word_index) + 1))
        padded_text = text[:SEQUENCE_LENGTH]  # Truncate or pad the input text
        for t, word in enumerate(padded_text):
            if word.strip():
                index = unique_word_index.get(word, 0)
                if index == 0:
                    print(f"Word '{word}' not found in the unique word index.")
                x[0, t, index] = 1
        return x

    def sample(preds, top_n=3):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return heapq.nlargest(top_n, range(len(preds)), preds.take)

    def predict_completions(text, n=3):
        if not text:
            return []

        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_indices = sample(preds, n)

        return [indices_char[idx] for idx in next_indices]

    quotes = [
        "hello",
        "we need to",
        "are we going to the",
        "lets not do that",
        "you are the"
    ]

    for q in quotes:
        seq = q[:40].lower()
        print(seq)
        print(predict_completions(seq, 5))
        print()

def store_data(uw_idx, seq_len, idx_char, file_path):
    data = {
        'uw_idx': uw_idx,
        'seq_len': seq_len,
        'idx_char': idx_char
    }
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['uw_idx'], data['seq_len'], data['idx_char']

def main():
    print(f"Running app.py script.")

    # Preprocessing
    uw_idx, seq_len, X, Y, idx_char = data_preprocessing()
    store_data(uw_idx, seq_len, idx_char, f'./models/misc/data_{dataset_name_ext}.pkl')

    # Model training
    # model_training(uw_idx, seq_len, X, Y)

    # Uncomment to skip model training
    uw_idx, seq_len, idx_char = load_data(f'./models/misc/data_{dataset_name_ext}.pkl')
    model = load_model(f'./models/trained/word_prediction_model_{dataset_name_ext}.h5')
    history = pickle.load(open(f"./models/metrics/history_{dataset_name_ext}.p", "rb"))

    # Model testing
    model_metrics(history)
    word_prediction(model, uw_idx, seq_len, idx_char)


# Entry point of the script
if __name__ == "__main__":
    print(f"\n\n\n\n\n\n")
    main()
