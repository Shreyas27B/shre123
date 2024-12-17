import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Parameters
max_features = 10000  # Top 10000 most common words
max_len = 200  # Max length of each review

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train, x_test = map(lambda x: pad_sequences(x, maxlen=max_len), (x_train, x_test))

# Build, compile, and train the model
model = Sequential([
    Embedding(input_dim=max_features, output_dim=64, input_length=max_len),
    Bidirectional(LSTM(32)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
print(f"Test Accuracy: {model.evaluate(x_test, y_test)[1]:.2f}")

# Test on a custom review
example_review = "The movie was absolutely amazing, I loved it!"
encoded_review = [imdb.get_word_index().get(word, 2) for word in example_review.lower().split()]
padded_review = pad_sequences([encoded_review], maxlen=max_len)
prediction = model.predict(padded_review)[0][0]
print(f"{'Positive' if prediction < 0.5 else 'Negative'} sentiment with confidence {1 - prediction if prediction < 0.5 else prediction:.2f}")
