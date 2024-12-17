import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize inputs to [0, 1]

# Define the model
model = Sequential([
    Input(shape=(28, 28)),  # Define the input shape explicitly using Input layer
    Flatten(),  # Flatten 28x28 images into a vector
    Dense(128, activation='relu'),  # First hidden layer
    Dense(128, activation='relu'),  # Second hidden layer
    Dense(64, activation='relu'),   # Third hidden layer
    Dense(64, activation='relu'),   # Fourth hidden layer
    Dense(10, activation='softmax') # Output layer with 10 classes
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
print("\n Training complete! \n")
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
