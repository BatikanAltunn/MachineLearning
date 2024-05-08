import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# Load MNIST dataset
##Loads the MNIST dataset and splits it into training and testing sets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()


fig, ax_arr = plt.subplots(5, 5, figsize=(5, 5))
fig.subplots_adjust(wspace=.025, hspace=.55)
ax_arr = ax_arr.ravel()
for i, ax in enumerate(ax_arr):
    r = np.random.randint(len(x_train))

    ax.imshow(x_train[r,:,:], cmap=plt.get_cmap('gray'))
    ax.axis("off")
    ax.title.set_text(str(y_train[r]))
plt.show()



# Preprocess the data
# Normalize pixel values to the range [0, 1] by dividing by 255.
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten the images
# Reshape the input images from 28x28 to a 1D array of size 784.
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# One-hot encode the labels
# Convert integer labels to one-hot encoded vectors.
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the input images
    Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    Dense(10, activation='softmax')  # Output layer with 10 neurons (for 10 classes)
])

# Compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']) # metrics=['accuracy']: Metric to monitor during training.

# Train the model
history = model.fit(x_train, y_train_encoded, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded)
print("Test Accuracy:", test_accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()   
