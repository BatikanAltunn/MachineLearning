import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# Simulated dataset
np.random.seed(42)
num_samples = 10000
roulette_outcomes = np.random.randint(0, 37, num_samples)  # Assuming European roulette
colors = ['Red' if num in [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36] else 'Black' for num in roulette_outcomes]

# Create DataFrame
data = pd.DataFrame({'Outcome': roulette_outcomes, 'Color': colors})

# Feature Engineering
data['Previous_Outcome'] = data['Outcome'].shift(1)
data['Previous_Color'] = data['Color'].shift(1)

# Drop NaNs
data.dropna(inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
data['Color'] = label_encoder.fit_transform(data['Color'])
data['Previous_Color'] = label_encoder.transform(data['Previous_Color'])

# Split data into features and target
X = data.drop('Outcome', axis=1)
y = to_categorical(data['Outcome'])  # One-hot encode target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(37, activation='softmax')  # Output layer with 37 neurons for each possible outcome
])

# Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.5)

# Model evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)


# Predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Combine predicted classes with actual outcomes
results = pd.DataFrame({'Actual_Outcome': np.argmax(y_test, axis=1), 'Predicted_Outcome': predicted_classes})

# Print all predictions
print("All Predictions:")
print(results)