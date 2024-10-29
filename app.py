import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Using the same dataset and conversion method from before
df = pd.read_csv('/kaggle/input/ckdataset/ckextended.csv')
df['image'] = df['pixels'].apply(pixels_to_image)

# Flatten images for SVM input
df['flattened_image'] = df['image'].apply(lambda x: x.flatten())

# Select training data
train_data = df[df['Usage'] == 'Training']

# Extract features and targets
X = np.stack(train_data['flattened_image'].values)
y = train_data['emotion'].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and (if necessary) validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train SVM
svm_model = SVC(kernel='linear', C=1, probability=True)  # Use 'rbf' for radial basis function kernel if suitable
svm_model.fit(X_train, y_train)


# Evaluate the model
y_pred = svm_model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Optionally visualize confusion matrix or other metrics

# If desired, save the SVM model using joblib
# from joblib import dump
# dump(svm_model, 'svm_emotion_model.joblib')

import matplotlib.pyplot as plt

# Define a mapping from numeric labels to emotion names
emotion_mapping = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral',
    7: 'Contempt'
}

# Select 5 random images from the validation set
random_indices = np.random.choice(len(X_val), size=5, replace=False)
random_images = X_val[random_indices]
random_labels = y_val[random_indices]

# Predict emotions for the selected images
predictions = svm_model.predict(random_images)

# Plot the images and their predicted values
plt.figure(figsize=(15, 6))

for i, index in enumerate(random_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(random_images[i].reshape(48, 48), cmap='gray')  # Reshape back to original image size
    # Map the numeric predictions and labels to their corresponding emotion names
    predicted_emotion = emotion_mapping[predictions[i]]
    true_emotion = emotion_mapping[random_labels[i]]
    plt.title(f"Predicted: {predicted_emotion}\nTrue: {true_emotion}")
    plt.axis('off')

plt.tight_layout()
plt.show()
