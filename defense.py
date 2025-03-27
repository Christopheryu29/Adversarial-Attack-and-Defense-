import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Set TensorFlow environment variables and configurations
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings

# Ensure TensorFlow uses XLA (Accelerated Linear Algebra) optimization
tf.config.optimizer.set_jit(True)  # Enable XLA

# Function to load images and extract features
def load_images_and_extract_features(data_dir, image_size=(512, 512)):
    features = []
    labels = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".jpeg"):
                    image_path = os.path.join(folder_path, file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                    image = cv2.resize(image, image_size)
                    features.append(image)
                    labels.append(folder)

    features = np.array(features)
    labels = np.array(labels)
    return features, labels

# Load images and extract features
data_dir = "/kaggle/input/images"
X, y = load_images_and_extract_features(data_dir)

# Convert string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=35)

# Normalize pixel values and reshape to include the channel dimension for grayscale images
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Define a simpler model architecture for grayscale images
def create_simple_model():
    input_layer = layers.Input(shape=(512, 512, 1))
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    flatten = layers.Flatten()(pool2)
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dropout1 = layers.Dropout(0.5)(dense1)  # Added dropout for regularization
    output = layers.Dense(len(np.unique(y_encoded)), activation='softmax')(dropout1)
    model = models.Model(inputs=input_layer, outputs=output)
    return model

# Define the custom training loop with gradient noise addition
def train_model_with_dp(model, X_train, y_train, X_test, y_test, epochs=15, learning_rate=0.0005, noise_multiplier=0.1):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    batch_size = 32

    train_accuracy = []
    val_accuracy = []

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()

        for step in range(len(X_train) // batch_size):
            x_batch_train = X_train[step * batch_size:(step + 1) * batch_size]
            y_batch_train = y_train[step * batch_size:(step + 1) * batch_size]

            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Add noise to gradients
            noise_stddev = noise_multiplier * learning_rate
            grads = [g + tf.random.normal(g.shape, stddev=noise_stddev) for g in grads]

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric.update_state(y_batch_train, logits)

        train_acc = train_acc_metric.result()
        print(f"Training accuracy over epoch: {float(train_acc):.4f}")
        train_accuracy.append(float(train_acc))

        # Run validation at the end of each epoch
        for step in range(len(X_test) // batch_size):
            x_batch_val = X_test[step * batch_size:(step + 1) * batch_size]
            y_batch_val = y_test[step * batch_size:(step + 1) * batch_size]

            val_logits = model(x_batch_val, training=False)
            val_acc_metric.update_state(y_batch_val, val_logits)

        val_acc = val_acc_metric.result()
        val_accuracy.append(float(val_acc))
        print(f"Validation accuracy: {float(val_acc):.4f}")

    # Plotting training and validation accuracy
    plt.plot(range(1, epochs + 1), train_accuracy, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

# Create and compile the model
simple_model_dp = create_simple_model()
simple_model_dp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with differential privacy
train_model_with_dp(simple_model_dp, X_train, y_train, X_test, y_test, epochs=15)

# Evaluate the DP model
test_loss_dp, test_accuracy_dp = simple_model_dp.evaluate(X_test, y_test)
print("Test Accuracy with Differential Privacy:", test_accuracy_dp)