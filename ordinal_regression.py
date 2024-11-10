import tensorflow as tf
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence

HEIGHT = 224
WIDTH = 224
SEED = 42
BATCH_SIZE = 128
EPOCHS = 50
RATING_MAP = {'g': 0, 'pg': 1, 'pg-13': 2, 'r': 3}
LEARNING_RATE = 0.001
np.random.seed(SEED)
tf.random.set_seed(SEED)

class DataGenerator(Sequence):
    def __init__(self, files_paths, labels, batch_size, max_frames, **kwargs):
        self.files_paths = files_paths
        self.labels = labels
        self.batch_size = batch_size
        self.max_frames = max_frames
        super().__init__(**kwargs)

    def __len__(self):
        return int(np.ceil(len(self.files_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_files_paths = self.files_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_gifs = []
        for gif_path in batch_files_paths:
            num_frames_to_extract = frames_for_gif_to_extract(gif_path)
            frames = self.extract_frames(gif_path, num_frames_to_extract)
            padded_frames = np.pad(frames, ((0, self.max_frames - len(frames)), (0, 0), (0, 0), (0, 0)), mode='constant')
            batch_gifs.append(padded_frames)

        return np.array(batch_gifs).astype(np.float32) / 255.0, batch_labels
    
    def extract_frames(self, gif_path, num_frames_to_extract):
        frames = []
        with Image.open(gif_path) as img:
            skip_frames_window = max(int(np.ceil(img.n_frames / num_frames_to_extract)), 1)
            for i in range(0, img.n_frames, skip_frames_window):
                img.seek(i)
                frame = img.convert('RGB')
                frame = frame.resize((WIDTH, HEIGHT))
                frame_array = img_to_array(frame).astype(np.uint8)
                frames.append(frame_array)

        return np.array(frames, dtype=np.uint8)

def frames_for_gif_to_extract(gif_path):
    gif_duration = 0
    with Image.open(gif_path) as img:
      for i in range(img.n_frames):
        img.seek(i)
        gif_duration += img.info.get('duration', 50)

    return max(gif_duration // 500, 1)

def get_max_frames_from_gifs(file_paths):
    if not 'max_num_frames' in os.listdir():
        max_num_frames = 0
        for file_path in file_paths:
            if file_path.endswith(".gif"):
                num_frames_to_extract = frames_for_gif_to_extract(file_path)
                if num_frames_to_extract > max_num_frames:
                    max_num_frames = num_frames_to_extract
        with open("max_num_frames", 'w') as f:
            f.write(str(max_num_frames))
    else:
        with open("max_num_frames", 'r') as f:
            max_num_frames = int(f.read())
    
    return max_num_frames

def load_dataset(folder_path):
    gif_paths = []
    ratings = []
    for rating_folder_name in os.listdir(folder_path):
        if rating_folder_name in RATING_MAP:
            rating_folder_path = os.path.join(folder_path, rating_folder_name)
            for filename in os.listdir(rating_folder_path):
                if filename.endswith(".gif"):
                    gif_paths.append(os.path.join(rating_folder_path, filename))
                    ratings.append(RATING_MAP[rating_folder_name])

    return np.array(gif_paths), np.array(ratings)

def build_model(max_frames, img_height, img_width, num_classes):
    model = Sequential()

    model.add(Input(shape=(max_frames, img_height, img_width, 3)))

    # Convolutional layers
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    # Flatten the convolutional features
    model.add(TimeDistributed(Flatten()))

    # Masking
    model.add(Masking(mask_value=0.0))

    # LSTM layers
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(50))

    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    # Output layers
    model.add(Dense(num_classes - 1, activation='sigmoid'))

    model.summary()

    return model

def corn_loss(y_true, y_pred):
    # Convert ordinal labels to binary tasks
    y_true_binary = tf.cast(tf.greater_equal(tf.cast(tf.expand_dims(y_true, -1), tf.float32), 
                                             tf.cast(tf.range(y_pred.shape[-1]), 
                                                     tf.float32)), tf.float32)
    
    # Calculate binary cross-entropy for each task
    bce = tf.keras.losses.binary_crossentropy(y_true_binary, y_pred)
    
    # Sum the losses across tasks
    return tf.reduce_sum(bce, axis=-1)

def train_model(model, train_generator, test_generator, epochs=50, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=corn_loss, metrics=['mae'])
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    return history

def predict_ratings(model, test_generator):
    predictions = model.predict(test_generator)
    predicted_ratings = np.sum(predictions >= 0.5, axis=1)
    return predicted_ratings

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    folder_path = r"gifs/train"

    X, y = load_dataset(folder_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=SEED, shuffle=True, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=SEED, shuffle=True, stratify=y_test)

    max_frames = get_max_frames_from_gifs(X)

    # Create data generators
    train_generator = DataGenerator(X_train, y_train, batch_size=BATCH_SIZE, max_frames=max_frames)
    test_generator = DataGenerator(X_test, y_test, batch_size=BATCH_SIZE, max_frames=max_frames)
    val_generator = DataGenerator(X_val, y_val, batch_size=BATCH_SIZE, max_frames=max_frames)

    model = build_model(max_frames, HEIGHT, WIDTH, len(RATING_MAP))

    history = train_model(model, train_generator, val_generator, EPOCHS, LEARNING_RATE)

    plot_training_history(history)

    test_loss, test_mae = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # Make predictions
    predicted_ratings = predict_ratings(model, test_generator)
    true_ratings = np.concatenate([y for x, y in test_generator])
    
    # Print classification report
    print(classification_report(true_ratings, predicted_ratings, target_names=['G', 'PG', 'PG-13', 'R']))
    
    # Plot confusion matrix
    cm = confusion_matrix(true_ratings, predicted_ratings)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(RATING_MAP))
    plt.xticks(tick_marks, ['G', 'PG', 'PG-13', 'R'])
    plt.yticks(tick_marks, ['G', 'PG', 'PG-13', 'R'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    # Save model
    model.save('gif_rating_model.h5')
    print("Model saved as 'gif_rating_model.h5'")
