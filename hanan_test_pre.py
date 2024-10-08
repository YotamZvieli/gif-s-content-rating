import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

HEIGHT = 64
WIDTH = 64
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class DataGenerator(Sequence):
    def __init__(self, gifs_frames, labels, batch_size, max_frames, **kwargs):
        self.gifs_frames = gifs_frames
        self.labels = labels
        self.batch_size = batch_size
        self.max_frames = max_frames
        super().__init__(**kwargs)

    def __len__(self):
        return int(np.ceil(len(self.gifs_frames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_gifs_frames = self.gifs_frames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_gifs = []
        for frames in batch_gifs_frames:
            padded_frames = np.pad(frames, ((0, self.max_frames - len(frames)), (0, 0), (0, 0), (0, 0)), mode='constant')
            new_frames = []
            for frame in padded_frames:
                new_frames.append(preprocess_input(frame))
            batch_gifs.append(np.array(padded_frames))

        return np.array(batch_gifs), batch_labels # .astype(np.float32) / 255.0, batch_labels

def extract_frames(gif_path, max_frames):
    frames = []
    with Image.open(gif_path) as img:
        skip_frames_window = max(img.n_frames // max_frames, 1)
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

def load_dataset(folder_path):
    gif_sequences = []
    ratings = []
    for rating_folder_name in os.listdir(folder_path):
        rating_folder_path = os.path.join(folder_path, rating_folder_name)
        for filename in os.listdir(rating_folder_path):
            if filename.endswith(".gif"):
                file_path = os.path.join(rating_folder_path, filename)
                num_frames_to_extract = frames_for_gif_to_extract(file_path)
                frames = extract_frames(file_path, num_frames_to_extract)
                gif_sequences.append(frames)
                ratings.append(rating_folder_name)

    return np.array(gif_sequences, dtype='object'), np.array(ratings)

def create_model(input_shape, num_classes, base_model_trainable):

    # Load pre-trained ResNet50 without top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape[1:])
    
    # Freeze the base model layers
    base_model.trainable = base_model_trainable
    
    # Create your model
    inputs = tf.keras.Input(shape=input_shape)
    x = TimeDistributed(base_model)(inputs)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(128)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)

    model.summary()
    
    return model

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    '''
    This function will plot the metrics passed to it in a graph.
    Args:
        model_training_history: A history object containing a record of training and validation
                                loss values and metrics values at successive epochs
        metric_name_1:          The name of the first metric that needs to be plotted in the graph.
        metric_name_2:          The name of the second metric that needs to be plotted in the graph.
        plot_name:              The title of the graph.
    '''

    # Get metric values using metric names as identifiers.
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_value_1))

    # Plot the Graph.
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)

    # Add title to the plot.
    plt.title(str(plot_name))

    # Add legend to the plot.
    plt.legend()

if __name__ == '__main__':
    folder_path = r"C:\gif-s-content-rating\gifs\train"

    if not all(cache_name in os.listdir() for cache_name in ['features_train_pre.npy', 
                                                             'features_test_pre.npy', 
                                                             'features_val_pre.npy', 
                                                             'labels_train_pre.npy', 
                                                             'labels_test_pre.npy', 
                                                             'labels_val_pre.npy']):
        print("started loading dataset...")
        X, y = load_dataset(folder_path)
        print("finished loading dataset...")
        print(f'Number of gifs: {len(X)}')

        print('started encodeing labels...')
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Convert labels into one-hot-encoded vectors
        one_hot_encoded_labels = to_categorical(y_encoded)
        print('finished encodeing labels...')

        X_train, X_test, y_train, y_test = train_test_split(X, one_hot_encoded_labels,
                                                            test_size=0.2, shuffle = True,
                                                            random_state=SEED)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle = True, random_state=SEED)

        np.save("features_train_pre", X_train)
        np.save("features_test_pre", X_test)
        np.save("features_val_pre", X_val)
        np.save("labels_train_pre", y_train)
        np.save("labels_test_pre", y_test)
        np.save("labels_val_pre", y_val)
    else:
        X_train = np.load("features_train_pre.npy", allow_pickle=True)
        X_test = np.load("features_test_pre.npy", allow_pickle=True)
        X_val = np.load("features_val_pre.npy", allow_pickle=True)
        y_train = np.load("labels_train_pre.npy")
        y_test = np.load("labels_test_pre.npy")
        y_val = np.load("labels_val_pre.npy")
        print("Loaded cache")

    max_frames = max(len(seq) for seq in np.concatenate((X_train, X_test, X_val), axis=0))
    num_classes = len(np.unique(np.concatenate((y_train, y_test, y_val), axis=0), axis=0))

    # Create data generators
    train_generator = DataGenerator(X_train, y_train, batch_size=8, max_frames=max_frames)
    test_generator = DataGenerator(X_test, y_test, batch_size=8, max_frames=max_frames)
    val_generator = DataGenerator(X_val, y_val, batch_size=8, max_frames=max_frames)

    input_shape = (max_frames, WIDTH, HEIGHT, 3)

    # ----------------First model----------------------
    model = create_model(input_shape, num_classes, False)
    print('created first model')

    # Create an Instance of Early Stopping Callback.
    early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)

    # Compile the model and specify loss function, optimizer and metrics to the model.
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

    print('started training first model...')
    # Start training the model.
    training_history = model.fit(train_generator, epochs = 70, shuffle=True, validation_data=val_generator, callbacks = [early_stopping_callback])
    print('finished training first model...')

    evaluation_history = model.evaluate(test_generator, verbose=2)
    test_loss, test_acc = evaluation_history
    print(f"\nFirst model test loss: {test_loss}")
    print(f"\nFirst model test accuracy: {test_acc}")

    # Visualize the training and validation loss metrices.
    plot_metric(training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')

    # Visualize the training and validation accuracy metrices.
    plot_metric(training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

    model_file_name = f'LRCN_model_64_ResNet50_Freeze__Loss_{test_loss}__Accuracy_{test_acc}.h5'

    # Save the Model.
    model.save(model_file_name)


    # ------------Second model---------------
    model = create_model(input_shape, num_classes, True)
    print('created second model')

    # Create an Instance of Early Stopping Callback.
    early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)

    # Compile the model and specify loss function, optimizer and metrics to the model.
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

    print('started training second model...')
    # Start training the model.
    training_history = model.fit(train_generator, epochs = 70, shuffle=True, validation_data=val_generator, callbacks = [early_stopping_callback])
    print('finished training second model...')

    evaluation_history = model.evaluate(test_generator, verbose=2)
    test_loss, test_acc = evaluation_history
    print(f"\nSecond model test loss: {test_loss}")
    print(f"\nSecond model test accuracy: {test_acc}")

    # Visualize the training and validation loss metrices.
    plot_metric(training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')

    # Visualize the training and validation accuracy metrices.
    plot_metric(training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

    model_file_name = f'LRCN_model_64_ResNet50_Trainable__Loss_{test_loss}__Accuracy_{test_acc}.h5'

    # Save the Model.
    model.save(model_file_name)