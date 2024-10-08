import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping

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
            batch_gifs.append(padded_frames)

        return np.array(batch_gifs).astype(np.float32) / 255.0, batch_labels

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

# def pad_sequences_3d(sequences, max_length, padding_value=0):
#     # Get the shape of the frames
#     frame_shape = sequences[0].shape[1:]

#     padded_sequences = np.full((len(sequences), max_length) + frame_shape, padding_value, dtype=np.uint8)

#     for i, seq in enumerate(sequences):
#         padded_sequences[i, :len(seq)] = seq

#     return padded_sequences

#     # # Create a list to store padded sequences
#     # padded_sequences = []

#     # for seq in sequences:
#     #     padding = np.full((max_length - len(seq),) + frame_shape, padding_value)
#     #     padded_seq = np.vstack((seq, padding))

#     #     padded_sequences.append(padded_seq)

#     # return np.array(padded_sequences)

def create_model(input_shape, num_classes, masking):

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu')))

    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    #model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))

    if masking:
        model.add(Masking(mask_value=0.0))

    model.add(LSTM(32))

    model.add(Dense(num_classes, activation = 'softmax'))

    ########################################################################################################################

    # Display the models summary.
    model.summary()

    # Return the constructed LRCN model.
    return model

def create_model_2(input_shape, num_classes):

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(TimeDistributed(Conv2D(24, (11, 11), strides=4, padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(64, (5, 5), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(96, (3, 3), padding='same',activation = 'relu')))

    model.add(TimeDistributed(Conv2D(96, (3, 3), padding='same',activation = 'relu')))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Dense(144, activation = 'relu')))
    model.add(TimeDistributed(Dense(144, activation = 'relu')))
    model.add(TimeDistributed(Dense(72, activation = 'relu')))

    model.add(TimeDistributed(Flatten()))

    model.add(Masking(mask_value=0.0))

    model.add(LSTM(32))

    model.add(Dense(num_classes, activation = 'softmax'))

    # Display the models summary.
    model.summary()

    # Return the constructed LRCN model.
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
    
    if not all(cache_name in os.listdir() for cache_name in ['features_train_64.npy', 
                                                             'features_test_64.npy', 
                                                             'features_val_64.npy', 
                                                             'labels_train_64.npy', 
                                                             'labels_test_64.npy', 
                                                             'labels_val_64.npy']):
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

        np.save("features_train_64", X_train)
        np.save("features_test_64", X_test)
        np.save("features_val_64", X_val)
        np.save("labels_train_64", y_train)
        np.save("labels_test_64", y_test)
        np.save("labels_val_64", y_val)
    else:
        X_train = np.load("features_train_64.npy", allow_pickle=True)
        X_test = np.load("features_test_64.npy", allow_pickle=True)
        X_val = np.load("features_val_64.npy", allow_pickle=True)
        y_train = np.load("labels_train_64.npy")
        y_test = np.load("labels_test_64.npy")
        y_val = np.load("labels_val_64.npy")
        print("Loaded cache")


    # print("started loading dataset...")
    # X, y = load_dataset(folder_path)
    # print("finished loading dataset...")
    # print(f'Number of gifs: {len(X)}')

    # max_frames = max(len(seq) for seq in X)

    # # print("started padding...")
    # # X_padded = pad_sequences_3d(X, max_frames)
    # # print("finished padding...")

    # #X_padded = X_padded.astype(np.float32) / 255.0

    # print('started encodeing labels...')
    # le = LabelEncoder()
    # y_encoded = le.fit_transform(y)
    # num_classes = len(le.classes_)
    # # Convert labels into one-hot-encoded vectors
    # one_hot_encoded_labels = to_categorical(y_encoded)
    # print('finished encodeing labels...')

    # X_train, X_test, y_train, y_test = train_test_split(X, one_hot_encoded_labels,
    #                                                     test_size=0.2, shuffle = True,
    #                                                     random_state=SEED)
    
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle = True, random_state=SEED)

    max_frames = max(len(seq) for seq in np.concatenate((X_train, X_test, X_val), axis=0))
    num_classes = len(np.unique(np.concatenate((y_train, y_test, y_val), axis=0), axis=0))

    # Create data generators
    train_generator = DataGenerator(X_train, y_train, batch_size=8, max_frames=max_frames)
    test_generator = DataGenerator(X_test, y_test, batch_size=8, max_frames=max_frames)
    val_generator = DataGenerator(X_val, y_val, batch_size=8, max_frames=max_frames)

    input_shape = (max_frames, WIDTH, HEIGHT, 3)

    # ----------------First model----------------------
    model = create_model(input_shape, num_classes, True)
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

    model_file_name = f'First_LRCN_model_64__Loss_{test_loss}__Accuracy_{test_acc}.h5'

    # Save the Model.
    model.save(model_file_name)


    # -------------Second model--------------
    model = create_model(input_shape, num_classes, False)
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

    model_file_name = f'Second_LRCN_model_64__Loss_{test_loss}__Accuracy_{test_acc}.h5'

    # Save the Model.
    model.save(model_file_name)


    #----------Third model-----------
    model = create_model_2(input_shape, num_classes)
    print('created third model')

    # Create an Instance of Early Stopping Callback.
    early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)

    # Compile the model and specify loss function, optimizer and metrics to the model.
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

    print('started training third model...')
    # Start training the model.
    training_history = model.fit(train_generator, epochs = 70, shuffle=True, validation_data=val_generator, callbacks = [early_stopping_callback])
    print('finished training third model...')

    evaluation_history = model.evaluate(test_generator, verbose=2)
    test_loss, test_acc = evaluation_history
    print(f"\nThird model test loss: {test_loss}")
    print(f"\nThird model test accuracy: {test_acc}")

    # Visualize the training and validation loss metrices.
    plot_metric(training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')

    # Visualize the training and validation accuracy metrices.
    plot_metric(training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

    model_file_name = f'Third_LRCN_model_64__Loss_{test_loss}__Accuracy_{test_acc}.h5'

    # Save the Model.
    model.save(model_file_name)