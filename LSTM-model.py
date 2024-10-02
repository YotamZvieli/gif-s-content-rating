import tensorflow as tf
import glob
import os
import datetime as dt
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision
import frames_extractor


if __name__ == '__main__':
    #process the data
    train_path = "gifs/DL project - gifs/train/"
    test_path = "gifs/DL project - gifs/test/"
    classes = ["g", "pg", "pg-13", "r"]
    label_to_num = {"g":0, "pg":1, "pg-13":2, "r":3}
    num_of_frames = 20
    height = 128
    width = 128
    mixed_precision.set_global_policy('mixed_float16')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    train_labels = []
    train_features = []

    for c in classes:
        for gif in glob.glob(os.path.join(train_path + c, "*.gif")):
            frames = frames_extractor.extract_frames(gif, width, height, num_of_frames)
            train_features.append(frames)
            train_labels.append(label_to_num[c])

    train_features = np.array(train_features).reshape(-1, num_of_frames, height, width, 3)
    train_labels = np.array(train_labels)

    encoded_labels = to_categorical(train_labels, num_classes=len(classes))
    features_train, features_test, labels_train, labels_test = train_test_split(train_features, encoded_labels, test_size=0.25, shuffle=True)

    #create the model
    model = Sequential()
    model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape=(num_of_frames, height, width, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh', data_format="channels_last", recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(ConvLSTM2D(filters=14, kernel_size=(3, 3), activation='tanh', data_format="channels_last", recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', data_format="channels_last", recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(len(classes), activation="softmax"))
    model.build()
    model.summary()

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

    # Compile the model and specify loss function, optimizer and metrics values to the model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

    # Start training the model.
    convlstm_model_training_history = model.fit(x=features_train, y=labels_train, epochs=50, batch_size=1,
                                                shuffle=True, validation_split=0.2, callbacks=[early_stopping_callback])

    model_evaluation_history = model.evaluate(features_test, labels_test)

    print(model_evaluation_history)

    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    model_file_name = f'convlstm_model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'
    model.save(model_file_name)













