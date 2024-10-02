import tensorflow as tf
import glob
import os
import numpy as np
from tensorflow.keras.layers import *
import frames_extractor


if __name__ == '__main__':
    test_path = "gifs/DL project - gifs/test/"
    classes = ["g", "pg", "pg-13", "r"]
    label_to_num = {"g":0, "pg":1, "pg-13":2, "r":3}
    num_of_frames = 5
    height = 128
    width = 128

    train_labels = []
    train_features = []
    for c in classes:
        for gif in glob.glob(os.path.join(test_path + c, "*.gif")):
            frames = frames_extractor.extract_frames(gif, width, height, num_of_frames)
            train_features.append(frames)
            train_labels.append(label_to_num[c])

    train_features = np.array(train_features).reshape(-1, num_of_frames, height, width, 3)
    train_labels = np.array(train_labels)

    model = tf.keras.models.load_model('convlstm_model___Date_Time_2024_10_02__15_16_21___Loss_0.9561542272567749___Accuracy_0.6666666865348816.h5')

    predicted = model.predict(train_features)

    print(predicted)

    predicted = predicted.tolist()
    results = []
    for i in range(len(predicted)):
        results.append(predicted[i].index(max(predicted[i])))

    print(results)
    print(train_labels.tolist())

