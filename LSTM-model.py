import tensorflow as tf
import glob
import os
import frames_extractor



if __name__ == '__main__':
    train_path = "gifs/DL project - gifs/train/"
    test_path = "gifs/DL project - gifs/test/"
    classes = ["g", "pg", "pg-13", "r"]

    for c in classes:
        for gif in glob.glob(os.path.join(test_path + c, "*.gif")):
            print(c," " ,gif)