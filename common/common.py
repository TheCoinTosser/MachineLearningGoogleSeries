import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def full_file_path(path, filename):
    return os.path.dirname(path) + "/" + filename


def open_file(path, filename):
    return open(full_file_path(path, filename))


def read_csv(path, filename):

    with open_file(path, filename) as f:
        csv = pd.read_csv(f)

    return csv


def encode_label(data_frame,
                 index,
                 label_encoder=LabelEncoder()):

    data_frame_column_encoded = label_encoder.fit_transform(data_frame.iloc[:, index])
    data_frame.iloc[:, index] = data_frame_column_encoded
    return label_encoder
