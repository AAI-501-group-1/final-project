import pandas as pd
import numpy as np

def load_wine_data_from_local():
    # Loading the datasets with the correct delimiter ( the delimiter is set to ; , which matches the dataset format.
    x_train = pd.read_csv("data/UCI_HAR_Dataset/train/X_train.txt", sep=r'\s+', header=None)
    y_train = pd.read_csv("data/UCI_HAR_Dataset/train/y_train.txt", sep=r'\s+', header=None)

    x_test = pd.read_csv("data/UCI_HAR_Dataset/test/X_test.txt", sep=r'\s+', header=None)
    y_test = pd.read_csv("data/UCI_HAR_Dataset/test/y_test.txt", sep=r'\s+', header=None)

    features = pd.read_csv("data/UCI_HAR_Dataset/features.txt", sep=r'\s+', header=None, names=["index", "feature_name"])
    activity_labels = pd.read_csv("data/UCI_HAR_Dataset/activity_labels.txt",sep=r'\s+',header=None,names=["id", "label"])

    return x_train, y_train, x_test, y_test, features, activity_labels

def load_raw_signals_data():
    # Read raw signals grouped into 2.56s windows
    acc_x = np.loadtxt('data/UCI_HAR_Dataset/train/Inertial_Signals/total_acc_x_train.txt')
    acc_y = np.loadtxt('data/UCI_HAR_Dataset/train/Inertial_Signals/total_acc_y_train.txt')
    acc_z = np.loadtxt('data/UCI_HAR_Dataset/train/Inertial_Signals/total_acc_z_train.txt')
    labels = np.loadtxt("data/UCI_HAR_Dataset/train/y_train.txt", dtype=int)

    return acc_x, acc_y, acc_z, labels