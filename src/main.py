from load_data import load_wine_data_from_local
from understanding_data import understanding_data
from normilaze_data import normilaze_data
import pandas as pd

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

x_train, y_train, x_test, y_test, features, activity_labels = load_wine_data_from_local()
understanding_data(x_train, y_train, x_test, y_test, features, activity_labels)
x_train_scaled, x_test_scaled = normilaze_data(x_train, x_test)
