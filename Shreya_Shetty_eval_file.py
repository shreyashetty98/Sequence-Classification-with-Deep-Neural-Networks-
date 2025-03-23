import numpy as np
import pandas as pd
from keras.models import load_model
import pickle

# Longitude and Latitude values are calculated in the other notebook
min_long, max_long = 100.267, 767.0096599999999
min_lat, max_lat = 16.0028, 28028.0


# Processing
def process_data(traj):
    """
    Input:
        traj: a list of list, contains one trajectory for one driver
    Output:
        data: numpy array that can be consumed by your model
    """
    traj_points = []

    for pnt in traj:
        longitude = (pnt[0] - min_long) / (max_long - min_long)
        latitude = (pnt[1] - min_lat) / (max_lat - min_lat)
        timestamp = pnt[2]
        constant = pnt[3]

        # Converting time
        time_in_seconds = float(timestamp.split()[1].split(':')[0]) * 3600 + \
                          float(timestamp.split()[1].split(':')[1]) * 60 + \
                          float(timestamp.split()[1].split(':')[2])

        # Sin and Cos transformations
        time_sin = np.sin(time_in_seconds * (2 * np.pi / 86400))
        time_cos = np.cos(time_in_seconds * (2 * np.pi / 86400))

        # Feature vector
        ft_vector = [longitude, latitude, time_sin, time_cos, constant]
        traj_points.append(ft_vector)

    # Converting to numpy array and reshaping for LSTM model
    return np.array(traj_points).reshape(1, len(traj_points), 5)


def run(data, model):
    """
    Input:
        data: the output of process_data function (numpy array).
        model: your trained model.
    Output:
        prediction: the predicted label (driver ID), an integer value.
    """
    pred = model.predict(data)

    # Returning the driver ID
    pred_label = np.argmax(pred, axis=1)[0]
    return pred_label


if __name__ == "__main__":
    # Loading the model
    model = load_model(r'C:\Users\shrey\Downloads\taxi_driver_model.h5')

    # Loading the test data file
    with open(r'C:\Users\shrey\Downloads\test.pkl', 'rb') as file:
        testing_data = pickle.load(file)

    for traj in testing_data:
        processed_data = process_data(traj)

        # Predicting the driver
        pred_driver = run(processed_data, model)

        print(f"Predicted Driver is : {pred_driver}")
