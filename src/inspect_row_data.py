import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from  load_data import load_raw_signals_data

def inspect_row_data(activity_labels):
    # Function to draw Accelerometer signals
    def draw_accelerometer_signals(axes, x_axis, y_axis, z_axis, label):
        time = np.linspace(0, len(x_axis) / 50, len(x_axis))
        sns.set_style('whitegrid')
        # plt.figure(figsize=(8,4))
        axes.plot(time, x_axis, color='r', label='X-axis')
        axes.plot(time, y_axis, color='g', label='Y-axis')
        axes.plot(time, z_axis, color='b', label='Z-axis')
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Acceleration (g)")
        axes.set_title(f"Accelerometer Signals for Sample {label}")
        axes.legend()

    # Read raw signals grouped into 2.56s windows
    acc_x, acc_y, acc_z, labels = load_raw_signals_data()

    # Find the first row of each activity, so we can draw plots of the signals of each activity
    first_indices = {}
    indices = []
    for i, label in enumerate(labels):
        if label not in first_indices:
            indices.append(i)
            first_indices[label] = i

    print(f'The shape of raw total_acc_x_train: {acc_x.shape}')
    print(f'The shape of lables: {labels.shape}')
    print('\n********************* Signals examples for each activity *********************\n')

    fig, axes = plt.subplots(2, 3, figsize=(15, 6))  # 2 rows, 3 columns
    axes = axes.flatten()

    for i in range(6):
        draw_accelerometer_signals(axes[i], acc_x[indices[i]], acc_y[indices[i]], acc_z[indices[i]],
                                   activity_labels.loc[labels[indices[i]] - 1]['label'])

    plt.tight_layout()
    plt.show()