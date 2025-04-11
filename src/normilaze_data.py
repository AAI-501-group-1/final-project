from sklearn.preprocessing import StandardScaler


def normalize_data(x_train, x_test):
    # Normalize feature data before training
    scaler = StandardScaler()

    # Fit the scaler to the training data, then transform
    # both training and test sets
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled