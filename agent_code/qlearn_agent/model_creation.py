import keras
import numpy as np


def get_model():
    # Create a simple model.
    inputs = keras.layers.Input(shape=(32,))
    outputs = keras.layers.Dense(6, activation="sigmoid")(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

if __name__ == '__main__':
    model = get_model()

    test_input = np.random.random((128, 32))
    test_target = np.random.random((128, 1))
    model.fit(test_input, test_target)

    model.save("my_model")
    