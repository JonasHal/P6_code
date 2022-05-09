import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Dropout, Flatten

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences):
            break

        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

def getModelStructure(type, layers, n_steps_in, n_steps_out, n_features, n_nodes, n_nodes_cnn):
    """Creates the structure of a model with the given hyperparameters. Kept simple
    @param type: Should be a string: either "LSTM", "GRU", "CNN" or "LSTM-CNN"
    @param layers: How many layers the Neural Network consist of. Should be 1 or 2
    @param n_steps_in: Number of rows in the input
    @param n_steps_out: Number of rows in the output
    @param n_features: Number of columns in the input
    @param n_nodes: Number of nodes in the Recurrent Neural Networks
    @param n_nodes_cnn: Number of nodes in the Convolutional Neural Networks

    @return: A model structure with the given hyperparameters
    """
    if type == "LSTM" and layers == 1:
        model = Sequential([
            # LSTM(n_nodes, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)),
            LSTM(n_nodes, activation='relu', input_shape=(n_steps_in, n_features)),
            Dense(n_steps_out, activation='relu'),
        ])
    elif type == "LSTM" and layers == 2:
        model = Sequential([
            LSTM(n_nodes, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)),
            LSTM(n_nodes, activation='relu', input_shape=(n_steps_in, n_features)),
            Dense(n_steps_out, activation='relu'),
        ])


    elif type == "GRU" and layers == 1:
        model = Sequential([
        GRU(n_nodes, activation='relu', input_shape=(n_steps_in, n_features)),
        Dense(n_steps_out, activation='relu')
        ])

    elif type == "GRU" and layers == 2:
        model = Sequential([
            GRU(n_nodes, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)),
            GRU(n_nodes, activation='relu', input_shape=(n_steps_in, n_features)),
            Dense(n_steps_out, activation='relu'),
        ])
    elif type == "CNN" and layers == 1:
        model = Sequential([
            Conv1D(n_nodes_cnn, kernel_size=n_features, activation='relu', input_shape=(n_steps_in, n_features)),
            MaxPooling1D(pool_size=n_features),
            Dropout(0.2),
            Flatten(),
            Dense(n_steps_out, activation='relu')
        ])

    elif type == "CNN" and layers == 2:
        model = Sequential([
            Conv1D(n_nodes_cnn, kernel_size=1, activation='relu', input_shape=(n_steps_in, n_features)),
            MaxPooling1D(pool_size=1),
            Dropout(0.2),
            Conv1D(n_nodes_cnn, kernel_size=1, activation='relu', input_shape=(n_steps_in, n_features,)),
            MaxPooling1D(pool_size=1),
            Dropout(0.2),
            Flatten(),
            Dense(n_steps_out, activation='relu')
        ])
    elif type == "LSTM-CNN":
        model = Sequential([
            LSTM(n_nodes, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)),
            Conv1D(n_nodes_cnn, kernel_size=2, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(n_steps_out, activation='relu')
        ])
    else:
        raise Exception("The type of the model should either be LSTM, GRU, CNN or LSTM-CNN. Input was: " + type)

    return model