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

def getModelStructure(type, n_steps_in, n_steps_out, n_features, n_nodes, n_nodes_cnn):
    if type == "LSTM":
        model = Sequential([
            LSTM(n_nodes, activation='relu', input_shape=(n_steps_in, n_features)),
            Dense(n_steps_out, activation='relu'),
        ])
    elif type == "GRU":
        model = Sequential([
            GRU(n_nodes, activation='relu', input_shape=(n_steps_in, n_features)),
            Dense(n_steps_out, activation='relu'),
        ])
    elif type == "CNN":
        model = Sequential([
            Conv1D(n_nodes_cnn, kernel_size=n_features, activation='relu', input_shape=(n_steps_in, n_features)),
            MaxPooling1D(pool_size=n_features),
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