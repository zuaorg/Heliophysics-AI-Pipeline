# =========================================================================
#   (c) Copyright 2020
#   All rights reserved
#   Programs written by Hao Liu
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================
import pandas as pd
from sklearn.utils import class_weight
from keras.models import *
from keras.layers import *
import numpy as np
import sys
import csv
import os
import warnings
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.callbacks import ReduceLROnPlateau
from CME_utils import *
from CME_attention import *


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('turn off loggins is not supported')

# Define the dictionary mapping indexes to column names.
index_to_column = {
    0: "flareClass",
    1: "CMELabel",
    2: "SEPLabel",
    3: "T_REC",
    4: "NOAA_AR",
    5: "HARPNUM",
    6: "TOTUSJH",
    7: "TOTPOT",
    8: "TOTUSJZ",
    9: "ABSNJZH",
    10: "SAVNCPP",
    11: "USFLUX",
    12: "AREA_ACR",
    13: "MEANPOT",
    14: "R_VALUE",
    15: "SHRGT45",
    16: "MEANGAM",
    17: "MEANJZH",
    18: "MEANGBT",
    19: "MEANGBZ",
    20: "MEANJZD",
    21: "MEANGBH",
    22: "MEANSHR",
    23: "MEANALP",
}

column_names = [
    "Flare Prediction", "flareClass", "CMELabel", "SEPLabel", "T_REC", "NOAA_AR", "HARPNUM",
    "TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX",
    "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45", "MEANGAM", "MEANJZH",
    "MEANGBT", "MEANGBZ", "MEANJZD", "MEANGBH", "MEANSHR", "MEANALP", "TOTBSQ"
]

def load_data(datafile, series_len, start_feature, n_features, mask_value, type, time_window):
    df = pd.read_csv(datafile, header=0)
    df_values0 = df.values
    if type == 'gru':
        if time_window == 12:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 13, 7, 8, 15, 18, 21, 6, 9, 10, 17, 5, 16, 4, 12, 19, 20, 14]]  # 12   GRU
        elif time_window == 24:
            # desired_indexes = [1, 3, 4, 5, 13, 15, 17, 7, 22, 11, 23, 9, 10, 8, 19, 20, 12, 16, 6, 14, 18, 21]
            # df_values = df_values0[:, desired_indexes]  # 24   GRU
            # Define the columns you actually need.
            desired_columns = [
                "CMELabel", "T_REC", "NOAA_AR", "HARPNUM", "MEANPOT",
                "SHRGT45", "MEANJZH", "TOTPOT", "MEANSHR", "USFLUX",
                "MEANALP", "ABSNJZH", "SAVNCPP", "TOTUSJZ", "MEANGBZ",
                "MEANJZD", "AREA_ACR", "MEANGAM", "TOTUSJH", "R_VALUE",
                "MEANGBT", "MEANGBH"
            ]
            # Create a list of indices corresponding to each desired column.
            desired_indices = [column_names.index(col) for col in desired_columns]
            # Use advanced indexing to extract the desired columns from the numpy array.
            df_values = df_values0[:, desired_indices]
        elif time_window == 36:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 9, 21, 15, 8, 7, 4, 6, 14, 12, 17, 10, 18, 16, 19]]  # 36   GRU
        elif time_window == 48:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 9, 14, 8, 7, 21, 6, 4, 15, 12, 17, 16, 10, 18, 19]]  # 48   GRU
        elif time_window == 60:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 7, 15, 8, 14, 6, 21, 4, 9, 12, 10, 19, 18, 16, 17]]  # 60   GRU
    elif type == 'lstm':
        if time_window == 12:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 13, 20, 7, 15, 8, 21, 6, 18, 5, 10, 9, 17, 16, 19, 12, 14, 4]]  # 12   LSTM
        elif time_window == 24:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 20, 11, 13, 9, 15, 14, 8, 7, 5, 21, 6, 17, 18, 10, 12, 16, 4, 19]]  # 24   LSTM
        elif time_window == 36:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 20, 13, 5, 14, 8, 15, 7, 9, 21, 6, 4, 12, 17, 18, 10, 16, 19]]  # 36   LSTM
        elif time_window == 48:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 20, 13, 9, 14, 7, 15, 8, 6, 4, 21, 12, 17, 18, 16, 10, 19]]  # 48   LSTM
        elif time_window == 60:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 7, 15, 8, 14, 6, 21, 4, 9, 12, 10, 19, 18, 16, 17]]  # 60   LSTM
    elif type == 'bilstm':
        if time_window == 24:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 4, 8, 17, 7, 5, 13, 12, 20, 11, 15, 18, 16, 19, 14, 21, 10, 6, 9]]  # 24   BiLSTM

    X = []
    y = []
    tmp = []
    for k in range(start_feature, start_feature + n_features):
        tmp.append(mask_value)
    n_neg = 0
    n_pos = 0
    for idx in range(0, len(df_values)):
        each_series_data = []
        row = df_values[idx]
        label = row[0]
        if label == 'padding':
            continue
        has_zero_record = False
        # # if one of the physical feature values is missing, then discard it.
        # for k in range(start_feature, start_feature + n_features):
        #     if float(row[k]) == 0.0:
        #         has_zero_record = True
        #         break

        if has_zero_record is False:
            cur_harp_num = int(row[3])
            each_series_data.append(row[start_feature:start_feature + n_features].tolist())
            itr_idx = idx - 1
            while itr_idx >= 0 and len(each_series_data) < series_len:
                prev_row = df_values[itr_idx]
                prev_harp_num = int(prev_row[3])
                if prev_harp_num != cur_harp_num:
                    break
                has_zero_record_tmp = False
                # for k in range(start_feature, start_feature + n_features):
                #     if float(prev_row[k]) == 0.0:
                #         has_zero_record_tmp = True
                #         break
                # if float(prev_row[-5]) >= 3500 or float(prev_row[-4]) >= 65536 or \
                #         abs(float(prev_row[-1]) - float(prev_row[-2])) > 70:
                #     has_zero_record_tmp = True

                if len(each_series_data) < series_len and has_zero_record_tmp is True:
                    each_series_data.insert(0, tmp)

                if len(each_series_data) < series_len and has_zero_record_tmp is False:
                    each_series_data.insert(0, prev_row[start_feature:start_feature + n_features].tolist())
                itr_idx -= 1

            while len(each_series_data) > 0 and len(each_series_data) < series_len:
                each_series_data.insert(0, tmp)

            if (label == 'N' or label == 'P') and len(each_series_data) > 0:
                X.append(np.array(each_series_data).reshape(series_len, n_features).tolist())
                if label == 'N':
                    y.append(0)
                    n_neg += 1
                elif label == 'P':
                    y.append(1)
                    n_pos += 1
    X_arr = np.array(X)
    y_arr = np.array(y)
    nb = n_neg + n_pos
    return X_arr, y_arr, nb


def attention_3d_block(hidden_states, series_len):
    hidden_size = int(hidden_states.shape[2])
    hidden_states_t = Permute((2, 1), name='attention_input_t')(hidden_states)
    hidden_states_t = Reshape((hidden_size, series_len), name='attention_input_reshape')(hidden_states_t)
    score_first_part = Dense(series_len, use_bias=False, name='attention_score_vec')(hidden_states_t)
    score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
    h_t = Lambda(lambda x: x[:, :, -1], output_shape=(hidden_size, 1), name='last_hidden_state')(hidden_states_t)
    score = dot([score_first_part_t, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states_t, attention_weights], [2, 1], name='context_vector')
    context_vector = Reshape((hidden_size,))(context_vector)
    h_t = Reshape((hidden_size,))(h_t)
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(hidden_size, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector


def lstm(n_features, series_len):
    inputs = Input(shape=(series_len, n_features,))
    lstm_out = LSTM(10, return_sequences=True, dropout=0.5, recurrent_dropout=0.3)(inputs)
    attention_mul = attention_3d_block(lstm_out, series_len)
    layer1 = Dense(100, activation='relu')(attention_mul)
    layer1 = Dropout(0.25)(layer1)
    output = Dense(1, activation='sigmoid', activity_regularizer=regularizers.l2(0.0001))(layer1)
    model = Model(inputs=[inputs], outputs=output)
    return model


def gru(n_features, series_len):
    inputs = Input(shape=(series_len, n_features,))
    lstm_out = GRU(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), dropout=0.3, recurrent_dropout=0.2)(inputs)
    lstm_out = BatchNormalization()(lstm_out)
    attention_mul = attention_3d_block(lstm_out, series_len)
    layer1 = Dense(100, activation='relu')(attention_mul)
    layer1 = Dropout(0.25)(layer1)

    # Additional Dense layer to introduce more non-linearity
    layer2 = Dense(50, activation='relu')(layer1)
    layer2 = Dropout(0.3)(layer2)

    output = Dense(1, activation='sigmoid', activity_regularizer=regularizers.l2(0.0001))(layer2)
    model = Model(inputs=[inputs], outputs=output)
    return model

def bilstm(n_features, series_len):
    # Input shape
    input_layer = Input(shape=(series_len, n_features))
    # Dense layers with regularization
    model = Dense(100, kernel_regularizer=l2(0.001), activation='relu')(input_layer)
    model = Dense(100, kernel_regularizer=l2(0.001), activation='relu')(model)
    # Bidirectional LSTM layers with dropout
    model = Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
                               kernel_regularizer=l2(0.001)))(model)
    model = Bidirectional(LSTM(units=50, return_sequences=True, kernel_regularizer=l2(0.001)))(model)
    model = CMEAttention()(model)
    # Flatten and Dense output layer
    model = Flatten()(model)
    model = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))(model)
    # Batch normalization for stability
    output = BatchNormalization(momentum=0.9)(model)
    model = Model(inputs=input_layer, outputs=output)
    return model


def output_result(test_data_file, result_file, type, time_window, start_feature, n_features, thresh):
    df = pd.read_csv(test_data_file, header=0)
    df_values0 = df.values
    if type == 'gru':
        if time_window == 12:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 13, 7, 8, 15, 18, 21, 6, 9, 10, 17, 5, 16, 4, 12, 19, 20, 14]]  # 12   GRU
        elif time_window == 24:
            # Define the columns you actually need.
            desired_columns = [
                "CMELabel", "T_REC", "NOAA_AR", "HARPNUM", "MEANPOT",
                "SHRGT45", "MEANJZH", "TOTPOT", "MEANSHR", "USFLUX",
                "MEANALP", "ABSNJZH", "SAVNCPP", "TOTUSJZ", "MEANGBZ",
                "MEANJZD", "AREA_ACR", "MEANGAM", "TOTUSJH", "R_VALUE",
                "MEANGBT", "MEANGBH"
            ]
            # Create a list of indices corresponding to each desired column.
            desired_indices = [column_names.index(col) for col in desired_columns]
            # Use advanced indexing to extract the desired columns from the numpy array.
            df_values = df_values0[:, desired_indices]
        elif time_window == 36:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 9, 21, 15, 8, 7, 4, 6, 14, 12, 17, 10, 18, 16, 19]]  # 36   GRU
        elif time_window == 48:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 9, 14, 8, 7, 21, 6, 4, 15, 12, 17, 16, 10, 18, 19]]  # 48   GRU
        elif time_window == 60:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 7, 15, 8, 14, 6, 21, 4, 9, 12, 10, 19, 18, 16, 17]]  # 60   GRU
    elif type == 'lstm':
        if time_window == 12:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 13, 20, 7, 15, 8, 21, 6, 18, 5, 10, 9, 17, 16, 19, 12, 14, 4]]  # 12   LSTM
        elif time_window == 24:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 20, 11, 13, 9, 15, 14, 8, 7, 5, 21, 6, 17, 18, 10, 12, 16, 4, 19]]  # 24   LSTM
        elif time_window == 36:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 20, 13, 5, 14, 8, 15, 7, 9, 21, 6, 4, 12, 17, 18, 10, 16, 19]]  # 36   LSTM
        elif time_window == 48:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 20, 13, 9, 14, 7, 15, 8, 6, 4, 21, 12, 17, 18, 16, 10, 19]]  # 48   LSTM
        elif time_window == 60:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 11, 5, 13, 20, 7, 15, 8, 14, 6, 21, 4, 9, 12, 10, 19, 18, 16, 17]]  # 60   LSTM
    elif type == 'bilstm':
        if time_window == 24:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 4, 8, 17, 7, 5, 13, 12, 20, 11, 15, 18, 16, 19, 14, 21, 10, 6, 9]]  # 24   BiLSTM
    elif type == 'ens':
        if time_window == 24:
            df_values = df_values0[:,
                        [0, 1, 2, 3, 4, 8, 17, 7, 5, 13, 12, 20, 11, 15, 18, 16, 19, 14, 21, 10, 6, 9]]  # 24   BiLSTM

    with open(result_file, 'w', encoding='UTF-8', newline='') as result_csv:
        w = csv.writer(result_csv)
        w.writerow(['Predicted Flare', 'flareClass', 'Predicted CME', 'CMELabel', 'SEPLabel', 'Timestamp', 'NOAA_AR', 'HARPNUM',
                      'TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'AREA_ACR',
                      'MEANPOT', 'R_VALUE', 'SHRGT45', 'MEANGAM', 'MEANJZH', 'MEANGBT', 'MEANGBZ',
                      'MEANJZD', 'MEANGBH', 'MEANSHR', 'MEANALP', "TOTBSQ"])
        idx = 0
        print('test data length', len(df_values))
        print('prob length', len(prob))
        for i in range(len(df_values0)):
            line = df_values0[i].tolist()
            # if line[2] == 'padding' or float(line[-5]) >= 3500 or float(line[-4]) >= 65536 \
            #         or abs(float(line[-1]) - float(line[-2])) > 70:
            #     continue
            # has_zero_record = False
            # # if one of the physical feature values is missing, then discard it.
            # for k in range(start_feature, start_feature + n_features):
            #     if float(line[k]) == 0.0:
            #         has_zero_record = True
            #         break
            # if has_zero_record:
            #     continue
            if prob[idx] >= thresh:
                line.insert(2, 'P')
            else:
                line.insert(2, 'N')

            idx += 1
            w.writerow(line)


def get_n_features_thresh(type, time_window):
    n_features = 0
    thresh = 0
    if type == 'gru':
        if time_window == 12:
            n_features = 16
            thresh = 0.45
        elif time_window == 24:
            n_features = 12
            thresh = 0.5
        elif time_window == 36:
            n_features = 9
            thresh = 0.45
        elif time_window == 48:
            n_features = 14
            thresh = 0.45
        elif time_window == 60:
            n_features = 5
            thresh = 0.5
    elif type == 'lstm':
        if time_window == 12:
            n_features = 15
            thresh = 0.4
        elif time_window == 24:
            n_features = 12
            thresh = 0.6
        elif time_window == 36:
            n_features = 8
            thresh = 0.45
        elif time_window == 48:
            n_features = 15
            thresh = 0.45
        elif time_window == 60:
            n_features = 6
            thresh = 0.5
    elif type == 'bilstm':
        if time_window == 24:
            n_features = 15
            thresh = 0.55
    elif type == 'ens':
        if time_window == 24:
            n_features = 15
            thresh = 0.55
    return n_features, thresh


if __name__ == '__main__':
    type = sys.argv[1]
    time_window = int(sys.argv[2])
    train_again = int(sys.argv[3])
    # train_data_file = './normalized_training_' + str(time_window) + '.csv'
    # test_data_file = './normalized_testing_' + str(time_window) + '.csv'
    train_data_file = '../train-data/cme_train.csv'
    test_data_file = '../flare-prediction/results/ENS_result.csv'
    result_file = './' + type + '-' + str(time_window) + '-output.csv'
    model_file = './' + type + '-' + str(time_window) + '-model.h5'
    start_feature = 6
    n_features, thresh = get_n_features_thresh(type, time_window)
    mask_value = 0
    series_len = 20
    epochs = 20
    batch_size = 256
    nclass = 2

    if train_again == 1:
        # Train
        print('loading training data...')
        if type == 'ens':
            X_train, y_train, nb_train = load_data(datafile=train_data_file,
                                                   series_len=series_len,
                                                   start_feature=start_feature,
                                                   n_features=n_features,
                                                   mask_value=mask_value,
                                                   type=type,
                                                   time_window=time_window)

            class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(y_train),
                                                              y=y_train)
            class_weight_ = {0: class_weights[0], 1: class_weights[1]}
        else:
            X_train, y_train, nb_train = load_data(datafile=train_data_file,
                                                   series_len=series_len,
                                                   start_feature=start_feature,
                                                   n_features=n_features,
                                                   mask_value=mask_value,
                                                   type=type,
                                                   time_window=time_window)

            class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                            classes=np.unique(y_train),
                                                            y=y_train)
            class_weight_ = {0: class_weights[0], 1: class_weights[1]}
        print('done loading training data...')

        if type == 'gru':
            model = gru(n_features, series_len)
        elif type == 'lstm':
            model = lstm(n_features, series_len)
        elif type == 'bilstm':
            model = bilstm(n_features, series_len)
        print('training the model, wait until it is finished...')
        optimizer = Adam(learning_rate=0.005)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        history = model.fit(X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[reduce_lr],
                            verbose=True,
                            shuffle=True,
                            class_weight=class_weight_)
        print('finished...')
        model.save(model_file)
    else:
        print('loading model...')
        if type == 'bilstm':
            # Load model with custom layer CMEAttention
            with custom_object_scope({'CMEAttention': CMEAttention}):
                model = load_model(model_file)
        elif type == 'ens':
            model_file = './' + 'bilstm' + '-' + str(time_window) + '-model.h5'
            with custom_object_scope({'CMEAttention': CMEAttention}):
                bilstm_model = load_model(model_file)
            model_file = './' + 'lstm' + '-' + str(time_window) + '-model.h5'
            lstm_model = load_model(model_file)
            model_file = './' + 'gru' + '-' + str(time_window) + '-model.h5'
            gru_model = load_model(model_file)
        else:
            model = load_model(model_file)
        print('done loading...')

    # Test
    print('loading testing data')
    if type == 'ens':
        n_features, thresh = get_n_features_thresh('gru', time_window)
        X_test_gru, y_test_gru, nb_test_gru = load_data(datafile=test_data_file,
                                            series_len=series_len,
                                            start_feature=start_feature,
                                            n_features=n_features,
                                            mask_value=mask_value,
                                            type='gru',
                                            time_window=time_window)
        n_features, thresh = get_n_features_thresh('lstm', time_window)
        X_test_lstm, y_test_lstm, nb_test_lstm = load_data(datafile=test_data_file,
                                            series_len=series_len,
                                            start_feature=start_feature,
                                            n_features=n_features,
                                            mask_value=mask_value,
                                            type='lstm',
                                            time_window=time_window)
        n_features, thresh = get_n_features_thresh('bilstm', time_window)
        X_test_bilstm, y_test, nb_test_bilstm = load_data(datafile=test_data_file,
                                            series_len=series_len,
                                            start_feature=start_feature,
                                            n_features=n_features,
                                            mask_value=mask_value,
                                            type='bilstm',
                                            time_window=time_window)

    else:
        X_test, y_test, nb_test = load_data(datafile=test_data_file,
                                            series_len=series_len,
                                            start_feature=start_feature,
                                            n_features=n_features,
                                            mask_value=mask_value,
                                            type=type,
                                            time_window=time_window)
    print('done loading testing data...')
    print('predicting testing data...')
    if type == 'ens':
        prob_gru = gru_model.predict(X_test_gru,
                             batch_size=batch_size,
                             verbose=False,
                             steps=None)
        prob_lstm = lstm_model.predict(X_test_lstm,
                             batch_size=batch_size,
                             verbose=False,
                             steps=None)
        prob_bilstm = bilstm_model.predict(X_test_bilstm,
                             batch_size=batch_size,
                             verbose=False,
                             steps=None)
        # Average the probabilities from all three models (soft voting)
        prob = (prob_gru + prob_lstm + prob_bilstm) / 3
    else:
        prob = model.predict(X_test,
                             batch_size=batch_size,
                             verbose=False,
                             steps=None)

    print(prob.flatten()[:500])
    print('done predicting...')
    print('writing prediction results into file...')
    output_result(test_data_file=test_data_file,
                  result_file=result_file,
                  type=type,
                  time_window=time_window,
                  start_feature=start_feature,
                  n_features=n_features,
                  thresh=thresh)
    print('done testing...')
    print('calculating performance metrics...')
    y_pred = (prob >= thresh).astype(int)  # Threshold for binary classification
    calculate_metrics(y_test, y_pred, prob, type, time_window)
    print('done...')
    from collections import Counter

    print("Class Distribution train:", Counter(y_train))
    print("Class Distribution test:", Counter(y_test))


