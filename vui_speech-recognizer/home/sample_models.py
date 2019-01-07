from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,
    MaxPooling1D, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif border_mode == 'causal':   #NP - added causal mode for dilated cnn
        output_length = input_length - dilated_filter_size + 1   #NP - dilated cnn
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, recur_type=SimpleRNN, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    assert recur_layers >= 1
    for i in range(1, recur_layers+1):
        rnn_i = recur_type(units, activation='relu',
                         return_sequences=True, implementation=2, dropout=0, recurrent_dropout=0, 
                         name='rnn'+str(i))(bn_rnn_i if i > 1 else input_data)
        bn_rnn_i = BatchNormalization(name='bn_rnn'+str(i))(rnn_i)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_i)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='bidir-rnn'))(input_data)
    bn_bidir_rnn = BatchNormalization(name='bn_bidir_rnn')(bidir_rnn)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn_birnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, rnn_type=SimpleRNN, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    bidir_rnn = Bidirectional(rnn_type(units, activation='relu',
        return_sequences=True, implementation=2, name='bidir-rnn'))(bn_cnn)
    bn_bidir_rnn = BatchNormalization(name='bn_bidir_rnn')(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def dilated_cnn_birnn_model(input_dim, filters, kernel_size, units, 
                            output_dim=29, dilation=2):
    """ Build a dilated cnn + bidirectional rnn network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_border_mode = 'causal'
    conv_stride = 1
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     dilation_rate=dilation,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='bidir-rnn'))(bn_cnn)
    bn_bidir_rnn = BatchNormalization(name='bn_bidir_rnn')(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, dilation)
    print(model.summary())
    return model

def deep_cnn_birnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer 1
    conv_1d_1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d_1')(input_data)
    # Add batch normalization
    #bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(conv_1d_1)
    #dropout_cnn_1 = Dropout(0.2, name='dropout_cnn_1')(conv_1d_1)
    # Add max pooling
    maxpool_cnn_1 = MaxPooling1D(pool_size=2, name='max_pool_1D_1')(conv_1d_1)
    # Add second convolutional layer    
    conv_1d_2 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d_2')(maxpool_cnn_1)
    # Add batch normalization
    #bn_cnn_2 = BatchNormalization(name='bn_conv_1d_2')(conv_1d_2)
    # Add max pooling
    maxpool_cnn_2 = MaxPooling1D(pool_size=2, name='max_pool_1D_2')(conv_1d_2)

    # Add a recurrent layer
    bidir_rnn_1 = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='bidir-rnn_1'))(maxpool_cnn_2)
    bn_bidir_rnn_1 = BatchNormalization(name='bn_bidir_rnn_1')(bidir_rnn_1)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_bidir_rnn_1)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)/2,  #NP maxpool size is 2
            kernel_size, conv_border_mode, conv_stride)/2 #NP 2 conv layers
    print(model.summary())
    return model

def deep_cnn_deep_birnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer 1
    conv_1d_1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d_1')(input_data)
    # Add batch normalization
    #bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(conv_1d_1)
    #dropout_cnn_1 = Dropout(0.2, name='dropout_cnn_1')(conv_1d_1)
    # Add max pooling
    maxpool_cnn_1 = MaxPooling1D(pool_size=2, name='max_pool_1D_1')(conv_1d_1)
    # Add second convolutional layer    
    conv_1d_2 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d_2')(maxpool_cnn_1)
    # Add batch normalization
    #bn_cnn_2 = BatchNormalization(name='bn_conv_1d_2')(conv_1d_2)
    # Add max pooling
    maxpool_cnn_2 = MaxPooling1D(pool_size=2, name='max_pool_1D_2')(conv_1d_2)

    # Add a recurrent layer
    bidir_rnn_1 = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='bidir-rnn_1'))(maxpool_cnn_2)
    bn_bidir_rnn_1 = BatchNormalization(name='bn_bidir_rnn_1')(bidir_rnn_1)
    # Add a recurrent layer
    bidir_rnn_2 = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='bidir-rnn_2'))(bn_bidir_rnn_1)
    bn_bidir_rnn_2 = BatchNormalization(name='bn_bidir_rnn_2')(bidir_rnn_2)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_bidir_rnn_2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)/2,  #NP pool size is 2
            kernel_size, conv_border_mode, conv_stride)/2 #NP 2 conv layers
    print(model.summary())
    return model


def cnn_deep_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, recur_layers=2, rnn_type=SimpleRNN, rnn_dropout=0, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add recurrent layers
    assert recur_layers >= 1
    for i in range(1, recur_layers+1):
        simp_rnn_i = rnn_type(units, activation='relu',
                         return_sequences=True, implementation=2, dropout=rnn_dropout, recurrent_dropout=rnn_dropout, 
                         name='rnn'+str(i))(bn_cnn if i == 1 else bn_rnn_i)
        bn_rnn_i = BatchNormalization(name='bn_rnn'+str(i))(simp_rnn_i)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_i)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_deep_birnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, recur_layers=2, rnn_type=SimpleRNN, rnn_dropout=0, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add recurrent layers
    assert recur_layers >= 1
    for i in range(1, recur_layers+1):
        birnn_i = Bidirectional(rnn_type(units, activation='relu',
                         return_sequences=True, implementation=2, dropout=rnn_dropout, recurrent_dropout=rnn_dropout, 
                         name='birnn'+str(i)))(bn_cnn if i == 1 else bn_birnn_i)
        bn_birnn_i = BatchNormalization(name='bn_birnn'+str(i))(birnn_i)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_birnn_i)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, recur_layers, rnn_type, output_dim=29):
    
    #the final model is cnn_deep_birnn_model
    rnn_dropout = 0
    model = cnn_deep_birnn_model(input_dim, filters, kernel_size, conv_stride,
                                 conv_border_mode, units, recur_layers, rnn_type, rnn_dropout, output_dim)
    return model