'''
Source: https://github.com/keon/pointer-networks
With Keras 2.0: https://github.com/zygmuntz/pointer-networks-experiments/tree/keras-2.0
Run at Keras 2.0.0, Tensorflow 1.15.2
'''
import keras.backend as K
from keras.activations import tanh, softmax
from keras.engine import InputSpec
from keras.layers import LSTM,Input
from keras.layers.recurrent import Recurrent,initializers
from keras.layers.recurrent import _time_distributed_dense
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import LearningRateScheduler
from seq_data import getTrainData
import numpy as np

import matplotlib.pyplot as plt

class PointerLSTM(LSTM):
    def __init__(self, hidden_shape, *args, **kwargs):
        self.hidden_shape = hidden_shape
        self.input_length = []
        super(PointerLSTM, self).__init__(*args, **kwargs)

    def get_initial_states(self, x_input):
        return Recurrent.get_initial_states(self, x_input)

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        # init = initializations.get('orthogonal')
        self.W1 = self.add_weight(name="W1",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)
        self.W2 = self.add_weight(name="W2",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)
        self.vt = self.add_weight(name="vt",
                                  shape=(input_shape[1], 1),
                                  initializer='uniform',
                                  trainable=True)
        self.trainable_weights += [self.W1, self.W2, self.vt]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1] - 1, :]
        x_input = K.repeat(x_input, input_shape[1])
        initial_states = self.get_initial_states(x_input)

        constants = super(PointerLSTM, self).get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])

        return outputs

    def step(self, x_input, states):
        # print "x_input:", x_input, x_input.shape
        # <TensorType(float32, matrix)>

        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = super(PointerLSTM, self).step(x_input, states[:-1])

        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        Eij = _time_distributed_dense(en_seq, self.W1, output_dim=1)
        Dij = _time_distributed_dense(dec_seq, self.W2, output_dim=1)
        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)

        # make probability tensor
        pointer = softmax(U)
        return pointer, [h, c]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])

class PointerLSTM_keon(LSTM):
    def __init__(self, hidden_shape, *args, **kwargs):
        self.hidden_shape = hidden_shape
        self.input_length = []
        super(PointerLSTM_keon, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(PointerLSTM_keon, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        self.W1 = self.add_weight(name="W1",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)
        self.W2 = self.add_weight(name="W2",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)
        self.vt = self.add_weight(name="vt",
                                  shape=(input_shape[1], 1),
                                  initializer='uniform',
                                  trainable=True)
        self.trainable_weights += [self.W1, self.W2, self.vt]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1]-1, :]
        x_input = K.repeat(x_input, input_shape[1])
        initial_states = self.get_initial_states(x_input)

        constants = super(PointerLSTM_keon, self).get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])

        return outputs

    def step(self, x_input, states):
        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = super(PointerLSTM_keon, self).step(x_input, states[:-1])

        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        Eij = _time_distributed_dense(en_seq, self.W1, output_dim=1)
        Dij = _time_distributed_dense(dec_seq, self.W2, output_dim=1)
        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)

        # make probability tensor
        pointer = softmax(U)
        return pointer, [h, c]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])

class HistoryShow(object):
    '''损失情况'''
    def showLoss(self,history):
        val_loss_values = history.history['val_loss']
        loss_values = history.history['loss']
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, 'bo', color='#A6C8E0',label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', color='#A6C8E0',label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.clf()
    
    '''准确率'''
    def showAccr(self,history):
        acc = history.history['acc']
        epochs = range(1, len(acc) + 1)
        val_acc = history.history['val_acc']
        plt.plot(epochs, acc, 'bo', color='#A6C8E0',label='Training acc')
        plt.plot(epochs, val_acc, 'b', color='#A6C8E0',label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()



def network1():
    print("preparing dataset...")
    X, Y , x_test, y_test= getTrainData(128)
    YY = []
    for y in Y:
        YY.append(to_categorical(y))
    YY = np.asarray(YY)
    YY_test = []
    for y in y_test:
        YY_test.append(to_categorical(y))
    YY_test = np.asarray(YY_test)
    print("building model...")
    hidden_size = 128 # 隐藏层 用于记忆和储存过去状态的节点个数
    n_steps = 5 # 输入层 先后依次输入的次数 (多边形个数?)
    batch_size = 128 # 一次性输入的样本数
    epochs = 20 # 迭代次数
    weights_file = 'model_weights/model_weights_{}_steps_{}.hdf5'.format(n_steps, hidden_size)

    main_input = Input(shape=(n_steps,X.shape[2]), name='main_input')
    encoder = LSTM(units=hidden_size, return_sequences=True, name="encoder")(main_input)
    decoder = PointerLSTM(hidden_size, units=hidden_size, name="decoder")(encoder)
    # Model类：函数式(Functional)模型 比Sequential更广泛
    model = Model(inputs=main_input, outputs=decoder)

    print("loading weights from {}...".format(weights_file))
    try:
        model.load_weights(weights_file)
    except IOError:
        print("no weights file, starting anew.")

    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    print('training and saving model weights each epoch...')

    validation_data = (x_test,YY_test)

    epoch_counter = 0

    history = model.fit(X, YY, epochs=epochs, batch_size=batch_size,
                        validation_data=validation_data)

    p = model.predict(x_test)
    for y_, p_ in list(zip(y_test, p))[:20]:
        print("epoch_counter: ", epoch_counter)
        print("y_test:", y_)
        print("p:     ", p_.argmax(axis=1))
        print()
    #model.save(weights_file)
    h=HistoryShow()
    
    h.showLoss(history)
    h.showAccr(history)

def network2():
    def scheduler(epoch):
        if epoch < nb_epochs/4:
            return learning_rate
        elif epoch < nb_epochs/2:
            return learning_rate*0.5
        return learning_rate*0.1

    print("preparing dataset...")
    X, Y , x_test, y_test= getTrainData(128)
    yy = []
    for y in Y:
        yy.append(to_categorical(y))
    yy = np.asarray(yy)
    yy_test = []
    for y in y_test:
        yy_test.append(to_categorical(y))
    yy_test = np.asarray(yy_test)
    print("building model...")

    hidden_size = 128
    seq_len = 5
    nb_epochs = 10
    learning_rate = 0.1

    print("building model...")
    main_input = Input(shape=(seq_len,X.shape[2]), name='main_input')

    encoder = LSTM(units = hidden_size, return_sequences = True, name="encoder")(main_input)
    decoder = PointerLSTM_keon(hidden_size, output_dim=hidden_size, name="decoder")(encoder)

    model = Model(inputs=main_input, outputs=decoder)
    model.compile(optimizer='adadelta',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X, yy, nb_epoch=nb_epochs, batch_size=64,callbacks=[LearningRateScheduler(scheduler),])
    print(model.predict(x_test))
    print("------")
    print(to_categorical(y_test))
    model.save_weights('model_weight_100.hdf5')

network1()