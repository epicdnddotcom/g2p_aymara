from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.callbacks import CSVLogger, Callback
import numpy as np
import csv
from six.moves import range
from VocabHandler import VocabHandler

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


handler = VocabHandler("dic_datasets/aymara.dic")

train = handler.getTrain(padded=True, one_hot=True)
X_train = train["X"]
y_train = train["y"]

valid = handler.getValid(padded=True, one_hot=True)
X_valid = valid["X"]
y_valid = valid["y"]

test = handler.getTest(padded=True, one_hot=True)
X_test = test["X"]
y_test = test["y"]

# Parameters for the model and dataset

INVERT = False
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 512
BATCH_SIZE = 64
LAYERS = 1
MAXLEN = handler.max_input_length


model_dir = "model/ohmodel5.json"
weights_dir = "model/ohmodel5.h5"
print(X_train.shape)
print(y_train.shape)


print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, handler.gr_size)))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(handler.max_output_length))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should
# be chosen
model.add(TimeDistributed(Dense(handler.ph_size)))
model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

print model.summary()
print "Salvando modelo en: " + model_dir
model_json = model.to_json()
with open(model_dir, "w") as json_file:
    json_file.write(model_json)

# Train the model each generation and show predictions against the
# validation dataset

hist = []
for iteration in range(1, 600):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    csv_logger = CSVLogger('training.log') # logger

    history = LossHistory()
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_valid, y_valid), callbacks=[csv_logger, history])
    hist.append(history.losses)
    ###
    # Select 10 samples from the validation set at random so we can visualize
    # errors
    for i in range(20):
        aciertos = 0
        ind = np.random.randint(0, len(X_valid))
        rowX, rowy = X_valid[np.array([ind])], y_valid[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        # print preds
        #oneh = handler.onehot(rowX[0], handler.gr_size)
        w = handler.decodeWord(np.argmax(rowX[0], axis=1))
        correct = handler.decodePhoneme(np.argmax(rowy[0], axis=1))
        guess = handler.decodePhoneme(preds[0])
        #print('W', w[::-1] if INVERT else w)
        #print('T', correct)
        #print('ok' if correct == guess else 'fail', guess)
        if correct == guess:
            aciertos += 1
        #print('---')
    
    print "aciertos: " + str(aciertos) + "/20"
    model.save_weights(weights_dir)
    print "modelo salvado!"
    #number = input("continue...")

print "modelo entrenado!"


with open("loss_h.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(hist)
print "test"
###
# Select 10 samples from the validation set at random so we can visualize
# errors
good = 0
for i in range(len(X_test)):
    #ind = np.random.randint(0, len(X_test))
    rowX, rowy = X_valid[np.array([i])], y_valid[np.array([i])]
    preds = model.predict_classes(rowX, verbose=0)
    # print preds
    #oneh = handler.onehot(rowX[0], handler.gr_size)
    w = handler.decodeWord(np.argmax(rowX[0], axis=1))
    correct = handler.decodePhoneme(np.argmax(rowy[0], axis=1))
    guess = handler.decodePhoneme(preds[0])
    #print('W', w[::-1] if INVERT else w)
    #print('T', correct)
    if correct == guess:
        good += 1
    # print('ok' if correct == guess else 'fail', guess)
    # print('---')

print "precision: " + str(good) + "/" + str(len(X_test))