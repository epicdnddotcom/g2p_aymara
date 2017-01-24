from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.callbacks import CSVLogger, Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop
import numpy as np
import csv
from six.moves import range
from VocabHandler import VocabHandler

from seq2seq.models import SimpleSeq2Seq
from VocabHandler import VocabHandler
from recurrentshop import *
from keras.models import model_from_json

from utils import *

model_dir = "model/ohmodel7.json"
weights_dir = "model/weights_thrd.h5"

## import dataset from dic
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

##################################
# load json and create model
json_file = open(model_dir, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json, custom_objects={'SimpleSeq2Seq': SimpleSeq2Seq, 'RecurrentContainer': RecurrentContainer})
# load weights into new model
model.load_weights(weights_dir)
print("Loaded model from disk")
 
# evaluate loaded model on test data
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
score = model.evaluate(X_test, y_test, verbose=0)
print "%s: %.2f%%" % (model.metrics_names[1], score[1]*100)

print "test"
###
# Select 10 samples from the validation set at random so we can visualize
# errors
good = 0
for i in range(len(X_test)):
    #ind = np.random.randint(0, len(X_test))
    rowX, rowy = X_test[np.array([i])], y_test[np.array([i])]
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

print "precision: " + str(good) + "/" + str(len(X_test)) + " = " + str(float(good) / len(X_test))

##################################

#####################
# retrain
#####################
print model.summary()

ITER = 6
BATCH_SIZE = 128

checkpoint = ModelCheckpoint(filepath="model/weights_thrd1.h5", verbose=1, save_best_only=True, monitor='val_acc')
csv_logger = CSVLogger('training_thrd.log') # logger
history = LossHistory()
guesses = CorrectGuess(valid, handler, model)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=40, min_lr=0.0001, verbose=1)
#######################################################################
model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=ITER,
            validation_data=(X_valid, y_valid), callbacks=[csv_logger, history, checkpoint, guesses, reduce_lr])

###
# Select 10 samples from the validation set at random so we can visualize
# errors

print "modelo entrenado!"