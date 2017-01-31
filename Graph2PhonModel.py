import numpy as np
from keras.models import Sequential, model_from_json
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.callbacks import CSVLogger, Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop

import seq2seq
from seq2seq.models import SimpleSeq2Seq
from recurrentshop import *

import csv
from six.moves import range
from VocabHandler import VocabHandler

import os
##
from utils import *  ## helpers


#model_dir = "model/ohmodel7.json"
weights_dir = "model/eng1.h5"


"""Graph2PhonModel
    model for train and predict conversions between graphemes and phonemes
"""
class Graph2PhonModel(object):

    # Try replacing GRU, or SimpleRNN
    RNN = recurrent.LSTM   ## we are using LSTM cells
    HIDDEN_SIZE = 128
    BATCH_SIZE = 128
    LAYERS = 2                  ## a tuple with(n_enc_layers, n_dec_layers)
    MAXLEN = 0
    opt = RMSprop(lr=0.01) #optimizer

    X_train = []
    y_train = []

    valid = {}

    ITER = 0

    best_weights = ""
    log_dir = "sample.log"
    def __init__(self, dic_file, model_dir=None, train=True, name="sample"):
        self.handler = VocabHandler(dic_file)
        self.model_dir = model_dir
        self.model_json = name + ".json"
        self.model_weights = name + ".h5"
        #self.weights_dir = weights_dir
        self.best_weights = self.model_weights
        self.log_dir = name+".log"
        self.train = train
        self.name = name
        self.model = Sequential()
        self.loadedModel = Sequential()
        self.MAXLEN = self.handler.max_input_length
        #preparing dataset
        self.prepareDatasets()

    def prepareDatasets(self):
        #training set
        train = self.handler.getTrain(padded=True, one_hot=True)
        self.X_train = train["X"]
        self.y_train = train["y"]

        self.valid = self.handler.getValid(padded=True, one_hot=True)
        self.X_valid = self.valid["X"]
        self.y_valid = self.valid["y"]

        test = self.handler.getTest(padded=True, one_hot=True)
        self.X_test = test["X"]
        self.y_test = test["y"]
    
    def prepareModel(self, layers=1, cells=128):
        # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
        # note: in a situation where your input sequences have a variable length,
        # use input_shape=(None, nb_feature).
        self.LAYERS = layers
        self.HIDDEN_SIZE = cells
        self.model.add(self.RNN(self.HIDDEN_SIZE, input_shape=(self.MAXLEN, self.handler.gr_size)))
        #for _ in range(LAYERS[0]):
            #model.add(RNN(HIDDEN_SIZE, return_sequences=True))
        # For the decoder's input, we repeat the encoded input for each time step
        self.model.add(RepeatVector(self.handler.max_output_length))
        # The decoder RNN could be multiple layers stacked or a single layer
        for _ in range(self.LAYERS):
            self.model.add(self.RNN(self.HIDDEN_SIZE, return_sequences=True))
        # For each of step of the output sequence, decide which character should
        # be chosen
        self.model.add(TimeDistributed(Dense(self.handler.ph_size)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        print self.model.summary()
        self.loadedModel = self.model
    
    def saveModel(self):
        print "Salvando modelo en: " + self.model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # Save model's architecture
        model_json = self.model.to_json()

        with open(os.path.join(self.model_dir, self.model_json), "w") as json_file:
            json_file.write(model_json)
        print "modelo salvado!"
    

    def trainModel(self, epoch=600):
        ## for training
        self.ITER = epoch
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_dir,self.best_weights), verbose=1, save_best_only=True)
        
        csv_logger = CSVLogger(os.path.join(self.model_dir,self.log_dir)) # logger
        history = LossHistory()
        guesses = CorrectGuess(self.valid, self.handler, self.model, scores_file=os.path.join(self.model_dir, "scores.csv"))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=160, min_lr=0.0001, verbose=1)
        #######################################################################
        if self.train == False:
            aux_model = self.loadedModel()
        else:
            aux_model = self.model
        
        aux_model.fit(self.X_train, self.y_train, 
                        batch_size=self.BATCH_SIZE, 
                        nb_epoch=self.ITER,
                        validation_data=(self.X_valid, self.y_valid), 
                        callbacks=[csv_logger, history, checkpoint, guesses, reduce_lr])
        
        return aux_model
        print "modelo entrenado!"

    def testModel(self, model):
        good = 0
        output_file = os.path.join(self.model_dir, self.name + ".test")
        test_results = []
        for i in range(len(self.X_test)):
            #ind = np.random.randint(0, len(X_test))
            rowX, rowy = self.X_test[np.array([i])], self.y_test[np.array([i])]
            preds = model.predict_classes(rowX, verbose=0)
            # print preds
            #oneh = handler.onehot(rowX[0], handler.gr_size)
            w = self.handler.decodeWord(np.argmax(rowX[0], axis=1))
            correct = self.handler.decodePhoneme(np.argmax(rowy[0], axis=1))
            guess = self.handler.decodePhoneme(preds[0])
            test_row = w + "\t -> " + guess
            test_row += " \t\t.....ok" if correct == guess else "\t\t.....fail"
            test_row += "\n"
            #print('W', w[::-1] if INVERT else w)
            #print('T', correct)
            if correct == guess:
                good += 1
            # print('ok' if correct == guess else 'fail', guess)
            # print('---')
            test_results.append(test_row)

        print "precision: " + str(good) + "/" + str(len(self.X_test)) + " = " + str(float(good) / len(self.X_test))

        with open(output_file, 'w') as out:
            for row in test_results:
                out.write(row.encode('utf-8'))
        
        return float(good) / len(self.X_test)

    def loadModel(self):
        json_file = open(os.path.join(self.model_dir, self.model_json), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loadedModel = model_from_json(loaded_model_json, 
                                        custom_objects={'SimpleSeq2Seq': SimpleSeq2Seq, 'RecurrentContainer': RecurrentContainer})
        
        self.loadedModel.load_weights(os.path.join(self.model_dir, self.model_weights))
        # evaluate loaded model on test data
        self.loadedModel.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        return self.loadedModel
        print ">modelo cargado exitosamente"


    def predictPhoneme(self, word, model):
        sample = self.handler.encodeWord(word, padded = True, one_hot=True)
        prediction = model.predict_classes(sample)
        phoneme = self.handler.decodePhoneme(prediction[0])
        print word + " -> " + phoneme
        return phoneme
    def runInteractive(self, model):
        print "entering interactive mode, type 'EXIT' if you want to leave the session"
        run = True
        while run:
            input_word = raw_input(">>")
            if input_word == "EXIT":
                run = False
                return
            print input_word

            self.predictPhoneme(input_word, model)