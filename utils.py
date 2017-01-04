import csv
import numpy as np
from keras.callbacks import Callback

## helper classes
### callbacks for training tracking
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class CorrectGuess(Callback):
    scores = []
    best = 0
    def __init__(self, valid, handler, model, scores_file="scores.csv"):
        self.X_valid = valid["X"]
        self.y_valid = valid["y"]
        self.handler = handler
        self.model = model
        self.scores_file = scores_file
    def on_epoch_end(self, epoch, logs={}):
        aciertos = 0.0
        for i in range(50):   
            ind = np.random.randint(0, len(self.X_valid))
            rowX, rowy = self.X_valid[np.array([ind])], self.y_valid[np.array([ind])]
            preds = self.model.predict_classes(rowX, verbose=0)
            w = self.handler.decodeWord(np.argmax(rowX[0], axis=1))
            correct = self.handler.decodePhoneme(np.argmax(rowy[0], axis=1))
            guess = self.handler.decodePhoneme(preds[0])
            if correct == guess:
                aciertos += 1.0

        score = aciertos / 50.0  
        print "score: " + str(aciertos) + "/50"
        if aciertos > self.best:
            self.best = aciertos
            print "salvando mejor modelo con aciertos: " + str(aciertos)
            self.model.save_weights("best_acc.h5")  
        self.scores.append(score)
    def on_train_end(self, logs={}):
        print "saving scores..."
        np.savetxt(self.scores_file, np.array(self.scores), delimiter=',')
            