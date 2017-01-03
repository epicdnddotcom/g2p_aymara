import os
import numpy as np
import seq2seq
from seq2seq.models import SimpleSeq2Seq
from VocabHandler import VocabHandler
from recurrentshop import RecurrentContainer

from keras.models import model_from_json

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

print "dimensiones del dataset:"

print "entrenamiento: " + str(X_train.shape)
print "validacion cruzada " + str(X_valid.shape)
print "testeo: " + str(X_test.shape)
print "-------------------------------"

model = SimpleSeq2Seq(input_dim=X_train.shape[2], 
                        hidden_dim=512, 
                        output_length=handler.max_output_length, 
                        output_dim=y_train.shape[2],
                        depth=(1, 1))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print model.summary()

model_dir = "model/vmodel3.json"
weights_dir = "model/vmodel3.h5"

print "Salvando modelo en: " + model_dir
model_json = model.to_json()
with open(model_dir, "w") as json_file:
    json_file.write(model_json)
print "arquitectura salvada!"
ITER = 500
print "Entrenando modelo: ..."
for i in range(ITER):
    print "iteracion: " + str(i)
    n_ok = 0
    model.fit(X_train, y_train,validation_data=(X_valid, y_valid), nb_epoch=1, batch_size=32, verbose=1)
    for i in range(10):
        ind = np.random.randint(0, len(X_valid))
        rowX, rowy = X_valid[np.array([ind])], y_valid[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        # print preds
        #oneh = handler.onehot(rowX[0], handler.gr_size)
        w = handler.decodeWord(np.argmax(rowX[0], axis=1))
        correct = handler.decodePhoneme(np.argmax(rowy[0], axis=1))
        guess = handler.decodePhoneme(preds[0])
        if correct == guess:
            n_ok += 1
        
        print('W', w)
        print('T', correct)
        print('ok' if correct == guess else 'fail', guess)
        print('---')

    print "aciertos: " + str(n_ok)

    model.save_weights(weights_dir)
    print "modelo salvado!"
    print "-"*50

print "Entrenamiento finalizado!"

scores = model.evaluate(X_test, y_test, verbose=1)
print "accuracy: " + str(scores[1]*100)




