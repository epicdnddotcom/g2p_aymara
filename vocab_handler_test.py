import numpy as np
import seq2seq
from seq2seq.models import SimpleSeq2Seq
from VocabHandler import VocabHandler

handler = VocabHandler("dic_datasets/aymara.dic")

train = handler.getTrain(padded=True)#, one_hot=True)
X_train = train["X"]
y_train = train["y"]

valid = handler.getValid(padded=True)#, one_hot=True)
X_valid = valid["X"]
y_valid = valid["y"]

test = handler.getTest(padded=True)#, one_hot=True)
X_test = test["X"]
y_test = test["y"]

print "dimensiones del dataset:"

print "entrenamiento: " + str(X_train.shape)
print "validacion cruzada " + str(X_valid.shape)
print "testeo: " + str(X_test.shape)
print "-------------------------------"

model = SimpleSeq2Seq(input_dim=X_train.shape[2], 
                        hidden_dim=64, 
                        output_length=handler.max_output_length, 
                        output_dim=y_train.shape[2],
                        depth=(4, 3))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print model.summary()
model.fit(X_train, y_train,validation_data=(X_valid, y_valid), nb_epoch=10, batch_size=32, verbose=1)

scores = model.evaluate(X_test, y_test, verbose=1)
print "accuracy: " + str(scores[1]*100)