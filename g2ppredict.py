
import numpy as np
import seq2seq
from seq2seq.models import SimpleSeq2Seq
from VocabHandler import VocabHandler
from recurrentshop import *
from keras.models import model_from_json


handler = VocabHandler("dic_datasets/aymara.dic")
#test = handler.getTest(padded=True, one_hot=True)
test = handler.getTest(padded=True, one_hot=True)
X_test = test["X"]
y_test = test["y"]

#model_dir = "model/model.json"
#weights_dir = "model/model.h5"
model_dir = "model/ohmodel4.json"
weights_dir = "model/ohmodel4.h5"
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

print "precision: " + str(good) + "/" + str(len(X_test))

run = True

while run:
    input = raw_input(">>")
    print input
    #sample = handler.encodeWord(input, padded=True, one_hot=True)
    sample = handler.encodeWord(input, padded=True)
    print sample.shape
    sample = np.reshape(sample, (sample.shape[0], sample.shape[1], 1))
    
    #prediction = loaded_model.predict_classes(sample)
    prediction = model.predict(sample)
    
    print prediction.shape
    print prediction[0]
    