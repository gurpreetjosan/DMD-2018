import warnings
import pickle
import re
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

# load doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_test_data(filename):
    sents = []
    text = load_doc(filename)
    for line in re.split('\n', text):  
        line = line.strip()  
        if len(line) < 1:
            continue
        sents.append(" ".join(line))  # insert space between each character
    return sents

# map an integer to a word
def word_for_label_id(integer):
    return encoder.inverse_transform(integer)

max_sent_length = 196

inFileName = "malicious-dmd-2L"
#load encoder
filename = inFileName + '-label_encoder.pkl'
f = open(filename, 'rb')
encoder = pickle.load(f)
f.close()
#load tokenizer
filename = inFileName + '-chrtokenizer.pkl'
f = open(filename, 'rb')
f.close()
#load model
model = load_model(inFileName + '-best.hdf5')
model._make_predict_function()

foldername = 'test/Task 1/testing/Testing 1/test1.txt'  # replace wit test file path
domains = load_test_data(foldername)
rsltfile = open("TeamJosan_Task1-testing1-test1-SubmissionNo-2.csv", 'w') # replace with result file path
print("writing file ")
count = 0
batchsize = min(50000,len(domains))
start = 0
end = batchsize
while end < len(domains):
    testseq = wrdtokenizer.texts_to_sequences(domains[start:end ])
    testseq = pad_sequences(testseq, maxlen=max_sent_length, padding='post')
    yhat = model.predict(testseq, verbose=0)
    yhat = argmax(yhat, axis=1).tolist()  
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        word = word_for_label_id(yhat)
        for j in range(len(yhat)):
            rsltfile.write(domains[start+j].replace(" ", "") + "," + word[j] + "\n")
    start = end
    end = end + batchsize
    print(end)
    if start<len(domains) and end > len(domains):
        end = len(domains)
testseq = wrdtokenizer.texts_to_sequences(domains[start:len(domains)])
testseq = pad_sequences(testseq, maxlen=max_sent_length, padding='post')
yhat = model.predict(testseq, verbose=0)
yhat = argmax(yhat, axis=1).tolist() 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    word = word_for_label_id(yhat)
    for j in range(len(yhat)):
        rsltfile.write(domains[start + j].replace(" ", "") + "," + word[j] + "\n")

rsltfile.close()
