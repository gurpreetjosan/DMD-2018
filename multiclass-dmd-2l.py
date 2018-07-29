import os
import re
from pickle import dump
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Bidirectional, TimeDistributed, Dropout, Masking
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder

outFileName="multiclass-dmd-2L-model" 

# load doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def create_tokenizer(descriptions):
    tokenizer = Tokenizer(oov_token="~",filters='',lower=False)
    tokenizer.fit_on_texts(descriptions)
    return tokenizer

def load_data_and_labels(filename):
    sents, labels = [], []
    text=load_doc(filename)
    for line in re.split('\n', text):  
        line = line.strip()  # remove blank spaces
        if len(line) < 1:
            continue
        domain = np.asarray(line.split(','))
        sents.append(" ".join(domain[0])) # insert space between each character
        labels.append(domain[1])
    return sents, labels

# define the model

def define_model():
    input = Input(shape=(max_wrd_len,))
    mask = Masking(mask_value=0)(input)
    model = Embedding(input_dim=chr_vocab_size, output_dim=300, input_length=max_wrd_len, mask_zero=True)(mask) 
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(model) 
    model = Bidirectional(LSTM(units=250, recurrent_dropout=0.1))(model)

    out = Dense(num_classes, activation="softmax")(model)
    model = Model([input], out)
    #load existing weight if exist
    if os.path.isfile(outFileName+"-best.hdf5"):
        model.load_weights(outFileName+"-best.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    plot_model(model, show_shapes=True, to_file=outFileName+'-plot.png')
    return model

#----Main logic starts here

# load dev set
foldername = 'Task-2/training/train.csv' #replace the path of training file
dataset,labels = load_data_and_labels(foldername)
print('Dataset: %d' % len(dataset))
#display distribution of data
plt.hist(np.array(labels),rwidth=0.8)
plt.show()

# prepare tokenizer
tokenizer = create_tokenizer(dataset)
chr_vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % chr_vocab_size)
#save tokenizer
dump(tokenizer, open(outFileName+'-chrtokenizer.pkl', 'wb'))

# determine the maximum sequence length
max_wrd_len = max(len(s.split()) for s in dataset) # max(len(s) for s in dataset)
print('Description Length: %d' % max_wrd_len)
# train-test split
train_descriptions, test_descriptions, train_labels, test_labels = train_test_split(dataset,labels, test_size=0.1, random_state=42)
print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))

trainseq= tokenizer.texts_to_sequences(train_descriptions)
trainseq = pad_sequences(trainseq, maxlen=max_wrd_len)

testseq= tokenizer.texts_to_sequences(test_descriptions)
Xtest = pad_sequences(testseq, maxlen=max_wrd_len)

#convert string labels to numeric
encoder = LabelEncoder()
encoder.fit(train_labels)

dump(encoder, open(outFileName+'-label_encoder.pkl', 'wb'))
num_classes=encoder.classes_.size
train_labels = encoder.transform(train_labels).astype(np.int32)
out_seq = to_categorical([train_labels], num_classes=num_classes)[0]

test_labels = encoder.transform(test_labels).astype(np.int32)
Ytest = to_categorical([test_labels], num_classes=num_classes)[0]

def generator(features, labels, batch_size):
 batch_features = np.zeros((batch_size, max_wrd_len))
 batch_labels = np.zeros((batch_size,num_classes))
 index=-1
 while True:
   for i in range(batch_size):
     index= index+1
     batch_features[i] = features[index]
     batch_labels[i] = labels[index]
   yield batch_features, batch_labels

model =define_model()
filepath=outFileName+"-best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystop=EarlyStopping(monitor='val_acc',patience=3)
callbacks_list = [checkpoint,earlystop]
batch_size=128

# Runs on CPU. 

history =model.fit_generator(generator(trainseq, out_seq, batch_size), samples_per_epoch=50, nb_epoch=15)

#if run on GPU, comment above line and uncomment following line
#history = model.fit(trainseq, out_seq, epochs=15, batch_size=batch_size,verbose=1, callbacks=callbacks_list,validation_split=0.1)

hist = pd.DataFrame(history.history)
dump(hist, open(outFileName+'-history.pkl', 'wb'))


plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.show()
plt.savefig(outFileName + "-graph.png",bbox_inches='tight')

scores = model.evaluate(Xtest,Ytest, batch_size=batch_size)
print('Raw test score:', scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
dump(scores, open(outFileName+'-testscore.pkl', 'wb'))
