# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:56:14 2019

@author: марк
"""
import numpy as np
import codecs
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation, concatenate, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger


START_CHAR = '\b'
END_CHAR = '\t'
PADDING_CHAR = '\a'
chars = set( [START_CHAR, '\n', END_CHAR])
with codecs.open('D:\\nnet_info_selection\\app\model_lang\\data\\exemple_to_train.txt', 'r','utf8') as f:
    for line in f:
        chars.update( list(line.strip().lower()) )
char_indices = { c : i for i,c in enumerate(sorted(list(chars)))}
char_indices[PADDING_CHAR] = 0
indices_to_chars = {i : c for c,i in char_indices.items()}
num_chars = len(chars)

def get_one(i, sz):
    res = np.zeros(sz)
    res[i] = 1
    return res


char_vectors = {
        c : (np.zeros(num_chars) if c == PADDING_CHAR else get_one(v, num_chars))
        for c,v in char_indices.items()
        }

sentence_end_markers = set('.!?:;')
sentences = []
current_sentence = ''
with codecs.open('D:\\nnet_info_selection\\app\model_lang\\data\\exemple_to_train.txt', 'r','utf8') as f:
    for line in f:
        s = line.strip().lower()
        if len(s) > 0:
            current_sentence += s + '\n'
        if len(s) == 0 or s[-1] in sentence_end_markers:
            current_sentence = current_sentence. strip()
            if len(current_sentence) > 4:
                sentences.append(current_sentence)
            current_sentence = ''
            
            
def get_matrices(sentences):
    max_sentence_len = np.max([ len(x) for x in sentences ])
    X = np.zeros((len(sentences), max_sentence_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), max_sentence_len, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        char_seq = (START_CHAR + sentence + END_CHAR).ljust(max_sentence_len+1, PADDING_CHAR)
        for t in range(max_sentence_len):
            X[i, t, :] = char_vectors[char_seq[t]]
            y[i, t, :] = char_vectors[char_seq[t+1]]
    return X,y

from keras.layers.wrappers import Bidirectional

vec = Input(shape=(None, num_chars))
L1 = LSTM(output_dim=128, activation='tanh', return_sequences=True)(vec)
L1_d = Dropout(0.2)(L1)
input2 = concatenate([vec, L1_d])
L2 = LSTM(output_dim=128, activation='tanh', return_sequences=True)(input2)
L2_d = Dropout(0.2)(L2)
input3 = concatenate([vec, L2_d])
L3 = LSTM(output_dim=128, activation='tanh', return_sequences=True)(input3)
L3_d = Dropout(0.2)(L3)
input_d = concatenate([L1_d, L2_d, L3_d])
dense3 = TimeDistributed(Dense(output_dim=num_chars))(input_d)
output_res = Activation('softmax')(dense3)
model = Model(input=vec, output=output_res)

model.compile(loss='categorical_crossentropy', optimizer=Adam(clipnorm=1.), metrics=['accuracy'])
model.summary()



test_indices = np.random.choice(range(len(sentences)), int(len(sentences)*0.05))
sentences_train = [ sentences[x] for x in set(range(len(sentences))) - set(test_indices) ]
sentences_test = [ sentences[x] for x in test_indices ]
X_test, y_test = get_matrices(sentences_test)
batch_size = 32
def generate_batch():
    while True:
        for i in range( int(len(sentences_train) / batch_size) ):
            sentences_batch = sentences_train[ i*batch_size : (i+1)*batch_size ]
            yield get_matrices(sentences_batch)


    # порождение нескольких текстов в виде еще одной функции обратного вызова.
output_fname = 'simple_test_net.txt'

from keras.callbacks import Callback
class CharSampler(Callback):
     def __init__(self, char_vectors, model):
         self.char_vectors = char_vectors
         self.model = model
     def on_train_begin(self, logs={}):
         self.epoch = 0
         if os.path.isfile(output_fname):
            os.remove(output_fname)
     def sample( self, preds, temperature=1.0):
         preds = np.asarray(preds).astype('float64')
         preds = np.log(preds) / temperature
         exp_preds = np.exp(preds)
         preds = exp_preds / np.sum(exp_preds)
         probas = np.random.multinomial(1, preds, 1)
         return np.argmax(probas)
     def sample_one(self, T):
         result = START_CHAR
         while len(result)<500:
             Xsampled = np.zeros( (1, len(result), num_chars) )
             for t,c in enumerate( list( result ) ):
                Xsampled[0,t,:] = self.char_vectors[ c ]
             ysampled = self.model.predict( Xsampled, batch_size=1 )[0,:]
             yv = ysampled[len(result)-1,:]
             selected_char = indices_to_chars[ self.sample( yv, T ) ]
#             if selected_char==END_CHAR: 
#                break
             result = result + selected_char
         return result
     def on_epoch_end(self, batch, logs={}):
         self.epoch = self.epoch+1
         if self.epoch % 10 == 0:
            print("\nEpoch %d text sampling:" % self.epoch)
            with open( output_fname, 'a' , encoding='utf-8') as outf:
                 outf.write( '\n===== Epoch %d =====\n' % self.epoch )
                 for T in [ 0.5, 0.6, 0.7]:
                     print('\tsampling, T = %.1f...' % T)
                     for _ in range(2):
                         self.model.reset_states()
                         res = self.sample_one(T)
                         outf.write( '\nT = %.1f\n%s\n' % (T, res[1:]) )                


model_fname = 'language_model'
cb_sampler = CharSampler(char_vectors, model)
cb_logger = CSVLogger(model_fname + '.log')
cb_checkpoint = ModelCheckpoint(filepath='char_checkpoint.h5',
                                  monitor='val_loss', save_best_only=True,)

model.fit_generator( generate_batch(),
   int(len(sentences_train) / batch_size) * batch_size,
   nb_epoch=1000, verbose=True, validation_data = (X_test, y_test),
   callbacks=[cb_logger, cb_sampler, cb_checkpoint] )  


model_json = model.to_json()
# Записываем модель в файл
json_file = open("LSTM_poetry.json", "w")
json_file.write(model_json) 
json_file.close()

model.save_weights("LSTM_poetry.h5")
