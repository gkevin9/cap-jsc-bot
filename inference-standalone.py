import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Concatenate

import argparse
import pickle
import string
import re

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='ask question')

    parser.add_argument('--question',
                        help='question for bot',
                        required=True,
                        type=str)

    args = parser.parse_args()

    return args

def main():
    arg = parse_args()
    question = arg.question

    #load model
    model = tf.keras.models.load_model("s2s", compile=False)

    #load vocab
    dbfile = open('vocabPickle', 'rb')     
    vocab = pickle.load(dbfile)
    inv_vocab = {v: k for k, v in vocab.items()}
    dbfile.close()

    #rebuild model
    latent_dim = 256
    enc_inputs = model.input[0]  # input_1
    enc_outputs, h, c = model.layers[4].output  # lstm1
    enc_states = [h, c]
    enc_model = Model(enc_inputs, enc_states)

    dec_inputs = model.input[1]  # input_2
    dec_state_input_h = Input(shape=(latent_dim,), name="input_3")
    dec_state_input_c = Input(shape=(latent_dim,), name="input_4")
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]

    dec_lstm = model.layers[5] #lstm2
    dec_outputs, state_h_dec, state_c_dec = dec_lstm(dec_inputs, initial_state=dec_states_inputs)
    dec_states = [state_h_dec, state_c_dec]
    dec_dense = model.layers[6] #dense layer
    dec_outputs = dec_dense(dec_outputs)
    decoder_model = Model([dec_inputs] + dec_states_inputs, [dec_outputs] + dec_states)

    #preprocess
    sequence = preprocessing(question, vocab)

    #decode
    states_value = enc_model.predict(sequence)

    target_seq = np.zeros((1, 1, len(vocab)))
    target_seq[0, 0, vocab["<start>"]] = 1.0

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = inv_vocab[sampled_token_index]
        
    
        if sampled_char == "<pad>" or sampled_char == "<end>":
            break
        
        decoded_sentence += sampled_char + " "
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, len(vocab)))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    
    print("Hasilnya :", decoded_sentence)

def preprocessing(question, vocab):
    inputArr = question.split(" ")
    removed = removePunctuation(inputArr)
    toLower = makeLower(removed)

    enc_input_data = np.zeros(
        (1, len(inputArr), len(vocab)), dtype="float32"
    )
    a = np.zeros((1,1,len(vocab)))
    # for i in range(len(inputArr)):
    #     enc_input_data[0, i, vocab[inputArr[i]]] = 1.0
    for i in range(len(inputArr)):
        if inputArr[i] not in vocab:
            continue
        else:
            a[0,0,vocab[inputArr[i]]] = 1.0

    return a

def makeLower(list):
    result = []
    for i in list:
        result.append(i.lower())
        
    return result

def removePunctuation(list):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    result = []
    for i in list:
        temp = regex.sub('', i)
        result.append(temp.lower())
        
    return result

if __name__ == '__main__':
    main()