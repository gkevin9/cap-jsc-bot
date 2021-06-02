import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Masking, Bidirectional, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import matplotlib.pyplot as plt

import pickle
import string
import re

def main():
    #prepare
    vocab = {}
    vocab_counter = 1
    vocab["<pad>"] = 0
    latent_dim = 256

    #read training data.txt
    line = open("tanya_jawab.txt", encoding="utf-8").read().split("\n")

    #preproces
    q, a = splitQnA(line)
    encode_q, length_q, q_text= preprocessing(q, vocab, False, vocab_counter)
    encode_a, length_a, a_text = preprocessing(a, vocab, True, vocab_counter)

    #encode decode
    encode_q = pad_sequences(encode_q, length_q, padding='post', truncating='post')
    encode_a = pad_sequences(encode_a, length_a, padding='post', truncating='post')
    output_encode_a = []
    for i in encode_a:
        output_encode_a.append(i[1:])
    #yg <start> dihilangin
    output_encode_a = pad_sequences(output_encode_a, length_a, padding='post', truncating='post')

    #model
    enc_input = Input(shape=(None , len(vocab)))
    enc_input_mask = Masking(mask_value=0)(enc_input)
    enc = LSTM(latent_dim, return_state=True)
    enc_output, h, c = enc(enc_input_mask)
    enc_state = [h, c]

    dec_inputs = Input(shape=(None, len(vocab)))
    dec_input_mask = Masking(mask_value=0)(dec_inputs)

    # dec_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
    dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

    dec_outputs, _, _ = dec_lstm(dec_input_mask, initial_state=enc_state)
    dec_dense = Dense(len(vocab), activation="softmax")
    dec_outputs = dec_dense(dec_outputs)

    model = Model([enc_input, dec_inputs], dec_outputs)

    #prepared one hot encode
    enc_input_data = np.zeros(
        (len(q_text), length_q, len(vocab)), dtype="float32"
    )
    dec_input_data = np.zeros(
        (len(a_text), length_a, len(vocab)), dtype="float32"
    )
    dec_target_data = np.zeros(
        (len(a_text), length_a, len(vocab)), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(zip(q_text, a_text)):
        for t, text in enumerate(input_text.split(" ")):
            enc_input_data[i, t, vocab[text]] = 1.0
        enc_input_data[i, t + 1 :, vocab["<pad>"]] = 1.0
        for t, text in enumerate(target_text.split(" ")):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            dec_input_data[i, t, vocab[text]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                dec_target_data[i, t - 1, vocab[text]] = 1.0
        dec_input_data[i, t + 1 :, vocab["<pad>"]] = 1.0
        dec_target_data[i, t:, vocab["<pad>"]] = 1.0

    #build model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    #train
    history = model.fit(
        [enc_input_data, dec_input_data],
        dec_target_data,
        batch_size=1,
        epochs=20,
        validation_split=0.2
    )

    #save model and vocab
    print("==>Saving Model")
    model.save("s2s")
    dbfile = open('vocabPickle', 'wb')

    print("==>Saving Vocab")
    pickle.dump(vocab, dbfile)                     
    dbfile.close()

    print("==>Show Training Accuracy")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def splitQnA(list):
    question = []
    answer = []
    for i in list:
        temp = i.split("=")
        question.append(temp[0])
        answer.append(temp[1])
        
    return question, answer

def removePunctuation(list):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    result = []
    for i in list:
        temp = regex.sub('', i)
        result.append(temp.lower())
        
    return result

def addStartEnd(list):
    result = []
    for i in list:
        result.append("<start> " + i + " <end>")
        
    return result

def tokenizerManual(list, vocab, vocab_counter):
    max_length = 0
    result = []
    for i in list:
        tempList = []
        for word in i.split(" "):
#             tempWord = word.lower()
#             print(tempWord)
            if word not in vocab:
                vocab[word] = vocab_counter
                tempList.append(vocab_counter)
                vocab_counter += 1
            else:
                tempList.append(vocab[word])
        if max_length < len(tempList):
            max_length = len(tempList)
        result.append(tempList)
            
            
    return result, max_length

def makeLower(list):
    result = []
    for i in list:
        result.append(i.lower())
        
    return result

def preprocessing(input, vocab, isA, vocab_counter):
    removed = removePunctuation(input)
    toLower = makeLower(removed)
#     print(toLower)
    if (isA):
        addedStartEnd = addStartEnd(toLower)
    else:
        addedStartEnd = toLower
    # print(addedStartEnd)
    tokenize, length = tokenizerManual(addedStartEnd, vocab, vocab_counter)
    
#     makeSameSize(tokenize, length)
    
    return tokenize, length, toLower

if __name__ == '__main__':
    main()