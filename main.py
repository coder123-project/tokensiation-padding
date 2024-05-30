import pandas as pd

#read excel file
train_data_raw = pd.read_excel("text-emotion-training-dataset.xlsx")

#display first five entries of training dataset
print(train_data_raw)

train_data = pd.DataFrame(train_data_raw["Text_Emotion"].str.split(";",1).tolist()
                          ,columns = ['Text','Emotion'])
print(train_data)


train_data["Emotion"].unique()
#Create a Dictionary to replace emotions with labels
encode_emotions = {"anger": 0, "fear": 1, "joy": 2, "love": 3, "sadness": 4, "surprise": 5}
train_data.replace(encode_emotions, inplace = True)
print(train_data)

training_sentences = []
training_labels = []



# append text and emotions in the list using the 'loc' method

for i in range(len(train_data)):
  
  sentence = train_data.loc[i, "Text"]
  training_sentences.append(sentence)

  label = train_data.loc[i, "Emotion"]
  training_labels.append(label)

print(training_sentences[100], training_labels[100])


import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

#Define parameters for Tokenizer

vocab_size = 10000 # words size
embedding_dim = 16 # array size for the words
oov_tok = "<OOV>" # out of vocabulory 
training_size = 20000

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

#Create a word_index dictionary

word_index = tokenizer.word_index
print("hi")
training_sequences = tokenizer.texts_to_sequences(training_sentences)

training_sequences[0:2]
print(training_sequences[0])
print(training_sequences[1])
print(training_sequences[2])


from tensorflow.keras.preprocessing.sequence import pad_sequences

padding_type='post'
max_length = 100
trunc_type='post'


training_padded = pad_sequences(training_sequences, maxlen=max_length, 
                                padding=padding_type, truncating=trunc_type)

print(training_padded)

