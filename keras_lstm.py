from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import optimizers

import numpy as np
import collections
from helpers import *


# Load text data
labeled_texts = read_file("labeledTrainData.tsv")

## Predictive modeling
# Prepare labeled data

# Convert between tokens and index (for one-hot representation)
token2idx = collections.defaultdict(lambda: len(token2idx)+1)

inputs = []
outputs = []
for text in labeled_texts:
    inputs.append([token2idx[token] for token in text['tokens']])
    outputs.append(text['sentiment'])

idx2token = dict((i, t) for t, i in token2idx.items())


max_len = 500 # Set maximum sequence length
inputs = pad_sequences(inputs, maxlen=max_len, dtype='float32')
outputs = to_categorical(outputs)


# Create separate test set
split_point = int(len(inputs)*0.9)
test_inputs = inputs[split_point:]
inputs = inputs[:split_point]
test_outputs = outputs[split_point:]
outputs = outputs[:split_point]


# Create network
model = Sequential()
model.add(Embedding(len(token2idx)+2, 30, input_length=max_len, mask_zero=True))
model.add(LSTM(30, activation='tanh'))
model.add(Dense(outputs.shape[1], activation='softmax'))

# Train network
model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(inputs, outputs, epochs=6, validation_split=0.2, batch_size=128)


## Evaluation
loss, acc = model.evaluate(test_inputs, test_outputs, batch_size=128)
print("Test accuracy:", acc)

model.save("model.keras")

## Apply the model
#from keras.models import load_model
#model = load_model("model.keras")

# Test model with example from test set
#print(' '.join([idx2token[int(x)] for x in test_inputs[17] if x > 0] ))
#model.predict(np.array([test_inputs[17]]))[0][1]


text = """I saw this movie first on the Berlin Film Festival, and I had never seen
Hong Kong cinema before. I felt like sitting in a roller coaster: the action was so
quick, and there wasn't one boring moment throughout the film. It has martial arts,
love, special effects and a fantastic plot. My favorite scene is when the Taoist
drinks, sings and fights for himself - one of the many scenes which stress the
extraordinary musical component of the movie. This film is a definite must!!"""
tokens = [token2idx[token] for token in preproc(text)]
print(model.predict(pad_sequences([tokens], maxlen=max_len)))

text = """I saw this movie first on the Berlin Film Festival, and I had never seen
Hong Kong cinema before. I felt like sitting in a roller coaster: the action was so
quick, and there wasn't one boring moment throughout the film. It has martial arts,
love, special effects and a fantastic plot. My favorite scene is when the Taoist
drinks, sings and fights for himself - one of the many scenes which stress the
extraordinary musical component of the movie."""
tokens = [token2idx[token] for token in preproc(text)]
print(model.predict(pad_sequences([tokens], maxlen=max_len)))


text = """I felt like sitting in a roller coaster: the action was so quick."""
tokens = [token2idx[token] for token in preproc(text)]
print(model.predict(pad_sequences([tokens], maxlen=max_len)))
