import gensim
import re
import random
import csv
import logging
import numpy as np
import theanets
from sklearn.metrics import classification_report, confusion_matrix


def preproc(text):
    """ Function for text preprocessing and tokenization """
    text = text.lower().decode('utf-8').replace("<br />", "\n")
    text = text.replace("!"," !\n").replace("?"," ?\n")\
        .replace("."," .\n").replace(",","")\
        .replace(":"," : ").replace(";"," ;")\
        .replace("\"", " \" ").replace("("," ( ").replace(")"," ) ")
    tokens = text.split()
    return tokens


def read_file(file_name):
    """ Function for reading CSV data file """
    print "Reading", file_name
    data = []
    csvfile = open(file_name, 'rb')
    reader = csv.DictReader(csvfile, delimiter="\t")
    for line in reader:
        line['tokens'] = preproc(line['review'])
        data.append(line)
    # Randomize order
    data = random.sample(data, len(data))
    return data


### Main program

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load text data
train_texts = read_file("labeledTrainData.tsv")
all_text = train_texts + read_file("unlabeledTrainData.tsv")# + read_file("blindTestData.tsv")


## Word vector training

print "Training vectors"
vectors = gensim.models.Word2Vec([x['tokens'] for x in all_text], size=100, window=8, min_count=2, workers=4)

# Alternative: train document vectors
#vectors = gensim.models.Doc2Vec([gensim.models.doc2vec.TaggedDocument(words=w, tags=[i]) for i, w in enumerate([x['tokens'] for x in all_text])], size=300, window=8, min_count=2, workers=4)

# Examples on how to use word vectors

vectors['good']
vectors.most_similar('good')
vectors.most_similar(['hollywood', 'boring'])
vectors.similarity('slow', 'boring')
vectors.most_similar(['woman','king'], negative=['man'])
vectors.most_similar(['saw','liked'], negative=['like'])

# Save word vectors
#vectors.save("vectors.d2v")
# Load word vectors
#vectors = gensim.models.Word2Vec.load("vectors.d2v")


## Supervised learning

# Prepare data for theanets (prediction)
inputs = []
outputs = []

for i, sample in enumerate(train_texts):
    if i % 1000 == 0:
        print "Generating vector representation of samples: %.1f%%" % (float(i)/len(train_texts)*100)
    # Get vectors for tokens
    vecs = np.transpose([vectors[token] for token in sample['tokens'] if token in vectors])
    # Create bag-of-vectors representation (average vector)
    avg_vec = np.array(map(np.average, vecs))
    ## Alt: get document vector
    #avg_vec = vectors.docvecs[i]
    inputs.append(avg_vec)
    outputs.append(int(sample['sentiment']))


inputs = np.array(inputs)
inputs = inputs.astype(np.float32)
outputs = np.array(outputs)
outputs = outputs.astype(np.int32)

# Specify learning parameters

method          = 'rprop'
learning_rate   = 0.001
momentum        = 0.5
regularization  = 0.0001
hidden          = 20


split_point1 = int(len(inputs)*0.8)
split_point2 = int(len(inputs)*0.9)
train_data = (inputs[:split_point1], outputs[:split_point1])
valid_data = (inputs[split_point1:split_point2], outputs[split_point1:split_point2])
test_data = (inputs[split_point2:], outputs[split_point2:])

# Create network
exp = theanets.Experiment(theanets.Classifier, layers=(len(inputs[0]), hidden, 2))

# Train network
exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            hidden_l1=regularization,
                                            min_improvement=0.05,
                                            validate_every=5,
                                            patience=2
                                            )


## Evaluation

confmx = confusion_matrix(exp.network.predict(test_data[0]), test_data[1])
accuracy = float(sum(np.diag(confmx)))/sum(sum(confmx))

print classification_report(exp.network.predict(test_data[0]),test_data[1]), "\nAverage accuracy:", accuracy
print "Confusion matrix:\n", confmx

