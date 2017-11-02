import gensim
import logging
from helpers import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Load text data
labeled_texts = read_file("labeledTrainData.tsv")
all_text = labeled_texts + read_file("unlabeledTrainData.tsv")


# Word vector training
print("Training vectors")
all_text = [text['tokens'] for text in all_text]
vectors = gensim.models.Word2Vec(all_text, size=50, min_count=1, workers=4)

# Examples on how to use word vectors

vectors['good']

vectors.most_similar('good')
vectors.most_similar(['hollywood', 'boring'])

vectors.similarity('slow', 'boring')
vectors.similarity('slow', 'fun')

# Man is to King as Woman is to ...
# vec(woman)+vec(king)-vec(man)
vectors.most_similar(['woman','king'], negative=['man'])

# Fall is to Fell as Watch is to ...
vectors.most_similar(['watch','fell'], negative=['fall'])

vectors.most_similar(['London','Italy'], negative=['Rome'])

# Save word vectors
#vectors.save("vectors.w2v")
