
from keras.models import Model
from keras.layers import Input, Dense, Reshape, add, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer

from numpy import asarray
import numpy as np

# create some input variables
input_target = Input((1,))
input_context = Input((1,))
vector_dim = 100    #defined by the glove file chosen below

# load the whole embedding into memory - this is a dictionary, we just need a matrix
embeddings_index = dict()
f = open('glove.6B/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# docs = ['art','surrealism','monet','football','table','play']

from get_text import doc_string
docs = doc_string
target_word=docs[0]
print docs


# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
print vocab_size
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
print t.word_index

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print embedding_matrix
print embedding_matrix.shape
embedding = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=1, trainable=False)   #this is the first 10000 rows of the Glove embeddings matrix

target = embedding(input_target)   #gives word vector for integer word
target = Reshape((vector_dim, 1))(target)
print target
print type(target)
print target.shape

context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

similarity = merge([target, context], mode='cos', dot_axes=0)
print similarity
validation_model = Model(input=[input_target, input_context], output=similarity)


target_word_idx = t.word_index[target_word]
top_k = 3
sim = np.zeros((vocab_size,))
word_sim_dict = {}
in_arr1 = np.zeros((1,))
in_arr2 = np.zeros((1,))
in_arr1[0,] = target_word_idx
for i in range(vocab_size):
    in_arr2[0,] = i
    out = validation_model.predict_on_batch([in_arr1, in_arr2])
    sim[i] = out
print sim
nearest = (-sim).argsort()[1:top_k + 1]     #index of top k biggest to smallest excluding the highest which is the actual word
print nearest
log_str = 'Nearest to %s:' % target_word
for k in range(top_k):
    for word in t.word_index:
        if t.word_index[word] == nearest[k]:
            close_word = word
            log_str = '%s %s,' % (log_str, close_word)

print(log_str)


