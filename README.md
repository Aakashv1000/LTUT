# LTUT
Language Translation using Transformers

# Abstract:
Currently, almost all prominent language translation models are based on complex recurrent or
convolutional neural networks that include an encoder (compresses inputs into a small vector)
and a decoder (converts the vector from the encoder into outputs). The best performing models
also connect the encoder and decoder through an attention mechanism (method of giving
priorities to each word in a sentence). This paper proposes a new simple network architecture,
the Transformer, based solely on attention mechanisms and disregarding the need for any
recurrent structures. We show, through our implementation of this paper that this method works
very well with translation tasks including English and Hindi.


# Dataset Description:
Our task includes dealing with words and characters instead of numbers as inputs and outputs.
Hence, to convert these into numbers, a great deal of data preprocessing is needed. A brief
summary of our information retrieval  is given below:

# 1. Dataset location: 
The text corpus for our task needs a large number of sentences in both
Hindi and English forms. Also, it needs to be open source so that we can use it for our
project. Hence, with the following constraints, we found the IIT Bombay English-Hindi
Corpus to best suit our needs.

# 2. Dataset Information:
The IIT Bombay English-Hindi Corpus is a humongous collection
of around 16,00,000 English and Hindi binary sentences collected from a wide variety of
sources such as TED Talks, Wikipedia articles, etc. The dataset is already divided into
train, dev and test subsets to help our development needs.

# 3. Dataset Segregation: To get an idea of the performance of our model while training, the
dataset is divided into train and dev subsets, so that we can monitor the training progress.
To check the model performance on real-world data, we further divide the dataset to
obtain a test subset.

# 4. Word Embeddings:
Each word in Hindi is represented in the form of a vector known as
word embeddings. These allow us to convert word-sentences into vector-sentences, thus
enabling us to enter these vectors in a neural network.

# 5. Dataset Conversion: 
In this phase, we use the word embeddings to map each word in the
text corpus into its respective vector. We are using the fast-text method by Facebook to
create word embeddings which provide a set of about 30,000 word-vector maps. This was
trained on 350M words from the Wikipedia dataset in the Hindi Language.

# 6. Data Cleaning: 
Along with the required words to be translated, there may be several
unwanted characters such as numbers, ‘(‘, ‘*’, etc. Thus the data has to be cleaned to get
rid of these undesirable members.

# 7. Data Preprocessing:
For acquiring better results, the training data has to be uniform and
regularized. To transform the dataset so as to match these criteria, we apply several
text-preprocessing methods such as sentence-clipping, padding, etc.


# Algorithms:
The language translation is achieved through the state-of-the-art Transformer model architecture
that is composed of 6 encoder and the 6 decoder components. The entire translation pipeline is
described below:
# 1. Self-attention: 
Each input word is multiplied by trainable vectors - namely Q, K and V.
These vectors then follow this formula to attain the required attention for that word.
Attention(Q, K, V ) = softmax(QKT/ sqrt(dk))V

Further, a positional encoding vector is added to the attention to give a sense of context to
the current word with respect to other words in the sentence. This process is repeated 8
times to obtain a combination of 8 different vectors.

# 2. Neural Network: 
This is a collection of several dense layers that transform the
self-attention vectors into a compact form. Residual links and regularization are added to
ensure that the model does not overfit to the training dataset.

# 3. Encoder-decoder Attention: 
This phase is similar to the attention phase observed in the
encoder. But, the main difference is that, while calculating attention, the Q vector is
obtained from one of the previous outputs, whereas the K and V vectors are obtained
from the output of the encoder stack.


Software and Tools: Tensorflow library, RapidMiner, Google Colab.
Programming Language: Python 3.
