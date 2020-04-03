from keras.preprocessing.text import Tokenizer
from gensim.models.fasttext import FastText
import numpy as np
import matplotlib.pyplot as plt
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer
from nltk.corpus.reader.aligned import AlignedCorpusReader
import re,fasttext
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

model = fasttext.load_model("/home/aakash/fil9.bin")
n_features = model.get_dimension()
stemmer = WordNetLemmatizer()
embedding_size = 60
window_size = 200
min_word = 5
down_sampling = 1e-2
#model = sent_tokenize(model)

def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()
            # Replace numbers and symbols with language
        document = document.replace('&', ' and ')
        document = document.replace('@', ' at ')
        document = document.replace('0', ' zero ')
        document = document.replace('1', ' one ')
        document = document.replace('2', ' two ')
        document = document.replace('3', ' three ')
        document = document.replace('4', ' four ')
        document = document.replace('5', ' five ')
        document = document.replace('6', ' six ')
        document = document.replace('7', ' seven ')
        document = document.replace('8', ' eight ')
        document = document.replace('9', ' nine ')

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text



def text_to_vector(text):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = preprocess_text(text)
    words = text.split()
    window = words[:]
    
    x = np.zeros((window_size, n_features))


    for i, word in enumerate(window):
        x[i, :] = model.get_word_vector(word).astype('float32')

    return x






y = text_to_vector('Artificial intelligence, is the most advanced technology of the present era')

for i in enumerate(y):
	print(i)
print(text_to_vector('Artificial intelligence, is the most advanced technology of the present era'))
