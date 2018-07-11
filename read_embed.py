import time
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText

class Embeddings():

	def __init__(self):
		self.embed_type  = None
		self.embed_dim   = 0
  

	def read_model(self,type,cat):
    
		print(' ')

		start = time.time()

		if type=='indo':
			if cat == 'w2v':
				print('Loading Indonesian Word Embedding')
				embeddings_google = 'modelapik.bin'
				embeddings_path = embeddings_google
				print('Loading Word2Vec')
				word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path, binary=True, unicode_errors='ignore')
			else:
				print('Loading Indonesian Word Embedding')
				embeddings_google = 'bjn.bin'
				embeddings_path = embeddings_google
				print('Loading Fasttext')
				word2vec_model = FastText.load_fasttext_format(embeddings_path)
		else:
			print('Loading Google Word Embedding')
			embeddings_google = 'google.bin'
			embeddings_path = embeddings_google
			print('Loading word2vec')
			word2vec_model = KeyedVectors.load_word2vec_format(embeddings_path, binary="False", unicode_errors='ignore')

		end = time.time()
		print("Word Embedding loading done in {} seconds".format(end - start))

		self.word2vec = word2vec_model
		self.embed_dim = 300

    # x = word2vec_model.wv.vocab
    # print(len(x))

    # exit()

		return word2vec_model

