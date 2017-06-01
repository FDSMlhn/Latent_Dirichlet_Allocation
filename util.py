from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os
import re

stpw1= stopwords.words('english')
stpw2= get_stop_words('en')
stpw3= ['th','says','say','said']
STPW = set(stpw1+stpw2+stpw3)

word_lower_bound = 1

def array2list(corpus):
	D,W= corpus.shape
	d_list = []
	w_list = []

	for d in range(D):
		for w in range(W):
			if corpus[d][w] > 0:
				d_list.extend([d for i in range(corpus[d][w])])
				w_list.extend([w for i in range(corpus[d][w])])
	return d_list,w_list

#txt_list = os.listdir(path)[1:]
#filenames = [os.path.join(path,txt) for txt in txt_list]
	
def file2corpus(filename_list):
	vectorizer = CountVectorizer(input = 'filename',lowercase = True,stop_words = 'english')
	dtm = vectorizer.fit_transform(filename_list).toarray()
	vocab = np.array(vectorizer.get_feature_names())
	title  = np.array([re.sub(".txt$","", txt) for txt in txt_list])

	return trans_special_token(dtm,title,vocab)
	#delete other stopwords as well! numbers, individual characters, and 
	#characters appears less than x document will be convert to a specific token
	
def trans_special_token(dtm,title,vocab):
	abandon_index = np.where([re.search(r'(^[a-zA-Z]$)|([0-9])',v) != None for v in vocab])[0]
	dtm_n= np.delete(dtm, abandon_index,axis=1)
	vocab_n = np.delete(vocab, abandon_index,axis=0)

	spec_list = []
	for col in range(dtm_n.shape[1]):
		if len(np.nonzero(dtm_n[:,col])[0]) <=word_lower_bound:
			spec_list.append(col)

	if spec_list:
		temp = dtm_n[:,spec_list]
		dtm_sp = np.delete(dtm_n, spec_list,axis=1)
		dtm_sp = np.append(dtm_sp, temp.sum(1)[np.newaxis].T,axis=1)

		vocab_sp = np.delete(vocab_n, spec_list,axis=0)
		vocab_sp = np.append(vocab_sp, '0s_token0')
		return dtm_sp,title,vocab_sp
	else:
		return dtm_n,title,vocab_n

def content2corpus(doc_list):
	#tokenize + stopwords deleted + word_stemming
	corpus =[]
	sb_stemmer = SnowballStemmer('english')
	tokenizer = RegexpTokenizer(r'\w+')
	for doc in doc_list:
	 	tokens = tokenizer.tokenize(doc.lower())
	 	tokens= [term for term in tokens if term not in STPW]
	 	stemmed_tk = [sb_stemmer(term) for term in tokens]
	 	corpus.append(stemmed_tk)
	
	vectorizer = CountVectorizer(input = 'content',lowercase = True,stop_words = 'english')
	dtm = vectorizer.fit_transform( [' '.join(doc) for doc in corpus] ).toarray()
	vocab = np.array(vectorizer.get_feature_names())
	return dtm, vocab
	
	#document-term matrix










