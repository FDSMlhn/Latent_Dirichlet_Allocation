import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os
import re

word_lower_bound = 3

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


def path2corpus(path):
	print('sss')
	txt_list = os.listdir(path)[1:]
	filenames = [os.path.join(path,txt) for txt in txt_list]
	vectorizer = CountVectorizer(input = 'filename',lowercase = True,stop_words = 'english')
	dtm = vectorizer.fit_transform(filenames).toarray()
	vocab = np.array(vectorizer.get_feature_names())

	#delete other stopwords as well! numbers, individual characters, and characters appears less than x document.
	abandon_index_1 = np.where(dtm.sum(axis = 0)<=word_lower_bound)[0]
	abandon_index_2 = np.where([re.search(r'(^[a-zA-Z]$)|([0-9])',v) == None for i in vocab])[0]
	abandon_index = np.union1d(abandon_index_1,abandon_index_2)
	print(abandon_index)
	dtm_n= np.delete(dtm, abandon_index,axis=1)
	vocab_n = np.delete(np.array(vectorizer.get_feature_names()), abandon_index,axis=1)
	title  = np.array([re.sub(".txt$","", txt) for txt in txt_list])
	return dtm_n, title, vocab_n