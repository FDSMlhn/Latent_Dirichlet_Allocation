from LDA_Gibbs_Sampling import LDA
from itertools import compress
import numpy as np
import util
import os
import re



#filenames for data

#path= os.path.join(os.path.dirname(__file__),'datasets')


class TOPIC_MODEL(object):
	def __init__(self):
		return
	def fit(self,allcontent,num_topic=10,train_prop = 0.9,**kwarg):
		try:
			np.random.seed(kwarg['random_seed'])
		except:
			print('Next time plz add random_seed here!')

		doc_size = len(allcontent)
		index = np.random.binomial(1, p = train_prop, size= doc_size)
		train_index, test_index = index.astype('bool'), (1 - index).astype('bool')

		train_content = list(compress(allcontent,train_index))
		corpus_train,vocab_train = util.corpus2dtm(util.content2corpus(train_content))

		model = LDA(num_topic= num_topic,**kwarg)
		model.fit(corpus_train)

		test_content = list(compress(allcontent,test_index))
		test_rawword = util.content2corpus(test_content)
		test_dtm = []
		for raw in test_rawword:
			dtm =  [raw.count(term) for term in vocab_train]
			diff= len(raw) - sum(dtm)
			if diff >0 :
				dtm[-1] = diff
			test_dtm.append(dtm)
		corpus_test = np.array(test_dtm)
		model.predict(corpus_test)
		model.results['test_index'] = test_index
		model.results['train_index'] = train_index
		model.results['vocabulary'] = vocab_train

		return model

if __name__ == "__main__ ":
	path="/Users/Dreamland/Desktop/Fudan University/2017第二学期/AI/Latent_Dirichlet_Allocation/datasets"
	txt_list = os.listdir(path)[1:]
	doc_size = len(txt_list)
	filenames = [os.path.join(path,txt) for txt in txt_list]
	title = np.array([re.sub(".txt$","", txt) for txt in txt_list])
	content = []
	for i in filenames:
		with open(i,'r') as f:
			content.append(f.read())
	tp_model = TOPIC_MODEL()
	tp_model.fit(content,num_topic= 10)


