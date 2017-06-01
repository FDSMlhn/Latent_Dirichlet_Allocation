from LDA_Gibbs_Sampling import lda
import numpy as np
import util
import os
TRAIN_PROP = 0.75 

#filenames for data

#path= os.path.join(os.path.dirname(__file__),'datasets')

#for notebook
path="/Users/Dreamland/Desktop/Fudan University/2017第二学期/AI/Latent_Dirichlet_Allocation/datasets"


txt_list = os.listdir(path)[1:]
doc_size = len(txt_list)
index = np.random.binomial(1, p = TRAIN_PROP, size= doc_size)
train_index, test_index = index.astype('bool'), (1 - index).astype('bool')


txt_list_train =  list(np.array(txt_list)[train_index])
filenames = [os.path.join(path,txt) for txt in txt_list_train]
corpus_train,vocab_train = util.file2corpus(filenames)
title_train = np.array([re.sub(".txt$","", txt) for txt in txt_list])

model = lda(num_topic= 10)
model.fit(corpus_train)






#modelling