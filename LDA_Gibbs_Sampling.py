import numpy as np
import util
import scipy.special as ss
from scipy.stats import hmean as hm


class LDA(object):
	"""
	The implement of LDA via collapsed gibbs sampling.
    Parameters:
        num_topic : Number of topics
        num_iter : Number of sampling iterations, default = 2000
        alpha : Dirichlet parameter for distribution over topics
        beta : Dirichlet parameter for distribution over words
    Attributes:
        components_` : array, shape = [num_topic, n_features]
            Point estimate of the topic-word distributions (Phi in literature)
        topic_word_` :
            Alias for `components_`
        n_zw_[num_topic, n_features]: Matrix of counts recording topic-word assignments in final iteration.
        n_dz_[n_samples, num_topic]: Matrix of counts recording document-topic assignments in final iteration.
        doc_topic_[n_samples, n_features] Point estimate of the document-topic distributions (Theta in literature)
        n_z_[num_topic]: Array of topic assignment counts in final iteration.

	"""
	def __init__(self, num_topic,alpha=None,beta=0.1,num_iter=2000,random_seed=1):
		#assure no meaningless values
		self.num_topic = num_topic
		self.num_iter = num_iter
		if alpha is None:
			self.alpha = 50/num_topic
		else:
			self.alpha = alpha

		self.beta = beta 
		self.random_seed = random_seed

		assert self.alpha>0
		assert beta >0

	def fit(self, corpus):
		'''
		the corpus is already array(Document-Term Matrix in topic modelling)
		'''
		np.random.seed(self.random_seed)
		self._initialise(corpus)

		#gibbs sampling
		for it in range(self.num_iter):
			self._gibbs_sampling()
			self._log_likelihood(it)
		self._output_prep()



		del self.d_list
		del self.w_list
		del self.z_list
		#del self._n_zw
		#del self._n_z
		del self._n_dz
		del self._n_d

		return self

	def _initialise(self, corpus):
		#Four arrays need to be initialed
		#Topic-Vocabulary, Topic, Document-Topic, Document
		#change a little bit -- yr

		self.D, self.V = D, V = corpus.shape
		n_t = self.num_topic
        
		self._n_zw = np.zeros((n_t, V),dtype=np.intc)
		self._n_z = np.zeros((n_t),dtype = np.intc)
		self._n_dz = np.zeros((D,n_t),dtype=np.intc)
		self._n_d = np.zeros((D),dtype= np.intc)

		self._n = n = int(corpus.sum())
		self.d_list, self.w_list = util.array2list(corpus)
		self.z_list = []
		self.log_likelihood = np.zeros(self.num_iter, dtype=float)

		for i in range(n):
			d= self.d_list[i] 
			w= self.w_list[i] 
			#get topic assignment randomly
			z = np.random.randint(0,n_t,dtype=np.intc)
			self.z_list.append(z)

			self._n_zw[z][w] += 1 # increment topic-word count 
			self._n_z[z] += 1  # increment topic-word sum
			self._n_dz[d][z] += 1   # increment doc-topic count
			self._n_d[d] += 1  # increment doc-topic sum
		
		#assert self._n_zw.sum() == self._n

	def _gibbs_sampling(self,predict= False):
		"""
		Only iterate once
		"""
		beta = self.beta
		alpha = self.alpha
		n_t = self.num_topic # the number of topics
		V = self.V

		for i in range(self._n):
			d= self.d_list[i] 
			w= self.w_list[i] 
			z= self.z_list[i]
                  
			self._n_zw[z][w] -= 1
			self._n_z[z] -= 1
			self._n_dz[d][z] -= 1 
			self._n_d[d] -= 1
			# K dimension
			# multi_prop = P(z|w) = P(w,z)/p(w)
			if not predict:
				multi_prop = (self._n_zw[:,w]+beta).astype(float)/(self._n_z + V*beta) * \
					(self._n_dz[d]+ alpha).astype(float)/(self._n_d[d] + n_t*alpha)
			else:
				train_n_z = self.results["topic-vocabulary-sum"]
				train_n_zw = self.results["topic-vocabulary"]
				multi_prop = (self._n_zw[:,w]+beta+train_n_zw[:,w]).astype(float)/(self._n_z + V*beta+train_n_z) * \
					(self._n_dz[d]+ alpha).astype(float)/(self._n_d[d] + n_t*alpha)
			assert multi_prop.shape[0] == n_t
			new_z =  np.random.multinomial(1, multi_prop/sum(multi_prop)).argmax()
			self.z_list[i] = new_z
			self._n_zw[new_z][w] += 1
			self._n_z[new_z] += 1
			self._n_dz[d][new_z] += 1 
			self._n_d[d] += 1


	def _log_likelihood(self,it):
		V = self.V
		beta = self.beta

		part1 = self.num_topic *(ss.gammaln(V*beta) - V * ss.gammaln(beta))
		part2 = (ss.gammaln(self._n_zw + beta).sum(axis=1) - ss.gammaln(self._n_z + V*beta)).sum(0)

		self.log_likelihood[it] = part1+part2


	def predict(self,corpus_test = None):
		if not corpus_test:
			raise ValueError("None corpus_test received!!")
		
		n = 0
		total_ln_pr = 0
		for M_dtm in corpus_test:
			n_m = M_dtm.sum()
			self._initialise(M_dtm)
			for i in range(self.num_iter):
				self._gibbs_sampling(predict= True)
			ln_pr = self._perplexity(M_dtm)
			n += n_m
			total_ln_pr += ln_pr
		self.results["perlexity"] = np.exp(-total_ln_pr/n)


	def _perplexity(self,m_dtm):
		#theta
		temp = self.alpha + self._n_dz
		theta = temp/temp.sum()

		temp = np.dot(self.results["topic-vocabulary-phi"].T, theta)
		ln_pr =(m_dtm * np.log(temp)).sum()
		return ln_pr


	def _output_prep(self):
		"""
		prepare for two arrays, 1. Topic-Vocabulary matrix to understand the topic
								2. Document-Topic to see the most relevent topic 
		"""
		beta = self.beta
		alpha = self.alpha
		n_t = self.num_topic
		V = self.V
		self.results = {}
		self.results["topic-vocabulary-phi"] = (self._n_zw + beta).astype(float)/(self._n_z + V*beta)[:,np.newaxis]
		self.results["document-topic-theta"] = (self._n_dz+ alpha).astype(float)/(self._n_d +  n_t*alpha)[:,np.newaxis]
		self.results["topic-vocabulary"] = self._n_zw
		self.results["topic-vocabulary-sum"] = self._n_z
