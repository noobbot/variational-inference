#
# Gaussian Mixture Model for VI experiments
#
#  Abhishek Shah 
#
#%matplotlib inline

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import six
import tensorflow as tf
import random

from edward.models import Categorical, InverseGamma, Mixture, MultivariateNormalDiag, Normal

plt.style.use('ggplot')

def build_dataset(N,K):
	sum=0.0
	maxx=0.0
	minx=0.0
	maxy=0.0
	miny=0.0

	pi=np.zeros(K,dtype=np.float32)
	for i in range(0,K):
		pi[i]=random.uniform(30,50)
		sum+=pi[i]
	for i in range(0,K):
		pi[i]/=sum

	mus=[]
	stds=[]
	for i in range(0,K):
		avaliable=False
		while not avaliable:
			avaliable=True
			_x=random.uniform(-17,17)
			_y=random.uniform(-17,17)
			for j in range(0,i):
				if (mus[j][0]-_x)**2+(mus[j][1]-_y)**2<=64:
					avaliable=False
					break
			if avaliable:
				mus.append([_x,_y])
				break
		stds.append([2,2])

	x=np.zeros((N,2),dtype=np.float32)
	for n in range(0,N):
		k=np.argmax(np.random.multinomial(1,pi))
		x[n,:]=np.random.multivariate_normal(mus[k],np.diag(stds[k]))

	return x

N = 1000	#Number of training set
K = 5		#Number of Distributions
D = 2
ed.set_seed(42)

x_train = build_dataset(N,K)

plt.scatter(x_train[:, 0], x_train[:, 1],c='green',s=10,alpha=0.5)
plt.axis([-20, 20, -20, 20])
plt.title("Simulated dataset")
plt.show()
