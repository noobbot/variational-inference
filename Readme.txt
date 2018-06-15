Our project explores Variational Inference, a method for computing approximate
posteriors using optimization. 
In this paper, we motivate the need for approximate
distributions, then we describe how to compute them with Variational
 Inference.
Although the technique of Variational Inference has existed for several decades,
we describe how approximate 
inference is implemented in modern software packages.
We introduce the algorithm of Black-box Variational Inference
 and apply it
to a Bayesian Mixture of Gaussians. Using this as a running example, we perform
 several studies that reveal 
interesting characteristics of Variational Inference algorithms.



Dependencies

- Python 3.

- Edward http://edwardlib.org/


To Run

$ python3 gmm.py

or

$ python3 gmm_data.py