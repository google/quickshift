Quickshift++
======
This is not an officially supported Google product

Density-based clustering algorithm based on mode-seeking.


Usage
======

**Initializiation**:

.. code-block:: python

  QuickshiftPP(k, beta) 
  
k: number of neighbors in k-NN

beta: fluctuation parameter which ranges between 0 and 1.

**Finding Clusters**:

.. code-block:: python

  fit(X)
  
X is the data matrix, where each row is a datapoint in euclidean space.

fit performs the clustering. The final result can be found in QuickshiftPP.memberships.

**Example** (mixture of two gaussians):

.. code-block:: python

  from QuickshiftPP import *
  import numpy as np
  
  X = [np.random.normal(0, 1, 2) for i in range(100)] + [np.random.normal(5, 1, 2) for i in range(100)]
  y = [0] * 100 + [1] * 100

  # Declare a Quickshift++ model with tuning hyperparameters.
  model = QuickshiftPP(k=20, beta=.5)

  # Compute the clustering.
  model.fit(X)
  y_hat = model.memberships

  from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
  print("Adj. Rand Index Score: %f." % adjusted_rand_score(y_hat, y))
  print("Adj. Mutual Info Score: %f." % adjusted_mutual_info_score(y_hat, y))


Install
=======

This package uses distutils, which is the default way of installing
python modules.

To install for all users on Unix/Linux::

  sudo python setup.py build; python setup.py install



Dependencies
=======

python 2.7, scikit-learn



