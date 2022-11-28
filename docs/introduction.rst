Introduction
^^^^^^^^^^^^^
**ICF-MI** combines two statistical algorithms, **TT-ICE** which is an incremental tensor-train decomposition alogorithm suitable for data streams and **GIM** which is a model identification alogorithm.
The Algorithm has the following salient features:

* **TT-ICE** is a novel incremental tensor-train (TT) decomposition algorithm which is developed to decompose high dimensional data streams into tensor-train format.
**TT-ICE** uses orthogonal projections and sequential application of SVD to compute the missing orthogonal directions in the TT-cores.
For a stream of d-dimensional tensors, **TT-ICE** accumulates the stream along an additional (d+1)-th dimension and trains d 3 dimensional TT-cores that span the first d dimensions of the stream.
The (d+1)-th TT-core resulting from the decomposition process contains coefficient vectors unique to each datum.
Within the scope of our pipeline, we will treat **TT-ICE** as a multilinear dimensionality reduction tool and use the coefficient vectors contained in the last TT-core to train the surrogate model.

* **GIM** is an information-theoretic model identification algorithm which can be used to identify the model which best fits the data.

* **ICF-MI** is a combination of **TT-ICE** and **GIM** and can therefore be used to identify the model parameter in a cost efficient manner.

.. HEADING:
.. ===============
.. * If necessart mention some points here.

REFERENCES:
===============
* `Google <https://www.google.com>`_ , search engine has been used throughout the project.
*  Aksoy et al., An Incremental Tensor-Train Decomposition Algorithm, 'arXiv preprint <https://arxiv.org/abs/2211.12487>'_
*  Other kind of text ``Bold reference``.
*  Bold **letters**.

Developers:
========
| Sahil Bhola (University of Michgan, Ann Arbor)
| Doruk Aksoy (University of Michgan, Ann Arbor)
| Carleen McKenna (University of Michgan, Ann Arbor)
| Codie Kawaguchi (University of Michgan, Ann Arbor)
