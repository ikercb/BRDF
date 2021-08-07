# BRDF
Implementation from scratch of the Bernoulli Random Density Forest. 

The Cython language is used to speed up Python.

This project is part of my master's thesis: "Density-based clustering with Random Forest".

### Parameters of the model
The model has the following parameters:
* nTrees: number of trees in the forest. Default: 10
* minSize: minimum number of samples required to split an internal node. Default = 5.
* p: probability of random split, otherwise the integrated squared error is minimized. Default = 0.1.

### Use

To build the Cython extension run the following line of code:
```python
python setup.py build_ext --inplace
```
Then, you can just import the density estimator:
```
from BRDF import densityRF
```
Example:
```
myForest = DensityRF(nTrees = 25, minSize = 10, p = 0.5)
myForest.fit(train)
preds = myForest.estimate(test)
```
