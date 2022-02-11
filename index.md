## Locally Weighted Regression and Random Forests

### Locally Weighted Regression
Locally weighted regression accounts for non-linear datasets involving particular variables, and so certain models like local kernel regression and local support vector regression can be used to more accurately predict data values. 
### Random Forests
Random forests . Shown below is Python code that sets up random forests to be used in conjunction with datasets.

```Python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000,max_depth=3)
```
### Boston Housing Prices Dataset

<img src=""Assets/Housing Price Dataset Graph.png" width="400" height="600" alt=hi class="inline"/>

### Locally Weighted Regression on Boston Housing Prices dataset
Using the median price, or "cmedv, as the output variable for this dataset, 

```Python

```
### Random Forests on Boston Housing Prices dataset
