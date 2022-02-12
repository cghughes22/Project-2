## Locally Weighted Regression and Random Forests

### Locally Weighted Regression
Locally weighted regression accounts for non-linear datasets involving particular variables, and so certain models like local kernel regression and local support vector regression can be used to more accurately predict data values. 
### Random Forests
A random forest combines multiple decision trees in order to create more accurate predictions. The logic behind this is that individual decision trees perform better when used in conjunction with one another than they do alone. Shown below is Python code that sets up these random forests.

```Python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000,max_depth=3)
```
### Boston Housing Prices Dataset
Below is a graph showing the relationship between the median housing price in Boston and the number of rooms in a house in Boston.
<img src="Assets/Housing Price Dataset Graph.png" width="800" height="600" alt=hi class="inline"/>

### Locally Weighted Regression on Boston Housing Prices dataset
Using the median price, or **cmedv**, as the output variable for this dataset, I utilized a few different models and performed local regressions in order to obtain better predictions for data values pertaining to the median housing price in Boston. 
Some of the Python code setting up the regressions is shown here.
```Python
from sklearn.model_selection import KFold
X = np.array(df['rooms']).reshape(-1,1)
y = np.array(df['cmedv']).reshape(-1,1)
dat = np.concatenate([X,y.reshape(-1,1)], axis=1)
```

```Python
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=1234)

y_train = y_train.reshape(len(y_train),)
y_test = y_test.reshape(len(y_test),)
```

```Python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
```

```Python
from sklearn.metrics import mean_absolute_error 

print("Intercept: {:,.3f}".format(lm.intercept_))
print("Coefficient: {:,.3f}".format(lm.coef_[0]))
    
mae = mean_absolute_error(y_test, lm.predict(X_test))
mae_lowess = mean_absolute_error(dat_test[:,1], yhat)
print("MAE = ${:,.2f}".format(1000*mae))
print("MAE LOWESS = ${:,.2f}".format(1000*mae_lowess))
```
The mean absolute error values for the regular model and the LOWESS model come out to be $4,140.78 and $3,951.13, respectively. The resulting line is fit to the aforementioned graph, shown here.
```Python
fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim(3, 9)
ax.set_ylim(0, 51)
ax.scatter(x=df['rooms'], y=df['cmedv'],s=25)
ax.plot(X_test, lm.predict(X_test), color='red')
ax.plot(dat_test[:,0], yhat, color='orange',lw=3)
ax.set_xlabel('Number of Rooms',fontsize=16,color='Darkgreen')
ax.set_ylabel('House Price (Tens of Thousands of Dollars)',fontsize=16,color='Darkgreen')
ax.set_title('Boston Housing Prices',fontsize=16,color='purple')
ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
```
<img src="Assets/Fitted Line.png" width="800" height="600" alt=hi class="inline">

### Random Forests on Boston Housing Prices dataset
After applying the local regressions onto the Boston dataset and obtaining different values for the mean absolute error, I set up random forests in order to find the mean absolute error. After doing so, I obtained a value of $3,991.68. The Python code here shows how to set up random forests, and obtain the mean absolute error for the dataset.
```Python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000,max_depth=3)
rf.fit(X_train,y_train)
yhat_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, yhat_rf)
print("MAE RF = ${:,.2f}".format(1000*mae_rf))
```
