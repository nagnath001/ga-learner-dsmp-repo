# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
df=pd.read_csv(path)
#Code starts here
df.head()
X = df.drop('Price',axis=1)
y = df['Price'].copy()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 6)
corr=X_train.corr()
print(corr)


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
r2=regressor.score (X_test,y_test)
print(r2)


# --------------
from sklearn.linear_model import Lasso

# Code starts here
# instantiate lasso model
lasso=Lasso()
# fit and predict
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)
r2_lasso=lasso.score(X_test,y_test)
print(r2_lasso)


# --------------
from sklearn.linear_model import Ridge

# Code starts here
# instantiate lasso model
ridge=Ridge()
# fit and predict
ridge.fit(X_train,y_train)
ridge_pred =ridge.predict(X_test )
# calculate R2
r2_ridge=ridge.score(X_test,y_test)
print(r2_ridge)


# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
regressor=LinearRegression()
score=cross_val_score(regressor,X_train,y_train,cv=10)
mean_score=score.mean()
print(mean_score)



# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here
model= make_pipeline(PolynomialFeatures(2), LinearRegression())     
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
r2_poly=model.score(X_test,y_test)
#print('r2_poly: {0:.2f}'.format(model.score(X_test, y_test)))
print(r2_poly)



