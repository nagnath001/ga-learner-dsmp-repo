# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here
data=pd.read_csv(path)
X= data.drop(['customer.id','paid.back.loan'], axis=1)
y=data['paid.back.loan']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.30, random_state = 0)
# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
fully_paid=y_train.value_counts()
print(fully_paid)
fully_paid.plot(kind='bar')

# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


#Code starts here

#Removing the last character from the values in column
X_train['int.rate'] = X_train['int.rate'].map(lambda x: str(x)[:-1])

#Dividing the column values by 100
X_train['int.rate']=X_train['int.rate'].astype(float)/100

#Removing the last character from the values in column
X_test['int.rate'] = X_test['int.rate'].map(lambda x: str(x)[:-1])

#Dividing the column values by 100
X_test['int.rate']=X_test['int.rate'].astype(float)/100

#Storing all the numerical type columns in 'num_df'
num_df=X_train.select_dtypes(include=['number']).copy()

#Storing all the categorical type columns in 'cat_df'
cat_df=X_train.select_dtypes(include=['object']).copy()

#Code ends here



# --------------
#Importing header files
import seaborn as sns

#Code starts here

#Setting the figure size
plt.figure(figsize=(20,20))


#Storing the columns of 'num_df'
cols=list(num_df.columns)


#Creating subplots
fig,axes=plt.subplots(9,1, figsize=(10,20))

#Looping across rows
for i in range(9):          
    
    #Plotting boxplot
    sns.boxplot(x=y_train,y=num_df[cols[i]],ax=axes[i])
    
    #Avoiding subplots overlapping
    fig.tight_layout()    

    
#Code ends here


# --------------
#Code starts here

#Storing the columns of 'cat_df'
cols=list(cat_df.columns)

#Setting up subplots
fig,axes=plt.subplots(2,2, figsize=(20,20))

#Looping through rows
for i in range(0,2):
    
    #Looping through columns
    for j in range(0,2):
        
        #Plotting count plot
        sns.countplot(x=X_train[cols[i*2+j]], hue=y_train,ax=axes[i,j])                        
        
        #Avoiding subplots overlapping
        fig.tight_layout()    

#Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Code starts here
for c in cat_df:
    X_train[c].fillna("NA")
    X_test[c].fillna('NA')
    
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for x in cat_df:
    X_train[x] = le.fit_transform(X_train[x])
    X_test[x] = le.transform(X_test[x])

y_train.replace(to_replace=['no', 'yes'], value=[0, 1])
y_test.replace(to_replace=['no', 'yes'], value=[0, 1])   
model = DecisionTreeClassifier( random_state=0)
model.fit(X_train, y_train)
model_predictions = model.predict(X_test)
acc=accuracy_score(y_test,model_predictions)
print("Decision tree accuracy: ",acc)

# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
model_2=DecisionTreeClassifier( random_state=0)
p_tree = GridSearchCV(estimator=model_2, param_grid=parameter_grid, cv=5)                  

p_tree.fit(X_train, y_train)
p_tree_predictions = p_tree.predict(X_test)
acc_2=accuracy_score(y_test, p_tree_predictions)
print("Decision tree Accuracy after tuning: ",acc_2)


# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
dot_data = tree.export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None, 
                                feature_names=X.columns, filled = True,  
                                class_names=['loan_paid_back_yes','loan_paid_back_no'])

# Draw graph
graph_big = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
#Image(graph.create_png())


# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


