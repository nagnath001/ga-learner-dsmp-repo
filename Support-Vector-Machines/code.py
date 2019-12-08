# --------------
import pandas as pd
from collections import Counter

# Load dataset
data=pd.read_csv(path)
print(data.isnull().sum())
print(data.describe)


# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

# Store the label values 
#print(data.columns)
label=data["Activity"]
# plot the countplot
ax =sns.countplot(label)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)





# --------------
# make the copy of dataset
data_copy=data.copy()
# Create an empty column 
data_copy["duration"]=data_copy.apply(lambda _: '', axis=1)

mask = label.isin(['WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'])
# Calculate the duration

duration_df=data_copy.groupby([label[mask],'subject'])['duration'].count()*1.28
duration_df=pd.DataFrame(duration_df)
# Sort the values of duration
plot_data = duration_df.reset_index().sort_values('duration', ascending=False)
plot_data['Activity'] = plot_data['Activity'].map({'WALKING_UPSTAIRS':'Upstairs', 'WALKING_DOWNSTAIRS':'Downstairs'})
plt.figure(figsize=(15,5))
sns.barplot(data=plot_data, x='subject', y='duration', hue='Activity')
plt.title('Participants Compared By Their Staircase Walking Duration')
plt.xlabel('Participants')
plt.ylabel('Total Duration [s]')
plt.show()



# --------------
#exclude the Activity column and the subject column
feature_cols = data.columns[: -2]   

#Calculate the correlation values
correlated_values = data[feature_cols].corr()
#stack the data and convert to a dataframe

correlated_values = (correlated_values.stack().to_frame().reset_index()
                    .rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0:'Correlation_score'}))


#create an abs_correlation column
correlated_values['abs_correlation'] = correlated_values.Correlation_score.abs()

#Picking most correlated features without having self correlated pairs
top_corr_fields = correlated_values.sort_values('Correlation_score', ascending = False).query('abs_correlation>0.8 ')
top_corr_fields = top_corr_fields[top_corr_fields['Feature_1'] != top_corr_fields['Feature_2']].reset_index(drop=True)



# --------------
# importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score
# Encoding the target variable
le = LabelEncoder()
for x in data["Activity"]:
    data['Activity'] = le.fit_transform(data.Activity)
X=data.drop(["Activity"],1) 
y=data["Activity"]   
# split the dataset into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=40)
# Baseline model 
classifier=SVC()
clf=classifier.fit(X_train,y_train)
y_pred=clf.predict(X_test)
precision,recall, f_score,_=precision_recall_fscore_support(y_test, y_pred, average='weighted')
model1_score=accuracy_score(y_test, y_pred)
print("precision=",precision)
print("recall=",recall)
print("f1_score=",f_score)
print("accuracy=",model1_score)






# --------------
# importing libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score

# Feature selection using Linear SVC
lsvc=LinearSVC(C=0.01, class_weight=None, dual=False, fit_intercept=True, penalty='l1', random_state=42).fit(X_train,y_train)
model_2=SelectFromModel(lsvc, prefit=True)
new_train_features= model_2.transform(X_train)
new_test_features= model_2.transform(X_test)
# model building on reduced set of features
classfier_2=SVC()
clf_2=classfier_2.fit(new_train_features,y_train)
y_pred_new=clf_2.predict(new_test_features)
model2_score=accuracy_score(y_test,y_pred_new)
precision,recall, f_score,_=precision_recall_fscore_support(y_test, y_pred_new,average='weighted')
print("precision=",precision)
print("recall=",recall)
print("f1_score=",f_score)
print("accuracy=",model2_score)







# --------------
# Importing Libraries
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score

# Set the hyperparmeters
# defining parameter range 
parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [100, 20, 1, 0.1]
}

selector = GridSearchCV(SVC(), parameters, scoring='accuracy') # we only care about accuracy here
selector.fit(new_train_features, y_train)

print('Best parameter set found:')
print(selector.best_params_)
print('Detailed grid scores:')
means = selector.cv_results_['mean_test_score']
stds = selector.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, selector.cv_results_['params']):
    print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
    print()
# fitting the model for grid search 
classifier_3=SVC(kernel='rbf', C=20)
clf_3=classifier_3.fit(new_train_features, y_train)
y_pred_final =clf_3.predict(new_test_features)
model3_score=accuracy_score(y_test,y_pred_final)
precision,recall, f_score,_=precision_recall_fscore_support(y_test, y_pred_final,average='weighted')
print("precision=",precision)
print("recall=",recall)
print("f1_score=",f_score)
print("accuracy=",model3_score)


# Usage of grid search to select the best hyperparmeters



    

# Model building after Hyperparameter tuning





