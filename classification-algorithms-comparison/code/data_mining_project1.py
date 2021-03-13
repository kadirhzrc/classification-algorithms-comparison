#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# defining necessary libraries
import numpy as np
import pandas as pd
import timeit
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


# In[ ]:


# Read corresponding data from dataset and
# keep it as a data frame by pandas library
filename = "Frogs_MFCCs.csv"
df = pd.read_csv(filename)
df.head(10)


# # Data Preprocessing 

# In[ ]:


# delete unnecessary columns
# these columns considered to be not having rich content which
# may be used for later classification purposes (?: tekrar döneceğim buraya)
del df["RecordID"]
del df['Family']
del df['Genus']

# seperate labels and feature columns
X = df.iloc[:, 0:22] # #Takes all rows of all columns except the range 0:22
y = df.select_dtypes(include=[object]) # a subset of the DataFrame’s columns only consisting of objects, which is "species" column

# transforming categorical to numbers
le = preprocessing.LabelEncoder() # Encode target labels with value between 0 and n_classes-1.
y = y.apply(le.fit_transform) # Fit label encoder and return encoded labels

# splitting dataset to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


# ## Gain ratio

# In[ ]:


# ID3 (Iterative Dichotomiser) decision tree algorithm uses information gain.
# C4.5, an improvement of ID3, uses the Gain ratio.
# Training phase
start = timeit.default_timer()
clf_entropy = DecisionTreeClassifier( criterion = "entropy") 
clf_entropy.fit(X_train, y_train)
# Testing phase: predicting X_test values over our model
y_pred_entropy = clf_entropy.predict(X_test)
stop = timeit.default_timer()
print('Runtime of default Gain Ratio: ', stop - start) 

# Evaluating cross validation
# Repeats Stratified K-Fold n times with different randomization in each repetition.
start = timeit.default_timer()
cv_gr1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate a score by cross-validation.
# In this case return type: Array of accuracy scores
scores_gr1 = cross_val_score(clf_entropy, X, y, scoring='accuracy', cv=cv_gr1)
stop = timeit.default_timer()
# Comparing results
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_entropy)) 
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy)*100)
print("Report: \n", classification_report(y_test, y_pred_entropy))
print('Accuracy (cross validation): %.3f ' % (np.mean(scores_gr1)*100))
print('Runtime of cross-validation applied Gain Ratio: ', stop - start)

# Bagging ensemble method implementation
start = timeit.default_timer()
bagging_model_gr = BaggingClassifier(base_estimator=clf_entropy)
bagging_model_gr.fit(X_train, y_train.values.ravel()) 
cv_gr = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores_gr = cross_val_score(bagging_model_gr, X, y.values.ravel(), scoring='accuracy', cv=cv_gr)
bgg_pred_gr = bagging_model_gr.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bgg_pred_gr)) 
print("Accuracy : ", accuracy_score(y_test, bgg_pred_gr)*100)
print("Report: \n", classification_report(y_test, bgg_pred_gr))
print('Accuracy (cross validation): %.3f ' % (np.mean(scores_gr)*100))
print('Runtime of cross-validation and bagging ensemble applied Gain Ratio: ', stop - start)

# Boosting ensemble method implementation
start = timeit.default_timer()
boosting_model_gr = GradientBoostingClassifier(init=clf_entropy)
boosting_model_gr.fit(X_train, y_train.values.ravel()) 
cv_bst_gr = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores_bst_gr = cross_val_score(boosting_model_gr, X, y.values.ravel(), scoring='accuracy', cv=cv_bst_gr)
bst_pred_gr = boosting_model_gr.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bst_pred_gr)) 
print("Accuracy : ", accuracy_score(y_test, bst_pred_gr)*100)
print("Report: \n", classification_report(y_test, bst_pred_gr))
print('Accuracy (cross validation): %.3f ' % (np.mean(scores_bst_gr)*100))
print('Runtime of cross-validation and boosting ensemble applied Gain Ratio: ', stop - start)


# ## Gini index

# In[ ]:


start = timeit.default_timer()
clf_gini = DecisionTreeClassifier(criterion = "gini")
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
stop = timeit.default_timer()
print('Runtime of default Gini Index: ', stop - start)
start = timeit.default_timer()
cv_gini1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores_gini1 = cross_val_score(clf_gini, X, y, scoring='accuracy', cv=cv_gini1)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_gini)) 
print("Accuracy : ", accuracy_score(y_test, y_pred_gini)*100)
print("Report: \n", classification_report(y_test, y_pred_gini))
print('Accuracy (cross validation): %.3f ' % (np.mean(scores_gini1)*100))
print('Runtime of cross-validation applied Gini Index: ', stop - start)

# Bagging ensemble method implementation
start = timeit.default_timer()
bagging_model_gini = BaggingClassifier(base_estimator=clf_gini)
bagging_model_gini.fit(X_train, y_train.values.ravel()) 
cv_gini = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores_gini = cross_val_score(bagging_model_gini, X, y.values.ravel(), scoring='accuracy', cv=cv_gini)
bgg_pred_gini = bagging_model_gini.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bgg_pred_gini)) 
print("Accuracy : ", accuracy_score(y_test, bgg_pred_gini)*100)
print("Report: \n", classification_report(y_test, bgg_pred_gini))
print('Accuracy (cross validation): %.3f ' % (np.mean(scores_gini)*100))
print('Runtime of cross-validation and bagging ensemble applied Gini Index: ', stop - start)

# Boosting ensemble method implementation
start = timeit.default_timer()
boosting_model_gini = GradientBoostingClassifier(init=clf_gini)
boosting_model_gini.fit(X_train, y_train.values.ravel()) 
cv_bst_gini = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores_bst_gini = cross_val_score(boosting_model_gini, X, y.values.ravel(), scoring='accuracy', cv=cv_bst_gini)
bst_pred_gini = boosting_model_gini.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bst_pred_gini)) 
print("Accuracy : ", accuracy_score(y_test, bst_pred_gini)*100)
print("Report: \n", classification_report(y_test, bst_pred_gini))
print('Accuracy (cross validation): %.3f ' % (np.mean(scores_bst_gini)*100))
print('Runtime of cross-validation and boosting ensemble applied Gini Index: ', stop - start)


# # Naive Bayes 

# In[ ]:


start = timeit.default_timer()
gnb = GaussianNB()
gnb.fit(X_train, y_train.values.ravel())
y_pred_gnb = gnb.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_gnb)) 
print("Accuracy : ", accuracy_score(y_test, y_pred_gnb)*100)
print("Report: \n", classification_report(y_test, y_pred_gnb))
print('Runtime of default Naive Bayes: ', stop - start)

# Bagging ensemble method implementation
start = timeit.default_timer()
bagging_model_nb = BaggingClassifier(base_estimator=gnb)
bagging_model_nb.fit(X_train, y_train.values.ravel()) 
cv_nb = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores_nb = cross_val_score(bagging_model_nb, X, y.values.ravel(), scoring='accuracy', cv=cv_nb)
bgg_pred_nb = bagging_model_nb.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bgg_pred_nb)) 
print("Accuracy : ", accuracy_score(y_test, bgg_pred_nb)*100)
print("Report: \n", classification_report(y_test, bgg_pred_nb))
print('Runtime of bagging ensemble applied Naive Bayes: ', stop - start)

# Boosting ensemble method implementation
start = timeit.default_timer()
boosting_model_nb = GradientBoostingClassifier(init=gnb)
boosting_model_nb.fit(X_train, y_train.values.ravel()) 
cv_bst_nb = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores_bst_nb = cross_val_score(boosting_model_nb, X, y.values.ravel(), scoring='accuracy', cv=cv_bst_nb)
bst_pred_nb = boosting_model_nb.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bst_pred_nb)) 
print("Accuracy : ", accuracy_score(y_test, bst_pred_nb)*100)
print("Report: \n", classification_report(y_test, bst_pred_nb))
print('Runtime of boosting ensemble applied Naive Bayes: ', stop - start)


# # Neural networks (1 hidden layer) 

# In[ ]:


start = timeit.default_timer()
mlp = MLPClassifier(hidden_layer_sizes=(150,), activation='relu', solver='adam', max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())
y_pred_nn = mlp.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_nn)) 
print("Accuracy : ", accuracy_score(y_test, y_pred_nn)*100)
print("Report: \n", classification_report(y_test, y_pred_nn))
print('Runtime of default Neural Network: ', stop - start)

# Bagging ensemble method implementation
start = timeit.default_timer()
bagging_model_nn = BaggingClassifier(base_estimator=mlp)
bagging_model_nn.fit(X_train, y_train.values.ravel()) 
cv_nn = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores_nn = cross_val_score(bagging_model_nn, X, y.values.ravel(), scoring='accuracy', cv=cv_nn)
bgg_pred_nn = bagging_model_nn.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bgg_pred_nn)) 
print("Accuracy : ", accuracy_score(y_test, bgg_pred_nn)*100)
print("Report: \n", classification_report(y_test, bgg_pred_nn))
print('Runtime of bagging ensemble applied Neural Network: ', stop - start)

# Boosting ensemble method implementation
start = timeit.default_timer()
boosting_model_nn = GradientBoostingClassifier(init=mlp)
boosting_model_nn.fit(X_train, y_train.values.ravel()) 
cv_bst_nn = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores_bst_nn = cross_val_score(boosting_model_nn, X, y.values.ravel(), scoring='accuracy', cv=cv_bst_nn)
bst_pred_nn = boosting_model_nn.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bst_pred_nn)) 
print("Accuracy : ", accuracy_score(y_test, bst_pred_nn)*100)
print("Report: \n", classification_report(y_test, bst_pred_nn))
print('Runtime of boosting ensemble applied Neural Network: ', stop - start)


# # Neural networks (2 hidden layers) 

# In[ ]:


start = timeit.default_timer()
mlp2 = MLPClassifier(hidden_layer_sizes=(75, 75), activation='relu', solver='adam', max_iter=1000)
mlp2.fit(X_train, y_train.values.ravel())
y_pred_nn2 = mlp2.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_nn2)) 
print("Accuracy : ", accuracy_score(y_test, y_pred_nn2)*100)
print("Report: \n", classification_report(y_test, y_pred_nn2))
print('Runtime of default Neural Network (2 hidden layers): ', stop - start)

# Bagging ensemble method implementation
start = timeit.default_timer()
bagging_model_nn2 = BaggingClassifier(base_estimator=mlp2)
bagging_model_nn2.fit(X_train, y_train.values.ravel()) 
bgg_pred_nn2 = bagging_model_nn2.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bgg_pred_nn2)) 
print("Accuracy : ", accuracy_score(y_test, bgg_pred_nn2)*100)
print("Report: \n", classification_report(y_test, bgg_pred_nn2))
print('Runtime of bagging ensemble applied Neural Network (2 hidden layers): ', stop - start)

# Boosting ensemble method implementation
start = timeit.default_timer()
boosting_model_nn2 = GradientBoostingClassifier(init=mlp2)
boosting_model_nn2.fit(X_train, y_train.values.ravel()) 
bst_pred_nn2 = boosting_model_nn2.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bst_pred_nn2)) 
print("Accuracy : ", accuracy_score(y_test, bst_pred_nn2)*100)
print("Report: \n", classification_report(y_test, bst_pred_nn2))
print('Runtime of boosting ensemble applied Neural Network (2 hidden layers): ', stop - start)


# # Support Vector Machine 

# In[ ]:


# Create the parameter grid based on the results of random search 
start = timeit.default_timer()
params_grid = [{'kernel': ['rbf', 'poly', 'sigmoid', 'linear'], 
                'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4],
                 'C': [1, 10, 100, 1000]}]
svm_model = GridSearchCV(SVC(), params_grid)
svm_model.fit(X_train, y_train.values.ravel())

final_model = svm_model.best_estimator_
y_pred_final = final_model.predict(X_test)
stop = timeit.default_timer()

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_final)) 
print("Report: \n", classification_report(y_test, y_pred_final))
print('Best C:',svm_model.best_estimator_.C) 
print('Best Kernel:',svm_model.best_estimator_.kernel)
print('Best Gamma:',svm_model.best_estimator_.gamma)
print('Best accuracy score :', svm_model.best_score_) 
print("Score (training set): %f" % final_model.score(X_train, y_train))
print("Score (testing set): %f" % final_model.score(X_test, y_test))
print('Runtime of default SVM: ', stop - start)

# Bagging ensemble method implementation
start = timeit.default_timer()
bagging_model_svm = BaggingClassifier(base_estimator=final_model)
bagging_model_svm.fit(X_train, y_train.values.ravel()) 
bgg_pred_svm = bagging_model_svm.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bgg_pred_svm)) 
print("Accuracy : ", accuracy_score(y_test, bgg_pred_svm)*100)
print("Report: \n", classification_report(y_test, bgg_pred_svm))
print('Runtime of bagging ensemble applied SVM: ', stop - start)

# Boosting ensemble method implementation
start = timeit.default_timer()
boosting_model_svm = GradientBoostingClassifier(init=final_model)
boosting_model_svm.fit(X_train, y_train.values.ravel()) 
bst_pred_svm = boosting_model_svm.predict(X_test)
stop = timeit.default_timer()
print("Confusion Matrix: \n", confusion_matrix(y_test, bst_pred_svm)) 
print("Accuracy : ", accuracy_score(y_test, bst_pred_svm)*100)
print("Report: \n", classification_report(y_test, bst_pred_svm))

