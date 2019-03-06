############## Classification#############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics


###reading data
missing_value=["?", " "]
data= pd.read_csv("Machine1.csv",na_values=missing_value, delimiter=",")

print(data.isnull().sum())

##removing missing values
data.dropna(inplace=True)
print(data.shape)

#replacing
# bmedian = data['Bare Nuclei'].median()
# data['Bare Nuclei'].fillna(bmedian,inplace=True)

X=data.values[:,1:]
y=data.values[:,0]
###### Devide data to test and train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)

#preprocessing
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)
X_transformed = scaler.transform(X)

####LogisticRegression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs',multi_class="auto",max_iter=600)
LogregModel = logreg.fit(X_train, y_train)
predicts=LogregModel.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, predicts))
print(metrics.classification_report(y_test,predicts))


####naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnbModel =gnb.fit(X_train, y_train)

predicts=gnbModel.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, predicts))
print(metrics.classification_report(y_test,predicts))




###K_fold
#######LogisticRegression
from sklearn.model_selection import KFold
kf = KFold(n_splits=10,shuffle=True)
kfsplit = kf.get_n_splits(X)
kfoldlogreg =LogisticRegression()
KFoldACC = []
# KFoldPREC = []
# KFoldREC = []
for train_index, test_index in kf.split(X_transformed):
    X_train, X_test = X_transformed[train_index], X_transformed[test_index]
    y_train, y_test = y[train_index], y[test_index]
    kfoldlogreg.fit(X_train, y_train)
    logpredict=kfoldlogreg.predict(X_test)
    KFoldACC.append(metrics.accuracy_score(y_test,logpredict))
    # KFoldPREC.append(metrics.precision_score(y_test,logpredict))
    # KFoldREC.append(metrics.recall_score(y_test,logpredict))

print("Accuracy for logistic regression-10foldCV",np.mean(KFoldACC))
# print("Precsion for logistic-10foldCV",np.mean(KFoldPREC))
# print("Recall for logistic-10foldCV",np.mean(KFoldREC))


###Gaussian Naive Bayes
kfoldGNB = GaussianNB()
acclist=[]

for train_index, test_index in kf.split(X_transformed):
    X_train, X_test = X_transformed[train_index], X_transformed[test_index]
    y_train, y_test = y[train_index], y[test_index]
    kfoldregrmodel = kfoldGNB.fit(X_train, y_train)
    predictions = kfoldGNB.predict(X_test)
    acclist.append(metrics.accuracy_score(y_test, predictions))

print('Gaussian Naive Bayes 10fold accuracy',np.mean(acclist))

###SVM
from sklearn import svm
kfoldGNB = svm.SVC()
acclist=[]

for train_index, test_index in kf.split(X_transformed):
    X_train, X_test = X_transformed[train_index], X_transformed[test_index]
    y_train, y_test = y[train_index], y[test_index]
    kfoldregrmodel = kfoldGNB.fit(X_train, y_train)
    predictions = kfoldGNB.predict(X_test)
    acclist.append(metrics.accuracy_score(y_test, predictions))

print('Support vector machine 10fold accuracy',np.mean(acclist))


####Leave One Out Cross Validation

##Logistic Regression
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)





#C: cost or penalty parameter

#kernel: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

#degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

#gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

#shrinking: Whether to use the shrinking heuristic.

#probability: Whether to enable probability estimates.
#  This must be enabled prior to calling fit, and will slow down that method.
#decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’


svmachine = svm.SVC(gamma='auto',kernel='rbf',degree=3)
svm_model = svmachine.fit(X_train_transformed, y_train)

print(svm_model.support_)
print(svm_model.support_vectors_)
print(svm_model.n_support_)
# print(svm_model.coef_)

# print(svm_model.predict_proba(X_test))
# print(svm_model.predict_log_proba(X_test))
print(svm_model.score(X_test_transformed,y_test))
y_pred = svm_model.predict(X_test_transformed)

########Confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test,y_pred )
print(cnf_matrix)
print(metrics.classification_report(y_test,y_pred))

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(X_transformed)
loologreg = LogisticRegression()
predicts=[]
for train_index, test_index in loo.split(X):
   X_train, X_test = X_transformed[train_index], X_transformed[test_index]
   y_train, y_test = y[train_index], y[test_index]
   loologregModel =loologreg.fit(X_train, y_train)
   predicts.append(loologreg.predict(X_test))

predict = np.array(predicts)
cnf_matrix = metrics.confusion_matrix(y, predict)

print(cnf_matrix)

acc = metrics.accuracy_score(y, predict)
print("Logistic Regression Accuracy by LOOCV", acc)

# recall = metrics.recall_score(y, predict)
# print("Logistic Regression Recall by LOOCV", recall)
#
# precession = metrics.precision_score(y, predict)
# print("Logistic Regression Precession by LOOCV", precession)

######ploting
# import required modules


#matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Logestic Regression Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.show()


#### Roc Curve
fpr, tpr, _ = metrics.roc_curve(y,  predict)
auc = metrics.roc_auc_score(y, predict)
auc = metrics.roc_auc_score(y_test, predict)
plt.plot(fpr,tpr,label="Logistic Regression, AUC = "+str(auc))
plt.legend(loc=4)
plt.show()

####GaussianNB
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(X)
loologreg = GaussianNB()
predicts=[]
for train_index, test_index in loo.split(X):
   X_train, X_test = X_transformed[train_index], X_transformed[test_index]
   y_train, y_test = y[train_index], y[test_index]
   loologregModel =loologreg.fit(X_train, y_train)
   predicts.append(loologreg.predict(X_test))

predict = np.array(predicts)
cnf_matrix = metrics.confusion_matrix(y, predict)

print(cnf_matrix)

acc = metrics.accuracy_score(y, predict)
print("Gaussian Naive Bayes Accuracy by LOOCV", acc)
######ploting
# import required modules


#matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Gaussian Naive Bayes Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.show()

#### Roc Curve
fpr, tpr, _ = metrics.roc_curve(y,  predict)
auc = metrics.roc_auc_score(y, predict)
auc = metrics.roc_auc_score(y_test, predict)
plt.plot(fpr,tpr,label="Gaussian Naive Bayes, AUC = "+str(auc))
plt.legend(loc=4)
plt.show()


from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(X)
loologreg = svmachine = svm.SVC(gamma='auto',kernel='rbf',degree=3)

predicts=[]
for train_index, test_index in loo.split(X_transformed):
   X_train, X_test = X_transformed[train_index], X_transformed[test_index]
   y_train, y_test = y[train_index], y[test_index]
   loologregModel =loologreg.fit(X_train, y_train)
   predicts.append(loologregModel.predict(X_test))

predict = np.array(predicts)
cnf_matrix = metrics.confusion_matrix(y, predict)

print(cnf_matrix)

acc = metrics.accuracy_score(y, predict)
print("Logistic Regression Accuracy by LOOCV", acc)

# recall = metrics.recall_score(y, predict)
# print("Logistic Regression Recall by LOOCV", recall)
#
# precession = metrics.precision_score(y, predict)
# print("Logistic Regression Precession by LOOCV", precession)

######ploting
# import required modules


#matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Support Vector Machine Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.show()

#### Roc Curve

fpr, tpr, _ = metrics.roc_curve(y,  predict)
auc = metrics.roc_auc_score(y, predict)
plt.plot(fpr,tpr,label="SVM, AUC = "+str(auc))
plt.legend(loc=4)
plt.show()



1+1