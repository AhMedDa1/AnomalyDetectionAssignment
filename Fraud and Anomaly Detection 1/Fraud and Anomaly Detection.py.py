#!/usr/bin/env python
# coding: utf-8

# import lib and load data


import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

path = 'datasetAssignment1.csv'
data = pd.read_csv(path)


#get insight from Data 
data.describe()
data.head()


#################################### Featrue Selection in many Method ########################################


#split the data to feature and target to implemet bulit-n function to help in find important feature

y = data.classs #i add extra x cause class is stored word in python
list = ['att2','classs']#i will discus why i have droped att2 later
x = data.drop(list,axis = 1 )

model = ExtraTreesClassifier()
model.fit(x,y)

print(model.feature_importances_) 


#################################### Second Method for select feature########################################


from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
clf_rf_3 = RandomForestClassifier()


rfe = RFE(estimator=clf_rf_3, n_features_to_select=10, step=1)
rfe = rfe.fit(x, y)


print('Chosen best 10 feature:',x_train.columns[rfe.support_])



########################################### split the data to train and test ############################################



Features = ['att1', 'att5', 'att7', 'att8', 'att11', 'att13', 'att14', 'att17',
       'att22','att23']
x = data[Features]

x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=183703)


########################################### The Model and cal accuracy ############################################
### the first model
Model = LogisticRegression(random_state=183703)

Model.fit(x_train,y_train)

v = Model.predict(x_val)


print(classification_report(y_val,v))

# confusion_matrix
conf_mat = confusion_matrix(y_true=y_val, y_pred=v)
print(conf_mat)


### the second model


Model = ExtraTreesClassifier(random_state=183703)

Model.fit(x_train,y_train)

v = Model.predict(x_val)


print(classification_report(y_val,v))

# confusion_matrix

conf_mat = confusion_matrix(y_true=y_val, y_pred=v)
print(conf_mat)

### the third model


Model = RandomForestClassifier(random_state=183703)

Model.fit(x_train,y_train)

v = Model.predict(x_val)


print(classification_report(y_val,v))


# confusion_matrix


conf_mat = confusion_matrix(y_true=y_val, y_pred=v)
print(conf_mat)



########################################### Handel with unbalanced data ############################################

#### Count the one's and zero's and plot the Shape ####


#to print the two class in our target data
ax = sns.countplot(y,label="Count")       
a, b = y.value_counts()
print('Number of Zero: ',a)
print('Number of One : ',b) 


#### Split the data to zero and one class ####


n_class_0, n_class_1 = data.classs.value_counts()

class_0 = data[data['classs'] == 0]
class_1 = data[data['classs'] == 1]


#### under Sampling ####



class_0_u = class_0.sample(n_class_1,random_state=183703) # choose random values from class zero and remove it until its == the class one
final_under = pd.concat([class_0_u, class_1], axis=0) # concatinate the zero's and one's cause we split it in last step

print('Random under-sampling:')
print(final_under.classs.value_counts())

#plot the shape of the data after under-sampling
from matplotlib import pyplot as plt
final_under.classs.value_counts().plot(kind='bar', title='Count (target)');


#### split data and run model and  cal accuracy

#run the first model to bulanced data and calculate accuracy and confusion_matrix
y_u = final_under.classs
list = ['classs','att2']
x_u = final_under.drop(list,axis=1)
x_train_u,x_test_u,y_train_u,y_test_u = train_test_split(x_u,y_u,test_size=0.2,random_state=183703)


Model = LogisticRegression(random_state=183703)

Model.fit(x_train_u,y_train_u)

v = Model.predict(x_test_u)


print(classification_report(y_test_u,v))


conf_mat = confusion_matrix(y_true=y_test_u, y_pred=v)
print(conf_mat)


# the second model


Model = RandomForestClassifier(random_state=183703)

Model.fit(x_train_u,y_train_u)

v = Model.predict(x_test_u)


print(classification_report(y_test_u,v))




conf_mat = confusion_matrix(y_true=y_test_u, y_pred=v)
print(conf_mat)



#### over Sampling ####



class_1_o = class_1.sample(n_class_0, replace=True,random_state=183703)
final_over = pd.concat([class_0, class_1_o], axis=0)

print('Random over-sampling:')
print(final_over.classs.value_counts())



final_over.classs.value_counts().plot(kind='bar', title='Count (target)');



y_o = final_over.classs
list = ['classs','att2']
x_o = final_over.drop(list,axis=1)
x_train_o,x_test_o,y_train_o,y_test_o = train_test_split(x_o,y_o,test_size=0.2,random_state=183703)




Model = LogisticRegression(random_state=183703)

Model.fit(x_train_o,y_train_o)

v = Model.predict(x_test_o)


print(classification_report(y_test_o,v))



conf_mat = confusion_matrix(y_true=y_test_o, y_pred=v)
print(conf_mat)



Model = RandomForestClassifier(random_state=183703)

Model.fit(x_train_o,y_train_o)

v = Model.predict(x_test_o)


print(classification_report(y_test_o,v))




conf_mat = confusion_matrix(y_true=y_test_o, y_pred=v)
print(conf_mat)



#####



