import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import math
import pickle

#importing the file
diabetes_dataset = pd.read_csv('./diabetes.csv')

#converting the normal data into standard data to feed into algorithm
x=diabetes_dataset.drop(columns="Outcome",axis=1)
y=diabetes_dataset['Outcome']
scaler=StandardScaler()
scaler.fit(x)
standard_data=scaler.transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


#Simple Vector Machine Alogrithm
classifier=svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)


#testing overall accuracy 
x_train_prediction=classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)
print('accuracy score of training data: ',math.floor(training_data_accuracy*100),'%')


#converting the single input data into  data that we can fit in the model 
input_data = (1,85,66,29,0,26.6,0.351,31)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#testing the data
prediction = classifier.predict(input_data_reshaped)
print(prediction[0])

#making the seperate file for trained model
filename = 'm.sav'
pickle.dump(classifier,open(filename,'wb'));


# importing the trained model 
loaded_model = pickle.load(open("m.sav",'rb'))
prediction = loaded_model.predict(input_data_reshaped)
print(f"From loaded model {prediction[0]}")
