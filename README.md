# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by:DHANUMALYA D 

Register Number:212222230030  

```
```
import pandas as pd
df=pd.read_csv('/content/Placement_Data (1).csv')
df.head()

df1=df.copy()
df1=df1.drop(['sl_no','salary'],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear') #library for linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
## Output:

![m1](https://github.com/Dhanudhanaraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119218812/1f27ecf4-7a8f-4728-9c4a-accd68757481)

![m2](https://github.com/Dhanudhanaraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119218812/fd429ba5-9693-4345-b062-0a8b64c49002)

![m3](https://github.com/Dhanudhanaraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119218812/1fd76e11-be2f-4594-b636-8bf73d6d4485)

![m4](https://github.com/Dhanudhanaraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119218812/f64784c5-c9ad-4b9b-9d3f-f76bd832eafb)

![m5](https://github.com/Dhanudhanaraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119218812/9173ecdf-8efc-49f4-8fee-a8a7f8183dc6)

![m6](https://github.com/Dhanudhanaraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119218812/c2d35385-a918-481b-bd84-6262fcc2ddbf)

![m7](https://github.com/Dhanudhanaraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119218812/5561b69c-521d-4b76-b920-034a886a2f1f)

## Result:

Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
