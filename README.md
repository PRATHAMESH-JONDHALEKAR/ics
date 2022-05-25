# ics













### DECISION TREE

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

dataset ={
    'Id':[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    'Age':['<21','<21','21-35','>35','>35','>35','21-35','<21','<21','>35','<21','21-35','21-35','>35'],
    'Income':['High','High','High','Medium','Low','Low','Low','Medium','Low','Medium','Medium','Medium','High','Medium'],
    'Gender':['Male','Male','Male','Male','Female','Female','Female','Male','Female','Female','Female','Male','Female','Male'],
    'MaritalStatus':['Single','Married','Single','Single','Single','Married','Married','Single','Married','Single','Married','Married','Single','Married'],
    'Buys':['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}
dataset=df=pd.DataFrame(dataset,columns=["ID","Age","Income","Gender","MaritalStatus","Buys"])
df = df.iloc[:,1:]
x= df.iloc[:,:-1]
y= df.iloc[:,-1]

le= LabelEncoder()
x=x.apply(le.fit_transform)

print("Age",list( zip(df.iloc[:,0], x.iloc[:,0])))
print("Income",list( zip(df.iloc[:,1], x.iloc[:,1])))
print("Gender",list( zip(df.iloc[:,2], x.iloc[:,2])))
print("MaritalStatus",list( zip(df.iloc[:,3], x.iloc[:,3])))


dt=DecisionTreeClassifier()
dt.fit(x,y)

query=np.array([1,1,0,0])
pred=dt.predict([query])
print("Model Predicted value",pred[0])
