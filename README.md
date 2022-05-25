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












import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn import tree

datasets = {
'Day':['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14'],
'OutLook':['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
'Temperature':['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
'Humidity':['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
'Wind':['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
'PlayTennis':['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']}

df=pd.DataFrame(datasets)
df

X=df.iloc[:,1:-1]
Y=df["PlayTennis"]

laen=LabelEncoder();
x=X.apply(laen.fit_transform)
x

print("Outllok:",list(zip(df.iloc[:,0], x.iloc[:,0])))
print("Temperature:",list(zip(df.iloc[:,1], x.iloc[:,1])))
print("Humidity:",list(zip(df.iloc[:,2], x.iloc[:,2])))
print("Wind:",list(zip(df.iloc[:,3], x.iloc[:,3])))

print(Y)

dtc=DecisionTreeClassifier()
dtc.fit(x,Y)

queryy=np.array([1,1,0,0])
predi=dtc.predict([queryy])
