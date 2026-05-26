import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#data reading and encoding
data=pd.read_csv(r"C:\flutter_projects\INTERNSHIP\class4\assignment\machine_failure.csv")

data = data[["Type","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","Machine failure","TWF","HDF","PWF","OSF","RNF"]]

label_encoder=LabelEncoder()
data["Type"]=label_encoder.fit_transform(data["Type"])
x=data[["Type","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","TWF","HDF","PWF","OSF","RNF"]]
y=data["Machine failure"]

#training
model=LogisticRegression()
model.fit(x,y)

print("model coefficient=",model.coef_)
print("model intercept=",model.intercept_)

#sample input
sample_type = label_encoder.transform(["M"])[0]
sample = [[sample_type,301,311,1450,52,28,1,0,0,0,0]]

#predicted value
prediction=model.predict(sample)
print("Predicted value=",prediction)
print("Value range=",model.predict_proba(sample)) 

#Predict whether the machine will fail:
print("Prediction whether the machine will fail:")
if prediction[0] == 0:
    print("No Failure")
else:
    print("Failure")
    
