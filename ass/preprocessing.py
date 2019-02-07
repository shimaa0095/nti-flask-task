import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

dict_churn={"No":0 ,"Yes":1}
dict_names={"xtrain":0,"xtest":1,"ytrain":2,"ytest":3}

def read_data(file_name="dataset_ass.csv"):
    df=pd.read_csv("dataset_ass.csv")
    new_df=df[["gender","SeniorCitizen" ,"Partner","Dependents","tenure"
    ,"PhoneService","MultipleLines","TotalCharges","Churn"]]
    return new_df

def missing_values(data):
    for col in data.columns:
        col_data=data[col]
        if col_data.isnull().sum()>0:
            if col_data.dtype=="object":
                obj_mode = col_data.mode()[0]
                data[col].fillna(obj_mode,inplace=True)
            else:
                obj_mean = col_data.mean()
                data[col].fillna(obj_mean,inplace=True)
    return data

def labelEncoding(data):
    d = {}
    for col in data.columns:
        if data[col].dtype == "object":
            d[col] = LabelEncoder()
            data[col]   =d[col].fit_transform(data[col])
    return data

def standarization(data):
    last_col=data.iloc[:,-1]
    temp_df=data.iloc[:,:-1]
    sc_x = StandardScaler()
    temp_df = pd.DataFrame(sc_x.fit_transform(temp_df),columns=temp_df.columns)
    new_df=temp_df.join(last_col)
    return new_df

def data_split(data):
    x=data.iloc[: , :-1]
    y=data.iloc[: , -1]
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=42)
    return (xtrain,xtest,ytrain,ytest)

def preprocess_data():
    data_frame=read_data()
    data_frame=missing_values(data_frame)
    data_frame=labelEncoding(data_frame)
    new_df=standarization(data_frame)
    (xtrain,xtest,ytrain,ytest)=data_split(new_df)
    return(xtrain,xtest,ytrain,ytest)

#l=preprocess_data()
#print(l[3])  
#print("hi all")

