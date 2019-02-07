from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from preprocessing import *
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler


model_obj={}

def log_reg(xtrain,xtest,ytrain,ytest):
    obj=LogisticRegression()
    obj.fit(xtrain,ytrain)
    model_obj["log"]=obj
    return obj.score(xtest,ytest)

def knn(xtrain,xtest,ytrain,ytest):
    obj=KNeighborsClassifier(n_neighbors=5)
    obj.fit(xtrain,ytrain)
    model_obj["knn"]=obj
    return obj.score(xtest,ytest)

def rand_forest(xtrain,xtest,ytrain,ytest):
    obj = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    obj.fit(xtrain,ytrain)
    model_obj["randF"]=obj
    return obj.score(xtest,ytest)    

def predict_churn(dict_row,model_name):
    df=pd.DataFrame([list(dict_row.values())])
    # ll=[list(dict_row.values())]
    # print(ll)
    # df=pd.DataFrame(ll)
    # print(df)
    sc_x = StandardScaler()
    df = pd.DataFrame(sc_x.fit_transform(df))
    vals=df.values
    p=model_obj[model_name].predict(vals)
    return p[0]


def get_conf_mat(xtest,ytest,model_name):
    p=model_obj[model_name].predict(xtest)
    cm = confusion_matrix(ytest, p)
    return cm

def get_report_mat(xtest,ytest,model_name):
    p=model_obj[model_name].predict(xtest)
    cm = classification_report(ytest, p)
    return cm


# data_tuple=preprocess_data()
# score=log_reg(data_tuple[0],data_tuple[1],data_tuple[2],data_tuple[3])
# k={"a":0 , "b":0, "c":0, "d":0, "e":30, "f":0, "g":0, "h":800 }
# p=predict_churn(k,"log")
# print(p)

# print(score)
# matrix=get_report_mat(data_tuple[dict_names["xtest"]] ,data_tuple[dict_names["ytest"]],"log" )
# print(matrix)
# print(type(matrix))

# m=matrix.split("\n")
# col_index = ['0','1','micro avg','macro avg','weighted avg'] 
# vx = [] 
# b =0  
# for i in m:
#     b +=1
#     s = i.split(' ')
#     s2 = list(filter(lambda a: a != '', s))
#     if len(s2) > 1:
#         if b > 1 and b<5:
#             vx.append(s2[1:])
#         else:
#              vx.append(s2[2:])   
#     #print(s2)

# hed_index = vx[0]
# del vx[0]
# vx = np.ndarray(vx)
# print(vx)

#f=pd.DataFrame(data=vx, index=col_index, columns=hed_index)


# n = np.array(m).reshape(-1,1)
# f=pd.DataFrame(n)
#print(f)