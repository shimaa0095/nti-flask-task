import pandas as pd 
import numpy as np 
from model import *
from flask import Flask, render_template, request,url_for, redirect
app=Flask(__name__)
m_name=None
data_tuple=preprocess_data()

@app.route("/")
def home():
    return render_template("main_page.html")

@app.route("/train_model",methods=['POST', 'GET'])
def apply_train():
    global m_name
    if request.method=='POST':
        res=request.form 
        if res["model"]=="Logistic Regression":
            m_name="log"
            #data_tuple=preprocess_data()
            score=log_reg(data_tuple[0],data_tuple[1],data_tuple[2],data_tuple[3])
            score=score*100
            res={res["model"]:score}
            return render_template("train_result.html",result=res)

        elif res["model"]=="KNN":
            m_name="knn"
            #data_tuple=preprocess_data()
            score=knn(data_tuple[0],data_tuple[1],data_tuple[2],data_tuple[3])
            score=score*100
            res={res["model"]:score}
            return render_template("train_result.html",result=res)

        elif res["model"]=="Random Forest":
            m_name="randF"
            #data_tuple=preprocess_data()
            score=rand_forest(data_tuple[0],data_tuple[1],data_tuple[2],data_tuple[3])
            score=score*100
            res={res["model"]:score}
            return render_template("train_result.html",result=res)
        
    return ("No Model be choosed !! try again.")

@app.route("/evaluation matrix")
def eval_mat():
    if m_name !=None:
        matrix=get_conf_mat(data_tuple[dict_names["xtest"]] ,data_tuple[dict_names["ytest"]],m_name)
        matrix=pd.DataFrame(matrix)  
        return render_template("eval_matrix.html" , data=matrix.to_html())
    return ("No Model be choosed to train !! try again.")


@app.route("/predict churn",methods=['POST', 'GET'])
def churn():
    if request.method=='POST':
        if m_name !=None:
            res=request.form
            l=list(res.values())
            if len(l)<8:
                return("Enter the All data Cells !!")
            p_c=predict_churn(res,m_name)
            if(p_c==0):
                p_c="No"
            elif(p_c==1):
                p_c="Yes" 
            return render_template("churn_res.html",result=p_c)
        else:
            return("No Model choosed to train !! try again")    
    return("Enter the data!! try again")

if __name__=="__main__":
    app.run(debug=True)
    
    