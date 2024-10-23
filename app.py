import sys,os
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse

import logging,warnings

logging.basicConfig(level=logging.WARN)
logger=logging.getLogger(__name__)

def eval_metrics(actual,pred):
    rmse=root_mean_squared_error(actual,pred)
    mse=mean_absolute_error(actual,pred)
    r2=r2_score(actual,pred)

    return rmse,mse,r2


if __name__=="__main__":
    
    csv_url=("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv")

    try:
        data=pd.read_csv(csv_url,sep=";")
    except Exception as e:
        logger.exception("unable to download the data")

    train,test=train_test_split(data)

    train_X=train.drop(["quality"],axis=1)
    test_X=test.drop(["quality"],axis=1)

    train_y=train[['quality']]
    test_y=test[['quality']] 

    alpha=float(sys.argv[1]) if len(sys.argv)>1 else 0.5
    l1_ratio=float(sys.argv[2]) if len(sys.argv)>2 else 0.5

    with mlflow.start_run():
        lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
        lr.fit(train_X,train_y)

        predicted_qualities=lr.predict(test_X)
        (rmse,mse,r2)=eval_metrics(test_y,predicted_qualities)

        print("ElasticNet model(apha={:f},l1_ratio={:f})".format(alpha,l1_ratio))
        print("RMSE:%s"%rmse)
        print("MSE : %s"%mse)
        print("r2 score : %s"%r2)

        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)
        mlflow.log_metric("RMSE",rmse)
        mlflow.log_metric("mse",mse)
        mlflow.log_metric("r2 score",r2)

        remote_server_ur="http://127.0.0.1:5000"
        mlflow.set_tracking_uri(remote_server_ur)

        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!="file":
            mlflow.sklearn.log_model(lr,"model",registered_model_name="ElasticNet")

        else:
            mlflow.sklearn.log_model(lr,"model")