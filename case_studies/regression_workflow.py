"""Regression workflow with baseline comparison and cross-validation."""
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT=Path("outputs/regression"); OUT.mkdir(parents=True,exist_ok=True)
X,y=make_regression(n_samples=6000,n_features=12,n_informative=8,noise=22,random_state=42); y=np.maximum(0,y-y.min()+25)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
baseline=DummyRegressor(strategy="median").fit(X_train,y_train); model=Pipeline([("scale",StandardScaler()),("model",HistGradientBoostingRegressor(max_iter=250,l2_regularization=1,random_state=42))])
cv=cross_validate(model,X_train,y_train,cv=5,scoring=["neg_mean_absolute_error","r2"]); model.fit(X_train,y_train); pred=model.predict(X_test)
metrics={"baseline_mae":mean_absolute_error(y_test,baseline.predict(X_test)),"model_mae":mean_absolute_error(y_test,pred),"rmse":mean_squared_error(y_test,pred)**.5,"r2":r2_score(y_test,pred),"cv_mae_mean":-cv["test_neg_mean_absolute_error"].mean()}
joblib.dump(model,OUT/"regression_pipeline.joblib"); (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2)); pd.DataFrame({"actual":y_test,"prediction":pred,"residual":y_test-pred}).to_csv(OUT/"predictions.csv",index=False); print(metrics)

