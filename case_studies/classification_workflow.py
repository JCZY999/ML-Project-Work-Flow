"""End-to-end classification workflow: data, pipeline, CV, tuning, threshold, export."""
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,roc_auc_score,average_precision_score,confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

OUT=Path("outputs/classification"); OUT.mkdir(parents=True,exist_ok=True)
X,y=make_classification(n_samples=8000,n_features=8,n_informative=5,weights=[.78,.22],class_sep=1.1,random_state=42)
df=pd.DataFrame(X,columns=[f"feature_{i}" for i in range(8)]); df["plan"]=pd.qcut(df.feature_0,3,labels=["Basic","Plus","Premium"]); df.loc[df.sample(frac=.04,random_state=1).index,"feature_2"]=np.nan
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=.2,stratify=y,random_state=42)
numeric=[c for c in df if c.startswith("feature")]; categorical=["plan"]
preprocess=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("cat",OneHotEncoder(handle_unknown="ignore"),categorical)])
pipe=Pipeline([("preprocess",preprocess),("model",LogisticRegression(max_iter=2000,class_weight="balanced"))])
search=GridSearchCV(pipe,{"model__C":[.1,1,10]},scoring="roc_auc",cv=5,n_jobs=-1); search.fit(X_train,y_train)
prob=search.predict_proba(X_test)[:,1]; threshold=.35; pred=(prob>=threshold).astype(int)
metrics={"roc_auc":roc_auc_score(y_test,prob),"average_precision":average_precision_score(y_test,prob),"threshold":threshold,"best_C":search.best_params_["model__C"]}
joblib.dump(search.best_estimator_,OUT/"classification_pipeline.joblib"); (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2)); pd.DataFrame(confusion_matrix(y_test,pred),index=["actual_0","actual_1"],columns=["pred_0","pred_1"]).to_csv(OUT/"confusion_matrix.csv")
print(metrics); print(classification_report(y_test,pred))

