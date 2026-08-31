"""Unsupervised workflow with scaling, PCA, silhouette selection, and profiles."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUT=Path("outputs/clustering"); OUT.mkdir(parents=True,exist_ok=True)
X,_=make_blobs(n_samples=3000,n_features=6,centers=4,cluster_std=[1.2,1.8,1.4,2],random_state=42); columns=["engagement","frequency","value","tenure","support","discount"]
df=pd.DataFrame(X,columns=columns); scaled=StandardScaler().fit_transform(df)
scores={k:silhouette_score(scaled,KMeans(k,random_state=42,n_init=20).fit_predict(scaled),sample_size=1500,random_state=42) for k in range(2,9)}; best_k=max(scores,key=scores.get)
model=KMeans(best_k,random_state=42,n_init=20); df["cluster"]=model.fit_predict(scaled); profile=df.groupby("cluster").mean(); coords=PCA(2,random_state=42).fit_transform(scaled)
df.to_csv(OUT/"cluster_assignments.csv",index=False); profile.to_csv(OUT/"cluster_profiles.csv"); pd.DataFrame({"k":scores.keys(),"silhouette":scores.values()}).to_csv(OUT/"model_selection.csv",index=False); pd.DataFrame(coords,columns=["pc1","pc2"]).assign(cluster=df.cluster).to_csv(OUT/"pca_projection.csv",index=False); print("best_k",best_k,"score",scores[best_k]); print(profile.round(2))

