"""Production monitoring example: feature drift, prediction drift, and alerting."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import ks_2samp

OUT=Path("outputs/monitoring"); OUT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(42)
reference=pd.DataFrame({"age":rng.normal(42,11,5000),"income":rng.lognormal(10.8,.45,5000),"score":rng.beta(3,4,5000),"prediction":rng.beta(2,7,5000)})
current=pd.DataFrame({"age":rng.normal(45,12,3000),"income":rng.lognormal(10.95,.5,3000),"score":rng.beta(3.5,3.5,3000),"prediction":rng.beta(2.5,6,3000)})
rows=[]
for col in reference:
    stat,p=ks_2samp(reference[col],current[col]); rows.append((col,reference[col].mean(),current[col].mean(),stat,p,"alert" if stat>.1 else "ok"))
report=pd.DataFrame(rows,columns=["feature","reference_mean","current_mean","ks_statistic","p_value","status"]); report.to_csv(OUT/"drift_report.csv",index=False); print(report.round(4))
