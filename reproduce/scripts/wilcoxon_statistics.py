from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
def exact_wilcoxon(d):
 d=np.round(np.asarray(d,float),12);d[np.abs(d)<=1e-12]=0.;d=d[d!=0]
 if not len(d):return 1.
 weights=np.rint(2*rankdata(np.abs(d),method="average")).astype(int);p=np.array([1.])
 for w in weights:
  q=np.zeros(len(p)+w);q[:len(p)]=p*.5;q[w:]+=p*.5;p=q
 observed=int(weights[d>0].sum());return min(1.,float(2*min(p[:observed+1].sum(),p[observed:].sum())))
def holm(p):
 p=np.asarray(p);i=np.argsort(p,kind="stable");out=np.empty(len(p));out[i]=np.minimum(1,np.maximum.accumulate(p[i]*np.arange(len(p),0,-1)));return out
def recompute_tests(R,family):
 I=R/"inputs/current";records=[]
 if family=="Fig2B":
  q=pd.read_csv(I/"Fig2B_reused_query_values.csv");target=pd.read_csv(I/"Fig2B_reused_pvalues.csv")
  for r in target.itertuples():
   g=q[q.cell.eq(r.cell)&q.direction.eq(r.direction)].pivot(index="query_id",columns="method",values="hit_fraction");records.append(dict(cell=r.cell,direction=r.direction,baseline=r.baseline,p_raw=exact_wilcoxon(g.BioMoAX-g[r.baseline])))
 elif family=="Fig3A":
  q=pd.read_csv(I/"fig3_query_tests/paired_query_values.csv")
  for baseline,g in q.groupby("baseline"):
   assert len(g)==115 and not g.query_key.duplicated().any()
   records.append(dict(baseline=baseline,p_raw=exact_wilcoxon(g.model_hit_fraction-g.baseline_hit_fraction)))
 elif family=="Fig4":
  q=pd.read_csv(I/"fig4_query_ci/paired_query_values.csv");target=pd.read_csv(I/"fig4_query_ci/tests.csv")
  for r in target.itertuples():
   g=q[q.task.eq(r.task)&q.direction.eq(r.direction)&q["head"].eq(r.head)&q.baseline.eq(r.baseline)];assert len(g)==r.n and not g.query_key.duplicated().any();records.append(dict(task=r.task,direction=r.direction,head=r.head,baseline=r.baseline,p_raw=exact_wilcoxon(g.model_hit_fraction-g.baseline_hit_fraction)))
 else:
  raise ValueError(f"Unknown test family: {family}")
 t=pd.DataFrame(records);t["p_holm"]=holm(t.p_raw);return t
