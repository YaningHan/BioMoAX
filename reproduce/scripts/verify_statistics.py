"""Recompute plotted statistical summaries from bundled per-query observations."""
from pathlib import Path
import json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];I=R/'inputs/current';O=R/'build/statistics';O.mkdir(parents=True,exist_ok=True)
checks=[]
def check(name,a,b,tol=1e-9):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 assert a.shape==b.shape,(name,a.shape,b.shape)
 ok=np.allclose(a,b,atol=tol,rtol=0,equal_nan=True)
 checks.append(dict(name=name,n_values=int(a.size),passed=bool(ok),tolerance=tol,max_abs_error=float(np.nanmax(abs(a-b))) if a.size else 0))
 assert ok,checks[-1]
# Fig2: each query is averaged across its existing crop repeats.
q=pd.read_csv(I/'Fig2B_reused_query_values.csv');target=pd.read_csv(I/'Fig2B_reused_statistics.csv');pt=pd.read_csv(I/'Fig2B_reused_pvalues.csv')
rows=[];tests=[]
for (cell,direction,method),g in q.groupby(['cell','direction','method']):
 g=g.sort_values('query_id');a=g.hit_fraction.to_numpy();rng=np.random.default_rng(42)
 lo,hi=np.quantile(a[rng.integers(0,len(a),(10000,len(a)))].mean(1)*100,[.025,.975])
 expected=target[(target.cell==cell)&(target.direction==direction)&(target.method==method)].iloc[0]
 check(f'Fig2/{cell}/{direction}/{method}',[a.mean()*100,lo,hi],[expected['mean'],expected.ci_low,expected.ci_high])
 rows.append(dict(cell=cell,direction=direction,method=method,mean=a.mean()*100,ci_low=lo,ci_high=hi))
from wilcoxon_statistics import recompute_tests
t=recompute_tests(R,'Fig2B')
key=['cell','direction','baseline'];check('Fig2 Wilcoxon p-values',t.set_index(key).sort_index()[['p_raw','p_holm']],pt.set_index(key).sort_index()[['p_raw','p_holm']],tol=1e-12)
pd.DataFrame(rows).to_csv(O/'Fig2_recomputed.csv',index=False);t.to_csv(O/'Fig2_recomputed_tests.csv',index=False)
t4=recompute_tests(R,'Fig4');target4=pd.read_csv(I/'fig4_query_ci/tests.csv').rename(columns={'p_holm_figure':'p_holm'})
key4=['task','direction','head','baseline'];check('Fig4 Wilcoxon p-values',t4.set_index(key4).sort_index()[['p_raw','p_holm']],target4.set_index(key4).sort_index()[['p_raw','p_holm']],tol=1e-12)
t4.to_csv(O/'Fig4_recomputed_tests.csv',index=False)
# Fig4: preserve the exact fixed-query averaging and 20,000 bootstrap draws.
q=pd.read_csv(I/'fig4_query_ci/paired_query_values.csv',dtype={'query_key':str});ref=pd.read_csv(I/'fig4_query_ci/summary.csv')
rows=[]
for (task,direction,head),g in q.groupby(['task','direction','head']):
 pivot=g.pivot(index='query_key',columns='baseline',values='model_hit_fraction')
 assert not pivot.isna().any().any();model=pivot.iloc[:,0].to_numpy()
 rng=np.random.default_rng(42);ix=rng.integers(0,len(model),(20000,len(model)))
 vals={'BioMoAX':model}
 for method,b in g.groupby('baseline'):vals[method]=b.set_index('query_key').loc[pivot.index].baseline_hit_fraction.to_numpy()
 for method,a in vals.items():
  lo,hi=np.quantile(a[ix].mean(1)*100,[.025,.975]);e=ref[(ref.task==task)&(ref.direction==direction)&(ref['head']==head)&(ref.method==method)].iloc[0]
  check(f'Fig4/{task}/{direction}/{head}/{method}',[a.mean()*100,lo,hi],[e.value,e.low,e.high])
  rows.append(dict(task=task,direction=direction,head=head,method=method,value=a.mean()*100,low=lo,high=hi))
pd.DataFrame(rows).to_csv(O/'Fig4_recomputed.csv',index=False)
# Fig5: fixed 29 genes, three checkpoints and matched baseline query ranks.
D=R/'inputs/fig5_query_results';model=[]
for ck in ['seed43_ep029','seed45_ep028','seed46_ep028']:
 p=pd.read_csv(D/f'A549_{ck}_ranks.csv');p=p[p['head'].eq('moa')].copy();p['checkpoint']=ck;model.append(p)
model=pd.concat(model);base=pd.read_csv(D/'baseline_query_ranks.csv');base=base[base.cell.eq('A549')]
others={}
for method,slug in [('CLOOME','cloome'),('CellCLIP','cellclip')]:
 p=pd.read_csv(R/f'inputs/baselines/{slug}/query_ranks.csv');p=p[p.task.eq('CRISPR_to_drug')].copy();p['gene']=p['query'].str.removeprefix('gene:');others[method]=p
genes=sorted(model.gene.unique());assert len(genes)==29
ix=np.random.default_rng(42).integers(0,29,(20000,29));ref=pd.read_csv(I/'Fig5_single_case_curve_audit.csv');rows=[]
for k in range(1,11):
 series={'BioMoAX':model.assign(hit=model['rank'].le(k)).groupby(['checkpoint','gene']).hit.mean().groupby('gene').mean()}
 for method,p in others.items():series[method]=p.assign(hit=p['rank'].le(k)).groupby('gene').hit.mean()
 for method in ['CellProfiler','DeepProfiler']:
  p=base[base.method.eq(method)];series[method]=p.assign(hit=p['rank'].le(k)).groupby('gene').hit.mean()
 for method,s in series.items():
  a=s.loc[genes].to_numpy();lo,hi=np.quantile(a[ix].mean(1)*100,[.025,.975]);e=ref[(ref.method==method)&(ref.k==k)].iloc[0]
  check(f'Fig5/{method}/Top{k}',[a.mean()*100,lo,hi],[e['mean'],e.low,e.high]);rows.append(dict(method=method,k=k,mean=a.mean()*100,low=lo,high=hi))
pd.DataFrame(rows).to_csv(O/'Fig5_recomputed.csv',index=False)
# Fig3A: query-paired Wilcoxon tests, four currently displayed baselines.
t3=recompute_tests(R,'Fig3A');target3=pd.read_csv(I/'fig3_query_tests/tests.csv')
check('Fig3A Wilcoxon p-values',t3.set_index('baseline').sort_index()[['p_raw','p_holm']],target3.set_index('baseline').sort_index()[['p_raw','p_holm']],tol=1e-12)
q3=pd.read_csv(I/'fig3_query_tests/paired_query_values.csv')
for baseline,g in q3.groupby('baseline'):
 e=target3[target3.baseline.eq(baseline)].iloc[0]
 check(f'Fig3A/{baseline}/means',[g.model_hit_fraction.mean()*100,g.baseline_hit_fraction.mean()*100],[e.model_percent,e.baseline_percent],tol=1e-10)
t3.to_csv(O/'Fig3A_recomputed_tests.csv',index=False)
# Fig3 selection is intentionally preserved, not converted into an average estimate.
c=pd.read_csv(R/'inputs/fig3_mixing_repeats/batch_mixing.csv');c=c[c.method.isin(['seed46_ep028','CellProfiler','DeepProfiler'])].copy();metrics=['ASW','iLISI','Entropy','Gini'];c['selection_score']=c[metrics].mean(axis=1)
selected=c.sort_values(['selection_score','repeat'],ascending=[False,True]).groupby('method',sort=False).head(1)
e=pd.read_csv(I/'Fig3_best_sampling_selection.csv')
check('Fig3 frozen best-repeat selection',selected.set_index('method').sort_index()[['repeat']+metrics],e.set_index('method').sort_index()[['repeat']+metrics])
report=dict(passed=all(x['passed'] for x in checks),checks=checks,scope='Frozen-query statistics and selection; no model inference or training rerun. Fig2B/Fig3A/Fig4 Wilcoxon p-values independently recomputed; all plotted means and CIs unchanged. Fig2 identity remains frozen.')
(R/'validation/statistics_comparison.json').write_text(json.dumps(report,indent=2))
print('STATISTICS_PASS',len(checks),'checks')
