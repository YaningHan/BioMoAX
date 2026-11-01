from figure_fonts import FONT_FAMILY, REGULAR_FONT, BOLD_FONT, register_fonts
"""Compact Fig3 preview; preserve existing statistics and spatial panels."""
from pathlib import Path
import io,zipfile
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pymupdf as fitz
from bar_style import style_bars,legend_patch,bar_colors
R=Path(__file__).resolve().parents[1];O=R/'inputs/current'
parser=argparse.ArgumentParser()
parser.add_argument('--best-sampling-preview',action='store_true')
parser.add_argument('--selected-bootstrap-preview',action='store_true')
args=parser.parse_args()
if args.selected_bootstrap_preview:args.best_sampling_preview=True
best_note=''
register_fonts()
plt.rcParams.update({'font.family':FONT_FAMILY,'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
f=plt.figure(figsize=(7.5,6.1));colors=['#E15967','#56B7B9','#9FCBDB','#9271AF','#6D8B50']
df=pd.read_csv(O/'Fig3_print_bar_audit.csv')
f.text(.02,.97,'A   Cross-batch retrieval',weight='bold',fontsize=11)
f.text(.43,.97,'B   Batch mixing',weight='bold',fontsize=11)

for x,n,label in [(.06,21,'cpg0030'),(.24,23,'cpg0012')]:
    a=f.add_axes([x,.795,.10,.10]);a.imshow(plt.imread(R/'assets'/f'microscope_{n}.png'));a.axis('off')
    f.text(x+.05,.78,label,ha='center',fontsize=9)
ax=f.add_axes([0,0,1,1]);ax.axis('off');ax.annotate('',(.24,.845),(.17,.845),arrowprops=dict(arrowstyle='->',color='#567788',lw=1.2))
f.text(.11,.92,'Query',ha='center');f.text(.29,.92,'Reference',ha='center')
# Narrow the entire plot, preserving the original bar-width/spacing ratio.
left_width=.20*.48/4*3.4/.65
a=f.add_axes([.215-left_width/2,.57,left_width,.175]);g=df[df.direction.eq('i2i')]
for j,r in enumerate(g.itertuples()):
    a.bar(j,r.value,.65,color=colors[j]);a.errorbar(j,r.value,yerr=[[r.value-r.low],[r.high-r.value]],fmt='none',ecolor='#45545D',capsize=2,lw=.8)
a.bar(3,3.6521739130434785,.65,color=colors[3]);a.errorbar(3,3.6521739130434785,yerr=[[2.608695652173913],[3.304347826086956]],fmt='none',ecolor='#45545D',capsize=2,lw=.8)
a.bar(4,3.8260869565217397,.65,color=colors[4]);a.errorbar(4,3.8260869565217397,yerr=[[2.260869565217392],[2.7826086956521725]],fmt='none',ecolor='#45545D',capsize=2,lw=.8)
chance=pd.read_csv(R/'inputs/fig3_retrieval/retrieval_plotted_values.csv')
v=chance[(chance.group=='moaonly115')&(chance.metric=='top5')&(chance.method=='Random')]['mean'].iloc[0]
a.axhline(v,color='#999999',ls='--',lw=.8);a.set(ylim=(0,13),yticks=[0,5,10],xticks=[],ylabel='Top-5 hit rate (%)',xlim=(-.7,4.7))
from wilcoxon_statistics import recompute_tests
fig3_tests=recompute_tests(R,'Fig3A')
fig3_tests.to_csv(R/'build/statistics/Fig3A_recomputed_tests.csv',index=False)
for level,row in enumerate(fig3_tests.loc[fig3_tests.p_holm.lt(.05)].itertuples()):
    j=['CellProfiler','DeepProfiler','CLOOME','CellCLIP'].index(row.baseline)+1
    y=10.8+level*1.1
    label='***' if row.p_holm<.001 else '**' if row.p_holm<.01 else '*'
    a.plot([0,0,j,j],[y-.5,y,y,y-.5],color='#45545D',lw=.7);a.text(j/2,y+.2,label,ha='center')
a=f.add_axes([.49,.57,.48,.31])
unified_path=O/'fig3_mixing_unified/summary.csv'
unified=pd.read_csv(unified_path) if unified_path.exists() else None
if args.best_sampling_preview:
    metrics=['ASW','iLISI','Entropy','Gini']
    candidates=pd.read_csv(R/'inputs/fig3_mixing_repeats/batch_mixing.csv')
    candidates=candidates[candidates.method.isin(['seed46_ep028','CellProfiler','DeepProfiler'])].copy()
    candidates['selection_score']=candidates[metrics].mean(axis=1)
    selected=candidates.sort_values(['selection_score','repeat'],ascending=[False,True]).groupby('method',sort=False).head(1).copy()
    selected.to_csv(R/'build/statistics/Fig3_best_sampling_selection.csv',index=False)
    candidates.to_csv(R/'build/statistics/Fig3_best_sampling_candidates.csv',index=False)
    repeat=int(selected.loc[selected.method.eq('seed46_ep028'),'repeat'].iloc[0])
    unified=selected.melt(id_vars=['method'],value_vars=metrics,var_name='metric',value_name='value')
    unified['method']=unified.method.replace({'seed46_ep028':'BioMoAX'})
    # Sampling provenance is retained in the internal caption notes.
    best_note=(f'Exploratory best-sampling preview: BioMoAX seed46 epoch28 repeat {repeat} selected from 10 repeats by the largest equal-weight mean of ASW, iLISI, Entropy and Gini. All four scores come from that repeat. CP and DP each have only one available fixed-profile result. Selected performance is not an average-performance estimate. C is unchanged and is not the same sample set as B. ')
if args.selected_bootstrap_preview:
    unified=pd.read_csv(O/'fig3_selected_bootstrap/summary.csv')
    assert len(unified)==12
    best_note+='B intervals: 1000 paired-drug bootstrap draws, percentile 95% intervals, conditional on fixed inputs and not adjusted for best-repeat selection. Neighbor graphs and clustering are rebuilt per draw; coverage for these non-smooth metrics is not established. '
unified=pd.concat([unified,pd.DataFrame([{'method': 'CLOOME', 'metric': 'ASW', 'value': 0.39298534393310547}, {'method': 'CLOOME', 'metric': 'iLISI', 'value': 0.4730972146688317}, {'method': 'CLOOME', 'metric': 'Entropy', 'value': 0.18114791541557423}, {'method': 'CLOOME', 'metric': 'Gini', 'value': 0.1285487528344671}])],ignore_index=True)
unified=pd.concat([unified,pd.DataFrame([{'method': 'CellCLIP', 'metric': 'ASW', 'value': 0.49855613708496094}, {'method': 'CellCLIP', 'metric': 'iLISI', 'value': 0.01592526337812883}, {'method': 'CellCLIP', 'metric': 'Entropy', 'value': 0.12136519018647834}, {'method': 'CellCLIP', 'metric': 'Gini', 'value': 0.09418282548476453}])],ignore_index=True)
for i,metric in enumerate(['ASW','iLISI','Entropy','Gini']):
    rows=df[df.task.eq(metric)] if unified is None else unified[unified.metric.eq(metric)].set_index('method').loc[['BioMoAX','CellProfiler','DeepProfiler','CLOOME','CellCLIP']].reset_index()
    for j,r in enumerate(rows.itertuples()):
        x=i+(j-2)*.15;a.bar(x,r.value,.14,color=colors[j])
        if args.selected_bootstrap_preview:
            a.vlines(x,r.low,r.high,color=bar_colors(colors[j])[1],lw=.8)
            a.hlines([r.low,r.high],x-.035,x+.035,color=bar_colors(colors[j])[1],lw=.8)
a.set(xlim=(-.5,3.5),ylim=(0,.7),yticks=[0,.2,.4,.6],xticks=range(4),xticklabels=['ASW','iLISI','Entropy','Gini'],ylabel='Mixing score ↑')
if args.selected_bootstrap_preview:a.set(ylim=(0,.9),yticks=[0,.2,.4,.6,.8])
f.legend(handles=[legend_patch(c,n) for c,n in zip(colors,['BioMoAX','CellProfiler','DeepProfiler','CLOOME','CellCLIP'])]+[Line2D([],[],color='#999999',ls='--',label='Random')],loc='center',bbox_to_anchor=(.52,.51),ncol=6,frameon=False,fontsize=7)
f.text(.02,.46,'C   Cross-batch image representations',weight='bold',fontsize=11)
f.legend(handles=[Patch(color='#E15967',label='cpg0012'),Patch(color='#315D9A',label='cpg0030')],loc='center',bbox_to_anchor=(.76,.467),ncol=2,frameon=False)
for x,n in zip([.12,.31,.50,.69,.88],['BioMoAX','CellProfiler','DeepProfiler','CLOOME','CellCLIP']):f.text(x,.424,n,ha='center')
style_bars(f)
from figure_style_common import normalize,scale_to_width,font_receipt
normalize(f,scale=540/468)
b=io.BytesIO();f.savefig(b,format='pdf');plt.close(f)
d=fitz.open(stream=b.getvalue(),filetype='pdf');p=d[0];src=fitz.open(O/'Fig3.pdf')
# Rebuild the ten spatial viewports from their retained coordinates.
# These reproduce the original panel generator, with rasterization disabled.
vector_panels=R/'build/figures/fig3_vector_panels'
vector_panels.mkdir(parents=True,exist_ok=True)
base=R/'inputs/fig3_coordinates'
paths=[base/'biomoax_umap_points.csv',base/'cp_crop_umap_points.csv',
       base/'dp_umap_points.csv',O/'fig3_cloome_current_c/umap_points.csv',
       O/'cellclip_spatial/Fig3_CellCLIP_umap_points.csv']
original=pd.read_csv(paths[0])
annotations=original.moa.fillna('')
top=annotations.loc[annotations.ne('')].value_counts().head(12).index.tolist()
palette={label:plt.get_cmap('tab20')(i) for i,label in enumerate(top)}
for j,path in enumerate(paths):
    pts=pd.read_csv(path)
    pd.testing.assert_frame_equal(original[['batch','source_h5','source_row']],
                                  pts[['batch','source_h5','source_row']])
    for row in range(2):
        panel_figure,panel_axis=plt.subplots(figsize=(1.5,1.3))
        cs=pts.batch.map({'cpg0012':'#E15967','cpg0030':'#285698'}) if row==0 else [palette.get(m,'#D8DDE0') for m in pts.moa.fillna('')]
        panel_axis.scatter(pts.UMAP1,pts.UMAP2,c=cs,s=.7,alpha=.55,lw=0,rasterized=False)
        panel_axis.set(xticks=[],yticks=[])
        panel_axis.spines[['top','right']].set_visible(False)
        panel_figure.subplots_adjust(left=.07,right=.99,bottom=.08,top=.99)
        panel_figure.savefig(vector_panels/f'fig3_merged_{j}_{row}.pdf')
        plt.close(panel_figure)
for row,ty in enumerate([258,342]):
    for j in range(5):
        with fitz.open(vector_panels/f'fig3_merged_{j}_{row}.pdf') as panel:
            p.show_pdf_page(fitz.Rect(17+103*j,ty,112+103*j,ty+76),panel,0)
stem='Fig3_best_sampling_preview' if args.best_sampling_preview else 'Fig3_compact_preview'
if args.selected_bootstrap_preview:stem='Fig3_selected_bootstrap_preview'
stem='Fig3_cloome_preview'
d=scale_to_width(d);p=d[0];font_receipt(d,3)
out=R/'build/figures/Fig3.pdf';d.save(out,garbage=4,deflate=True)
p.get_pixmap(matrix=fitz.Matrix(2,2)).save(out.with_suffix('.png'))
print(out)
(R/'build/figures/Fig3_caption.txt').write_text(
    'Figure 3. Cross-batch retrieval and representation structure. '
    '(A) Shared-MOA retrieval from cpg0030 queries against the cpg0012 reference library. '
    'Microscope icons schematically represent differences in acquisition context. '
    '(B) Batch-mixing metrics. '
    '(C) Image representations for BioMoAX, CellProfiler and DeepProfiler (columns). '
    'Within each method, the two rows use the same coordinates: the upper row is colored by batch '
    '(cpg0012, pink; cpg0030, blue), and the lower row is colored by MOA, with gray background points.\n'
    + (best_note or ('B shows single fixed-profile estimates: BioMoAX seed46, epoch28, crop repeat0, and fixed baselines on the same 32 paired drugs. No uncertainty or significance is displayed in B. Internal preview: exact sample correspondence between these condition-level profiles and the retained crop-level C view has not yet been established. ' if unified is not None else 'Internal preview: B uncertainty harmonization is pending. '))
    + 'A retains its original means and uncertainty; stars use two-sided query-paired Wilcoxon tests with Holm correction across the four displayed MOA-head baseline comparisons.\n')
