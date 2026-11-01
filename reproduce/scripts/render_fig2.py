from figure_fonts import FONT_FAMILY, REGULAR_FONT, BOLD_FONT, register_fonts
from crop_assets import CropFile
"""Independent Fig2 preview from completed matched evaluations; no formal edits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

R=Path(__file__).resolve().parents[1]
OUT=R/'inputs/current'
register_fonts()
plt.rcParams.update({'font.family':FONT_FAMILY,'font.size':10,'axes.titlesize':11,
                     'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors={'BioMoAX':'#E15967','CellProfiler':'#56B7B9','DeepProfiler':'#9FCBDB','CLOOME':'#9271AF','CellCLIP':'#6D8B50'}
methods=['BioMoAX','DeepProfiler','CLOOME']
tests=[]
def paired_test(panel,cell,k,values):
    for baseline in methods[1:]:
        pair=pd.concat([values['BioMoAX'].rename('model'),values[baseline].rename('baseline')],axis=1)
        assert pair.notna().all().all()
        delta=(pair.model-pair.baseline).to_numpy()
        observed=abs(delta.mean());rng=np.random.default_rng(42);extreme=0
        for _ in range(100):
            signs=rng.integers(0,2,(1000,len(delta)))*2-1
            extreme+=int((np.abs((signs*delta).mean(axis=1))>=observed-1e-12).sum())
        tests.append(dict(panel=panel,cell=cell,k=k,baseline=baseline,n_queries=len(delta),
            difference_pp=delta.mean()*100,p_raw=(extreme+1)/100001))
fig=plt.figure(figsize=(11,9.6))
def axis(x,y,w,h):return fig.add_axes([x,y,w,h])
def heading(y,label,title):
    fig.text(.035,y,label,weight='bold',fontsize=14)
    fig.text(.065,y,title,weight='bold',fontsize=13)
heading(.965,'A','Same-compound retrieval from held-out image crops')
from matplotlib.patches import FancyArrowPatch
# The schematic describes disjoint crop sets, not a train/test compound split.
fig.text(.065,.928,'Query crops (20%)',fontsize=10,va='center')
fig.text(.70,.928,'Reference crops (80%)',fontsize=10,va='center')
fig.text(.49,.946,'Same compound · disjoint crops',fontsize=9,ha='center',color='#526574')
fig.add_artist(FancyArrowPatch((.335,.928),(.625,.928),transform=fig.transFigure,
    arrowstyle='-|>',mutation_scale=12,color='#8BAE70',lw=1.5))
audit=[];tests=[]
old_identity=pd.read_csv(R/'inputs/fig2_identity/curve_summary.csv')
old_identity=pd.concat([old_identity,pd.read_csv(OUT/'Fig2A_cloome_aligned_summary.csv')],ignore_index=True)
cellclip_identity=OUT/'Fig2A_cellclip_aligned_summary.csv'
if cellclip_identity.exists():
    old_identity=pd.concat([old_identity,pd.read_csv(cellclip_identity)],ignore_index=True)
for j,(cell,n) in enumerate([('U2OS',90),('A549',301)]):
    ax=axis(.09+j*.47,.775,.38,.12)
    for key,name in [('compound','BioMoAX'),('CellProfiler','CellProfiler'),('DeepProfiler','DeepProfiler'),('CLOOME','CLOOME'),('CellCLIP','CellCLIP')]:
        g=old_identity[(old_identity.cell==cell)&(old_identity.method==key)].sort_values('k')
        if g.empty:continue
        x=g.k.to_numpy();y=g['mean'].to_numpy();sd=g.sd.to_numpy()
        ax.fill_between(x,np.clip(y-sd,0,100),np.clip(y+sd,0,100),color=colors[name],alpha=.18,lw=0)
        ax.plot(x,y,color=colors[name],lw=1.6,marker='o',ms=3)
        for row in g.to_dict('records'):audit.append(dict(panel='A',**row))
    ax.plot(range(1,6),np.arange(1,6)/n*100,c='#999999',ls='--',lw=1)
    ax.set(title=cell,ylim=(0,104),yticks=[0,50,100],xticks=range(1,6),xlabel='Top K')
    if j==0:ax.set_ylabel('Hit rate (%)')
handles=[Line2D([],[],color=colors[m],lw=2,marker='o',ms=3,label=m) for m in ['BioMoAX','CellProfiler','DeepProfiler','CLOOME']]
if cellclip_identity.exists():
    handles.append(Line2D([],[],color=colors['CellCLIP'],lw=2,marker='o',ms=3,label='CellCLIP'))
handles.append(Line2D([],[],color='#999999',ls='--',label='Random (A)'))
fig.legend(handles=handles,loc='center',bbox_to_anchor=(.52,.715),ncol=5,frameon=False)
# Small real crops in the identity schematic, placed after the panel restoration.
example=np.load(R/'inputs/fig2_identity/U2OS_five_splits.npz')
same=example['drug_index']==0
query_rows=example['h5_rows'][same&(example['partitions'][0]==0)][:1]
reference_rows=example['h5_rows'][same&(example['partitions'][0]==1)][:2]
for x,rows in [(.245,query_rows),(.635,reference_rows)]:
    for offset,row in enumerate(rows):
        with CropFile('crops_cpg0012_690') as f:
            a=f['crops'][row].astype(float)
        rgb=[]
        for ch in [2,1,0]:
            low,high=np.percentile(a[ch],[1,99])
            rgb.append(np.clip((a[ch]-low)/max(high-low,1),0,1))
        ia=axis(x+offset*.016,.914+offset*.003,.027,.031)
        ia.imshow(np.stack(rgb,-1));ia.axis('off')
heading(.635,'B','Retrieving drugs with shared MOA')
# Retain the original query/gallery schematic and four modality columns.
for x,row in [(.30,210),(.60,310)]:
    with CropFile('crops_cpg0012_690') as f:
        a=f['crops'][row].astype(float)
    rgb=[]
    for ch in [2,1,0]:
        low,high=np.percentile(a[ch],[1,99])
        rgb.append(np.clip((a[ch]-low)/max(high-low,1),0,1))
    ia=axis(x,.580,.040,.046);ia.imshow(np.stack(rgb,-1));ia.axis('off')
    from matplotlib.colors import ListedColormap
    ha=axis(x+.047,.580,.032,.046)
    ha.imshow([[0,0,1,0],[0,1,1,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],cmap=ListedColormap(['#B1CDD4','#D2AD9E']))
    ha.set_xticks(np.arange(-.5,4,1),minor=True);ha.set_yticks(np.arange(-.5,5,1),minor=True)
    ha.grid(which='minor',color='white',lw=.5)
    ha.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    for sp in ha.spines.values():sp.set_visible(False)
fig.text(.285,.597,'Query',ha='right',fontsize=10)
fig.text(.690,.597,'Reference',fontsize=10)
fig.text(.49,.559,'Other compounds · shared MOA · query compound excluded',ha='center',fontsize=9,color='#526574')
from matplotlib.patches import FancyArrowPatch
fig.add_artist(FancyArrowPatch((.400,.602),(.575,.602),transform=fig.transFigure,
    arrowstyle='-|>',mutation_scale=12,color='#8BAE70',lw=1.8))
axes=[axis(.075+j*.235,.405,.185,.112) for j in range(4)]
# Replace the provisional image-only statistics with audited retained ranks.
reused=pd.read_csv(OUT/'Fig2B_reused_statistics.csv')
reused_tests=pd.read_csv(OUT/'Fig2B_reused_pvalues.csv')
ns_selection=pd.read_csv(OUT/'Fig2B_ns_comparisons.csv')
reused_tests=reused_tests.merge(ns_selection.assign(display_ns=True),on=['cell','direction','baseline'],how='left',validate='one_to_one')
reused_tests['display_ns']=reused_tests.display_ns.eq(True)
assert reused_tests.loc[reused_tests.display_ns,'p_holm'].ge(.05).all()
annotation_records=[]
repeat_values=pd.read_csv(OUT/'Fig2B_reused_repeat_values.csv')
audit=[row for row in audit if row['panel']!='B']
colors.update({'Raw L1000':'#FF8956','CP + Ridge':colors['CellProfiler'],'DP + Ridge':colors['DeepProfiler'],'CLOOME + Ridge':colors['CLOOME']})
for ax,direction,title in zip(axes,['i2i','e2e','i2e','e2i'],['I2I','E2E','I2E','E2I']):
    ax.clear()
    names=reused[reused.direction.eq(direction)&reused.method.ne('Random')].method.drop_duplicates().tolist()
    for idx,cell in enumerate(['U2OS','A549']):
        chance=reused[(reused.direction==direction)&(reused.cell==cell)&(reused.method=='Random')]['mean'].item()
        ax.hlines(chance,idx-.40,idx+.40,color='#888888',linestyle='--',linewidth=.9,zorder=3)
    width=min(.22,.80/len(names))
    for j,name in enumerate(names):
        g=reused[reused.direction.eq(direction)&reused.method.eq(name)].set_index('cell').loc[['U2OS','A549']]
        y=g['mean'].to_numpy();x=np.arange(2)+(j-(len(names)-1)/2)*width
        from matplotlib.colors import to_rgb
        base=np.array(to_rgb(colors[name]));edge=base*.72;fill=base*.48+.52
        ax.bar(x,y,width=width*.91,color=fill,edgecolor=edge,linewidth=1.1,
               yerr=np.stack([y-g.ci_low,g.ci_high-y]),error_kw={'elinewidth':1,'capsize':2.5,'ecolor':edge})
        for cell,row in g.iterrows():audit.append(dict(panel='B',cell=cell,**row.to_dict()))
    panel_peak=float(reused.loc[reused.direction.eq(direction),'ci_high'].max())
    annotation_peak=panel_peak
    for idx,cell in enumerate(['U2OS','A549']):
        sig=reused_tests[(reused_tests.cell==cell)&(reused_tests.direction==direction)&((reused_tests.p_holm<.05)|reused_tests.display_ns)]
        for level,row in enumerate(sig.itertuples()):
            cell_peak=float(reused.loc[(reused.direction==direction)&(reused.cell==cell),'ci_high'].max())
            left=idx-(len(names)-1)/2*width;right=left+names.index(row.baseline)*width;yy=cell_peak+3+level*5
            annotation_peak=max(annotation_peak,yy+3)
            ax.plot([left,left,right,right],[yy-1,yy,yy,yy-1],c='#45545D',lw=.7)
            label='***' if row.p_holm<.001 else '**' if row.p_holm<.01 else '*' if row.p_holm<.05 else 'ns'
            ax.text((left+right)/2,yy+.5,label,ha='center',fontsize=9)
            annotation_records.append(dict(cell=cell,direction=direction,baseline=row.baseline,p_holm=row.p_holm,label=label))
    ax.set_title(title,fontsize=10,pad=8,linespacing=1.05)
    ax.spines['left'].set_linewidth(1);ax.spines['bottom'].set_linewidth(1)
    upper=float(np.ceil((annotation_peak+1)/5)*5)
    ax.set(xticks=[0,1],xticklabels=['U2OS','A549'],ylim=(0,upper),yticks=np.arange(0,upper+1,10),xlim=(-.5,1.5))
axes[0].set_ylabel('Top-5 hit rate (%)')
pd.DataFrame(annotation_records).to_csv(R/'build/statistics/Fig2_displayed_annotations.csv',index=False)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=colors[m],label=label) for m,label in
    [('BioMoAX','BioMoAX'),('CellProfiler','CellProfiler'),('DeepProfiler','DeepProfiler'),('Raw L1000','L1000')]+([('CLOOME','CLOOME')] if 'CLOOME' in reused.method.values else [])]+[Line2D([],[],color='#888888',ls='--',label='Random')],
    loc='center',bbox_to_anchor=(.51,.359),ncol=6,frameon=False,fontsize=10,
    handlelength=1,handletextpad=.5,columnspacing=1.5)
heading(.310,'C','MOA neighborhoods of image crops')
fig.text(.065,.286,'Same A549 crops across methods · 20 MOAs × 4 drugs × 3 highlighted crops',fontsize=10,color='#526574')
source=R/'inputs/fig2_coordinates'
preview=R/'inputs/fig2_highlights'
sel=pd.read_csv(preview/'highlighted_crops.csv');moas=sorted(sel.moa.unique());palette=plt.get_cmap('tab20').colors
for j,(slug,method) in enumerate([('biomoax','BioMoAX'),('cp','CellProfiler'),('dp','DeepProfiler'),('cloome','CLOOME'),('cellclip','CellCLIP')]):
    xy=np.load(OUT/'cellclip_spatial/fig2_coordinates.npy' if slug=='cellclip' else (preview if slug=='cloome' else source)/f'{slug}_coordinates.npy')
    ax=axis(.045+j*.245,.062,.215,.185)
    ax.scatter(xy[:,0],xy[:,1],s=.45,c='#ADB7BE',alpha=.17,lw=0,rasterized=False)
    for moa,color in zip(moas,palette):
        for marker,g in sel[sel.moa.eq(moa)].groupby('marker'):
            z=xy[g.row.to_numpy()];ax.scatter(z[:,0],z[:,1],s=4,c=[color],marker=marker,
                edgecolors='white',linewidths=.08,alpha=.9,rasterized=False)
    lo=xy.min(0);hi=xy.max(0);pad=(hi-lo)*.04
    ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),xticks=[],yticks=[],title=method)
    ax.set_box_aspect(1)
fig.text(.5,.040,'Color: MOA     Shape: drug within MOA     Gray: background crops',ha='center',fontsize=9)
fig.text(.5,.015,'Preview · A: original curves + aligned CLOOME · B: query-bootstrap 95% CI · CellCLIP pending',ha='center',fontsize=9,color='#526574')
# Close the space released by the shorter A plots; retain B/C internal layout.
for ax in fig.axes:
    pos=ax.get_position()
    if pos.y0 < .70:ax.set_position([pos.x0,pos.y0+.03,pos.width,pos.height])
for txt in fig.texts:
    x,y=txt.get_position()
    if y < .70:txt.set_position((x,y+.03))
for artist in fig.artists:
    if isinstance(artist,FancyArrowPatch):
        start,end=artist._posA_posB
        if start[1] < .70:artist.set_positions((start[0],start[1]+.03),(end[0],end[1]+.03))
for legend in fig.legends:
    pos=legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    if pos.y0 < .70:legend.set_bbox_to_anchor((pos.x0,pos.y0+.03))
path=R/'build/figures/Fig2_working.pdf'
fig.savefig(path);fig.savefig(path.with_suffix('.png'),dpi=160)
from fig2_layout import save_layout_trial
save_layout_trial(fig,R/'build/figures',colors,reused)
plt.close(fig)
pd.DataFrame(audit).to_csv(R/'build/statistics/Fig2_rendered_values.csv',index=False)
