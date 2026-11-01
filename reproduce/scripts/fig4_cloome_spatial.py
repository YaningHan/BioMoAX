"""Four matched cross-cell condition-profile views for the Fig4 preview."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
OUT = R/'inputs/current/fig4_cloome_spatial'

def draw(f,h,ns,offset=0):
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection
    from itertools import combinations
    source=R/'inputs/fig4_coordinates'
    highlight=pd.read_csv(R/'inputs/fig4_highlights/highlighted_drug_profiles.csv')
    palette=['#C65A55','#357BB2','#5B9B60','#A778B6','#D69C31','#39A3A3','#A87857','#D36BA0','#77783C','#803E75','#3B6670','#E47C43']
    colors=dict(zip(highlight.moa.drop_duplicates(),palette))
    markers={'U2OS':'o','A549':'s','HT29':'^'}
    cellcolors={'U2OS':'#3878B5','A549':'#E49B35','HT29':'#B14583'}
    viewports={v['method']:v['limits'] for v in json.loads((R/'inputs/fig4_highlights/viewport_audit.json').read_text())}
    ns['heading'](f,h,4,337+offset,'C','Cross-cell-line image representations')
    f.legend(handles=[Line2D([],[],marker=markers[c],ls='',color=cellcolors[c],markersize=4,label=c) for c in markers],loc='center',bbox_to_anchor=(.80,1-(335+offset)/h),ncol=3,frameon=False,fontsize=8)
    link_records=[]
    for col,method in enumerate(['BioMoAX','CellProfiler','DeepProfiler','CLOOME','CellCLIP']):
        folder=R/'inputs/current/cellclip_spatial' if method=='CellCLIP' else (OUT if method=='CLOOME' else source)
        p=pd.read_csv(folder/f'{method}_projection_points.csv')
        for row,top in enumerate([360,438]):
            ax=ns['axat'](f,h,18+91*col,top+offset,81,64)
            if row==0:
                ax.set_title(method,fontsize=9,pad=5)
                for cell in markers:
                    d=p[p.cell.eq(cell)]
                    ax.scatter(d.UMAP1,d.UMAP2,c=cellcolors[cell],marker=markers[cell],s=10 if cell=='HT29' else 3,alpha=.85 if cell=='HT29' else .5,lw=0,rasterized=False)
            else:
                ax.scatter(p.UMAP1,p.UMAP2,c='#D5DBDF',s=3,alpha=.45,lw=0,rasterized=False)
                selected=p[p.gid.isin(highlight.gid)]
                assert len(selected)==29
                for moa,group in selected.groupby('moa',sort=False):
                    segments=[]
                    for left,right in combinations(group.itertuples(),2):
                        segments.append([(left.UMAP1,left.UMAP2),(right.UMAP1,right.UMAP2)])
                        link_records.append(dict(method=method,moa=moa,gid_a=int(left.gid),gid_b=int(right.gid),x_a=left.UMAP1,y_a=left.UMAP2,x_b=right.UMAP1,y_b=right.UMAP2))
                    if segments:
                        ax.add_collection(LineCollection(segments,colors=colors[moa],linewidths=.85,alpha=.65,zorder=2,rasterized=False))
                for cell in markers:
                    d=p[p.cell.eq(cell)&p.gid.isin(highlight.gid)]
                    ax.scatter(d.UMAP1,d.UMAP2,c=d.moa.map(colors),marker=markers[cell],s=16 if cell=='HT29' else 12,edgecolors='#354550' if cell=='HT29' else 'white',lw=.3,zorder=3)
            if method in viewports:
                ax.set_xlim(viewports[method][0]);ax.set_ylim(viewports[method][1])
            else:
                xy=p[['UMAP1','UMAP2']].to_numpy();pad=np.ptp(xy,axis=0)*.05
                ax.set_xlim(xy[:,0].min()-pad[0],xy[:,0].max()+pad[0]);ax.set_ylim(xy[:,1].min()-pad[1],xy[:,1].max()+pad[1])
            ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal');ax.set_box_aspect(1)

    links=pd.DataFrame(link_records)
    assert links.groupby('method').size().eq(22).all()
    links.to_csv(R/'build/statistics/Fig4_MOA_links.csv',index=False)
