from figure_fonts import FONT_FAMILY, REGULAR_FONT, BOLD_FONT, register_fonts
"""Two complete retrieval-network cases at final paper width."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle,ConnectionPatch
R=Path(__file__).resolve().parents[1]

def draw(fig,h,ns):
    def axat(x,y,w,hh):return ns['axat'](fig,h,x,y,w,hh)
    def label(x,y,s,color='#263746',bold=False):
        return fig.text(x/468,1-y/h,s,fontsize=10,color=color,weight='bold' if bold else 'normal',bbox=dict(facecolor='white',edgecolor='none',pad=.25))
    cases=[('B','Receptor kinases',0,'seed45_ep028',['KDR','DDR2','linifanib'],R/'inputs/fig5_network'),
           ('C','Aldehyde metabolism',244,'seed43_ep029',['ALDH2','AKR1B1','disulfiram'],R/'inputs/fig5_case2_network')]
    nodes_out=[];edges_out=[];boxes=[]
    for letter,title,x0,ck,names,S in cases:
        if letter=='B':
            nodes=pd.read_csv(S/'nodes_case1.csv');edges=pd.read_csv(S/'edges.csv').query('checkpoint == @ck');sources=pd.read_csv(S/'displayed_crop_sources.csv').query('checkpoint == @ck')
        else:
            nodes=pd.read_csv(S/'network_nodes.csv');edges=pd.read_csv(S/'network_edges.csv');sources=pd.read_csv(S/'displayed_crop_sources.csv')
        pos={r.node:np.array([r.display_x,r.display_y]) for r in nodes.itertuples()};drug=nodes[nodes.name.eq(names[2])].node.item();focus=['g:'+names[0],'g:'+names[1],drug]
        assert all(((edges.gene==gene)&(edges.key==drug)).any() for gene in names[:2])
        segments=[[pos['g:'+r.gene],pos[r.key]] for r in edges.itertuples()]
        xy=np.array([pos[n] for n in focus]);mid=xy.mean(0);span=max(np.ptp(xy,axis=0).max()*1.9,.12);lo=mid-span/2;hi=mid+span/2
        fig.text((x0+4)/468,1-183/h,letter,fontsize=11,weight='bold',color='#1F2933')
        fig.text((x0+22)/468,1-183/h,title,fontsize=11,weight='bold',color='#1F2933')
        glob=axat(x0+12,274,46,46);glob.set_aspect('equal');glob.axis('off')
        local=axat(x0+78,257,140,132);local.set_aspect('equal');local.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xticks=[],yticks=[])
        for sp in local.spines.values():sp.set_visible(True);sp.set_color('#B9C8D1');sp.set_linewidth(.65)
        for ax,islocal in [(glob,False),(local,True)]:
            ax.add_collection(LineCollection(segments,colors='#BBC8D1',linewidths=.26,alpha=.27,zorder=1))
            for kind,marker,color in [('gene','^','#9EAFBB'),('drug','o','#C1CCD4')]:
                pts=np.array([pos[r.node] for r in nodes.itertuples() if r.kind==kind and (not islocal or r.node not in focus)])
                ax.scatter(pts[:,0],pts[:,1],s=9 if islocal else 3.5,marker=marker,c=color,edgecolors='white',lw=.2,zorder=2)
        glob.autoscale();glob.margins(.12);glob.add_patch(Rectangle(lo,span,span,fill=False,edgecolor='#718997',lw=.6))
        for yy in [lo[1],hi[1]]:fig.add_artist(ConnectionPatch(xyA=(hi[0],yy),coordsA=glob.transData,xyB=(lo[0],yy),coordsB=local.transData,color='#BBCBD5',lw=.5,zorder=0))
        for i in range(2):
            p,q=pos[focus[i]],pos[drug];local.plot([p[0],q[0]],[p[1],q[1]],color='#D35E79' if i==0 else '#B38B4B',ls='-' if i==0 else '--',lw=1.5,zorder=4)
        for i,node in enumerate(focus):
            local.scatter(*pos[node],s=48,marker='^' if i<2 else 'o',color='#426A87' if i<2 else '#D35E79',edgecolors='white',lw=.7,zorder=5)
            local.annotate(str(i+1),pos[node],xytext=[(-11,4),(6,5),(6,-4)][i],textcoords='offset points',fontsize=10,color='#243440',bbox=dict(facecolor='white',edgecolor='none',pad=.35),zorder=6)
            rows=sources[sources.name.eq(names[i])].sort_values('tile') if 'tile' in sources else sources[sources.name.eq(names[i])]
            tiles=[ns['photo']('crispr' if i<2 else 'a549',int(r.h5_row)) for r in rows.itertuples()]
            assert len(tiles)==2
            arr=np.concatenate([tiles[0],np.ones((128,3,3)),tiles[1]],axis=1)
            xx,yy=[(14,219),(112,219),(14,352)][i];ns['image'](fig,h,x0+xx,yy,arr,56,28);label(x0+xx,yy-6,f'{i+1} {names[i]}',color='#426A87' if i<2 else '#C84F70')
        nodes_out.append(dict(panel=letter,checkpoint=ck,nodes=len(nodes),genes=int(nodes.kind.eq('gene').sum()),drugs=int(nodes.kind.eq('drug').sum())))
        edges_out.append(dict(panel=letter,checkpoint=ck,edges=len(edges)))
        boxes.append(dict(panel=letter,bbox=[x0+78,257,x0+218,389]))
    fig.add_artist(Line2D([234/468,234/468],[(h-175)/h,(h-389)/h],transform=fig.transFigure,color='#E2E8ED',lw=.7))
    handles=[Line2D([],[],marker='^',color='#9EAFBB',ls='',label='Gene'),Line2D([],[],marker='o',color='#C1CCD4',ls='',label='Drug'),Line2D([],[],color='#D35E79',lw=1.5,label='Reference-annotated pair'),Line2D([],[],color='#B38B4B',lw=1.5,ls='--',label='Additional co-retrieval')]
    fig.legend(handles=handles,ncol=4,loc='center',bbox_to_anchor=(.5,1-408/h),frameon=False,fontsize=10,handlelength=1.2,columnspacing=.9,handletextpad=.4)
    (R/'build/statistics/Fig5_case_layout.json').write_text(json.dumps(dict(nodes=nodes_out,edges=edges_out,insets=boxes,final_width_pt=468,final_height_pt=h,font_family=FONT_FAMILY,minimum_label_size_pt=10),indent=2))
