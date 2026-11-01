from figure_fonts import FONT_FAMILY, REGULAR_FONT, BOLD_FONT, register_fonts
from crop_assets import CropFile
"""Paper-width Fig3–5: common 10pt type, frozen quantitative and spatial results."""
from pathlib import Path
from datetime import datetime, timezone
import io,json,shutil,hashlib,sys
import numpy as np,pandas as pd,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle,ConnectionPatch
from matplotlib.collections import LineCollection
import pymupdf as fitz


R=Path(__file__).resolve().parents[1];C=R/'inputs/current'
register_fonts()
def style():
    plt.rcParams.update({'font.family':FONT_FAMILY,'font.size':10,'axes.titlesize':10,'axes.labelsize':10,
      'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'pdf.fonttype':42,
      'axes.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False})
colors=['#E15967','#56B7B9','#9FCBDB','#FF8956']
def canvas(h):
    style();f=plt.figure(figsize=(6.5,h/72));return f,h
def axat(f,h,x,y,w,hh):return f.add_axes([x/468,(h-y-hh)/h,w/468,hh/h])
def text(f,h,x,y,s,bold=False,color='#202020'):
    f.text(x/468,1-y/h,s,fontsize=11 if bold else 10,weight='bold' if bold else 'normal',color=color)
def heading(f,h,x,y,letter,title):text(f,h,x,y,letter,True);text(f,h,x+18,y,title,True)
def photo(source,row):
    if source=='drug':source='a549'
    paths={'u2os':'crops_cpg0012_690','a549':'crops_cpg0004_a549x1055','batch':'crops_cpg0030_u2os','crispr':'crops_cpg0000_A549'}
    with CropFile(paths[source]) as d:a=d['crops'][row].astype(float)
    rgb=[]
    for ch in [2,1,0]:
        lo,hi=np.percentile(a[ch],[1,99]);rgb.append(np.clip((a[ch]-lo)/max(hi-lo,1),0,1))
    return np.stack(rgb,-1)
def image(f,h,x,y,arr,w=24,hh=24):
    ax=axat(f,h,x,y,w,hh);ax.imshow(arr);ax.axis('off')
def arrow(f,h,x1,x2,y):
    a=axat(f,h,0,0,468,h);a.set(xlim=(0,468),ylim=(h,0));a.axis('off')
    a.annotate('',(x2,y),(x1,y),arrowprops=dict(arrowstyle='simple,head_width=4,head_length=5,tail_width=1.6',mutation_scale=1,color='#8BAE70',lw=0))
def heat(f,h,x,y):
    a=axat(f,h,x,y,18,24)
    from matplotlib.colors import ListedColormap
    a.imshow([[0,0,1,0],[0,1,1,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],cmap=ListedColormap(['#B1CDD4','#D2AD9E']),vmin=0,vmax=1)
    a.set_xticks(np.arange(-.5,4,1),minor=True);a.set_yticks(np.arange(-.5,5,1),minor=True)
    a.grid(which='minor',color='white',lw=.5);a.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    for sp in a.spines.values():sp.set_visible(False)
def query(f,h,n):
    text(f,h,4,12,'A',True)
    for x,label in [(34,'Query'),(208,'Gallery'),(379,'Top-5')]:text(f,h,x,12,label,True)
    rows=([('batch',0,'cpg0030','u2os',0,'cpg0012',3)] if n==3 else [('u2os',410,'U2OS','a549',810,'A549',3),('a549',910,'A549','u2os',510,'U2OS',5)])
    for i,(qs,qr,ql,gs,gr,gl,hit) in enumerate(rows):
        y=20+i*30;multi=n==4
        for x,source,row,label in [(34,qs,qr,ql),(208,gs,gr,gl)]:
            image(f,h,x,y,photo(source,row))
            if multi:heat(f,h,x+28,y)
            text(f,h,x+(50 if multi else 29),y+15,label)
        arrow(f,h,143,194,y+12);arrow(f,h,324,365,y+12)
        for k in range(1,6):
            xx=379+(k-1)*15
            if k==hit:
                figrect=Rectangle((xx/468,(h-y-21)/h),10/468,18/h,transform=f.transFigure,fc='#FAEBEE',ec='none');f.add_artist(figrect)
            text(f,h,xx,y+15,str(k))
def legend(f,h,y,expression=False):
    labs=['BioMoAX','CellProfiler','DeepProfiler']+(['L1000'] if expression else [])
    hs=[Patch(color=c,label=l) for c,l in zip(colors,labs)]+[Line2D([],[],c='#999999',ls='--',lw=.8,label='Random')]
    f.legend(handles=hs,ncol=len(hs),loc='center',bbox_to_anchor=(.52,1-y/h),frameon=False,handlelength=.8,columnspacing=.8,handletextpad=.4)
def method_titles(f,h,y):
    for x,name in [(53,'BioMoAX'),(191,'CellProfiler'),(338,'DeepProfiler')]:text(f,h,x,y,name)
def save(f,n,inserts=()):
    from figure_style_common import normalize,font_receipt
    normalize(f)
    buf=io.BytesIO();f.savefig(buf,format='pdf');plt.close(f)
    doc=fitz.open(stream=buf.getvalue(),filetype='pdf');p=doc[0]
    font_receipt(doc,n)
    doc.save(R/'build/figures'/f'Fig{n}.pdf',garbage=4,deflate=True)
    p.get_pixmap(matrix=fitz.Matrix(2,2)).save(R/'build/figures'/f'Fig{n}.png');doc.close()


curves=pd.read_csv(C/'Fig5_single_case_curve_audit.csv')
# Fig5: the same overview/zoom networks, now drawn at final physical size.
f,h=canvas(420)
heading(f,h,4,12,'A','Retrieving drugs from CRISPR profiles')
# Compact left-to-right query scene, with a proportionate plot to its right.
text(f,h,17,39,'Query',True);text(f,h,97,39,'Reference',True)
image(f,h,23,49,photo('crispr',130026),28,28)
image(f,h,104,49,photo('a549',200144),28,28)
arrow(f,h,57,97,63)
text(f,h,14,94,'CRISPR image');text(f,h,94,94,'Drug images')
table=pd.read_csv(R/'assets/fig5/random_curve.csv')
a=axat(f,h,187,35,264,84)
for method,color in [('BioMoAX',colors[0]),('CellProfiler',colors[1]),('DeepProfiler',colors[2]),('CLOOME','#9271AF'),('CellCLIP','#6D8B50'),('Random','#999999')]:
    if method=='Random':
        q=table[(table.cell=='A549')&(table.method==method)].sort_values('k');x=q.k.to_numpy();y=q.value.to_numpy()
    else:
        q=curves[curves.method.eq(method)].sort_values('k');x=q.k.to_numpy();y=q['mean'].to_numpy()
        a.fill_between(x,q.low.to_numpy(),q.high.to_numpy(),color=color,alpha=.12,lw=0)
    a.plot(x,y,color=color,lw=1.2,ls='--' if method=='Random' else '-',marker=None if method=='Random' else 'o',ms=2)
a.set(xlim=(1,10),ylim=(0,40),xticks=[1,5,10],yticks=[0,10,20,30,40],xlabel='Top K',ylabel='Hit rate (%)')
f.legend(handles=[Line2D([],[],color=c,lw=1.2,ls='--' if m=='Random' else '-',label=m) for m,c in [('BioMoAX',colors[0]),('CellProfiler',colors[1]),('DeepProfiler',colors[2]),('CLOOME','#9271AF'),('CellCLIP','#6D8B50'),('Random','#999999')]],loc='center',bbox_to_anchor=(.51,1-158/h),ncol=6,frameon=False,fontsize=10,handlelength=.9,columnspacing=.65,handletextpad=.35)
from fig5_cases import draw
draw(f,h,globals())
curves.to_csv(R/'build/statistics/Fig5_plotted_curves.csv',index=False)
save(f,5)
