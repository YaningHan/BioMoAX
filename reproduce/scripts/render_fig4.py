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
from bar_style import style_bars,legend_patch
from fig4_cloome_spatial import draw as draw_spatial
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
    f.text(x/468,1-y/h,s,fontsize=10,weight='bold' if bold else 'normal',color=color)
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
    style_bars(f)
    from figure_style_common import normalize,reflow_vertical_stars,font_receipt
    normalize(f)
    for ax in {a for a,line,label in annotation_objects if a is not ht29_axis}:
        direction=ax.get_title().lower()
        peaks={j:max(v['high'] for v in cache[(task,direction)]) for j,task in enumerate(['U2OS → A549','A549 → U2OS'])}
        records=[(line,label,int(np.mean([min(line.get_xdata()),max(line.get_xdata())])>45)) for a,line,label in annotation_objects if a is ax]
        reflow_vertical_stars(ax,peaks,records)
    f.canvas.draw();renderer=f.canvas.get_renderer();scale=72/f.dpi;checks=[]
    for i,(ax,line,label) in enumerate(annotation_objects):
        box=label.get_window_extent(renderer);axis=ax.get_window_extent(renderer)
        assert box.y1<=axis.y1+.1 and box.x0>=axis.x0-.1 and box.x1<=axis.x1+.1,(i,'outside axis')
        for j,(other_ax,other_line,other_label) in enumerate(annotation_objects):
            if other_ax is not ax:continue
            path=other_line.get_path().transformed(other_line.get_transform()).get_extents()
            assert not box.overlaps(path),(i,j,'star/bracket overlap')
            if j!=i:assert not box.overlaps(other_label.get_window_extent(renderer)),(i,j,'star/star overlap')
        checks.append(dict(index=i,text=label.get_text(),star_bbox_points=[float(x*scale) for x in box.extents]))
    (R/'validation/fig4_annotation_spacing.json').write_text(json.dumps(dict(passed=True,level_spacing_points=12,font_size_points=9,annotations=checks),indent=2))
    label_box=query_direction_label.get_window_extent(renderer)
    center_error=abs((label_box.x0+label_box.x1)/2/f.dpi*72-228)
    assert center_error<.1
    axis_box=ht29_axis.get_window_extent(renderer)
    assert abs(axis_box.height/f.dpi*72-68)<.1 and abs(axis_box.width/f.dpi*72-411)<.1
    (R/'validation/fig4_panel_layout.json').write_text(json.dumps(dict(passed=True,query_label_center_error_points=center_error,ht29_axis_width_points=411,ht29_axis_height_points=68),indent=2))
    buf=io.BytesIO();f.savefig(buf,format='pdf');plt.close(f)
    doc=fitz.open(stream=buf.getvalue(),filetype='pdf');p=doc[0]
    font_receipt(doc,4)
    doc.save(R/'build/figures/Fig4.pdf',garbage=4,deflate=True)
    p.get_pixmap(matrix=fitz.Matrix(2,2)).save(R/'build/figures/Fig4.png');doc.close()


ci_table=pd.read_csv(C/'fig4_query_ci/summary.csv')
ci_tests=pd.read_csv(C/'fig4_query_ci/tests.csv');ci_tests=ci_tests[ci_tests['head'].eq('moa')]
frozen_bars=pd.read_csv(R/'assets/fig4/frozen_bars.csv')
random_references=pd.read_csv(R/'assets/fig4/random_references.csv')
def update_intervals(rows,task,direction):
    names={'i2i':['BioMoAX','CellProfiler','DeepProfiler'],'e2e':['BioMoAX','Raw L1000'],'i2e':['BioMoAX','CP + Ridge','DP + Ridge'],'e2i':['BioMoAX','CP + Ridge','DP + Ridge']}[direction]
    for row,method in zip(rows,names):
        table=ci_table;v=table[(table.task==task)&(table.direction==direction)&(table['head']=='moa')&(table.method==method)].iloc[0]
        assert abs(row['value']-v.value)<.001,(task,direction,method,row['value'],v.value)
        row.update(low=float(v.low),high=float(v.high))
    if task in ['U2OS → A549','A549 → U2OS','HT29 → U2OS+A549'] and direction!='e2e':
        method='CLOOME' if direction=='i2i' else 'CLOOME + Ridge'
        table=ci_table;extra=table[(table.task==task)&(table.direction==direction)&(table['head']=='moa')&(table.method==method)]
        if len(extra):
            v=extra.iloc[0]
            rows.append(dict(value=float(v.value),low=float(v.low),high=float(v.high),method=method))
        extra=table[(table.task==task)&(table.direction==direction)&(table['head']=='moa')&(table.method=='CellCLIP')]
        if len(extra):
            v=extra.iloc[0]
            rows.append(dict(value=float(v.value),low=float(v.low),high=float(v.high),method='CellCLIP'))

# Fig4: B and C share a quantitative row; D and E retain matched viewports.
f,h=canvas(510)
annotation_objects=[]
heading(f,h,4,12,'A','Cross-cell-line retrieval')
text(f,h,72,43,'U2OS (U)');image(f,h,133,28,photo('u2os',410));heat(f,h,161,28)
arrow(f,h,191,265,40);arrow(f,h,265,191,47)
query_direction_label=f.text(228/468,1-25/h,'Query → Reference',fontsize=10,color='#202020',ha='center')
image(f,h,279,28,photo('a549',810));heat(f,h,307,28);text(f,h,333,43,'A549 (A)')
# Recover the frozen bars including random references and significance marks.
cache={}
for (task,direction),g in frozen_bars.groupby(['task','direction'],sort=False):
    rr=g.to_dict('records');update_intervals(rr,task,direction);cache[(task,direction)]=rr
for col,(direction,title) in enumerate([('i2i','I2I'),('e2e','E2E'),('i2e','I2E'),('e2i','E2I')]):
    a=axat(f,h,40+108*col,72,92,90)
    peak=max(v['high'] for task in ['U2OS → A549','A549 → U2OS'] for v in cache[(task,direction)])
    upper=5*np.ceil((peak*138/60)/5)
    a.set(xlim=(0,90),ylim=(0,upper),yticks=np.arange(0,upper+.1,10));a.set_title(title,pad=5,linespacing=1)
    units_per_point=upper/138
    if col==0:a.set_ylabel('Top-5 hit rate (%)')
    for center,task in zip([22.5,67.5],['U2OS → A549','A549 → U2OS']):
        rows=cache[(task,direction)];xx=center+(np.arange(len(rows))-(len(rows)-1)/2)*7.5
        for k,(x,r) in enumerate(zip(xx,rows)):
            a.bar(x,r['value'],6,color=('#6D8B50' if k==4 else '#9271AF' if k==3 else colors[3] if direction=='e2e' and k==1 else colors[k]))
            a.errorbar(x,r['value'],yerr=[[r['value']-r['low']],[r['high']-r['value']]],fmt='none',ecolor='#354550',elinewidth=.85,capsize=2)
        # Random reference is taken from the original displayed vector line.
        chance=random_references[(random_references.task==task)&(random_references.direction==direction)].value.iloc[0]
        a.plot([xx[0]-5,xx[-1]+5],[chance]*2,c='#999999',ls='--',lw=.75)
        sig=ci_tests[(ci_tests.task==task)&(ci_tests.direction==direction)&(ci_tests.p_holm_figure<.05)]
        for lev,r in enumerate(sig.sort_values('baseline').itertuples()):
            k=4 if r.baseline=='CellCLIP' else 3 if r.baseline.startswith('CLOOME') else 1 if r.baseline in ['CP + Ridge','CellProfiler','Raw L1000'] else 2
            y=max(v['high'] for v in rows)+(8+lev*16)*units_per_point
            bracket,=a.plot([xx[0],xx[0],xx[k],xx[k]],[y-2*units_per_point,y,y,y-2*units_per_point],lw=.55,c='#354550')
            label=a.text((xx[0]+xx[k])/2,y+2*units_per_point,r.stars_figure,ha='center',va='bottom',fontsize=10)
            annotation_objects.append((a,bracket,label))
    a.set_xticks([22.5,67.5],['U → A','A → U'],fontsize=9)
f.legend(handles=[legend_patch(c,m) for m,c in [('BioMoAX',colors[0]),('CellProfiler',colors[1]),('DeepProfiler',colors[2]),('L1000',colors[3]),('CLOOME','#9271AF'),('CellCLIP','#6D8B50')]]+[Line2D([],[],color='#999999',ls='--',label='Random')],ncol=7,loc='center',bbox_to_anchor=(.51,1-192/h),frameon=False,fontsize=8,handlelength=.9,columnspacing=.8)
heading(f,h,4,214,'B','Retrieving drugs from unseen HT29 cells')
ht29_random=[random_references[(random_references.task=='HT29 → U2OS+A549')&(random_references.direction==d)].value.iloc[0] for d in ['i2i','i2e']]
rows=frozen_bars[frozen_bars.task.eq('HT29 → U2OS+A549')].to_dict('records');a=axat(f,h,40,230,411,68);ht29_axis=a;a.set(xlim=(0,40),ylim=(-.6,1.6),xticks=[0,20,40],yticks=[1,0],yticklabels=['I2I','I2E'],xlabel='Top-5 hit rate (%)')
for group,direction in enumerate(['i2i','i2e']):
    rr=rows[group*3:group*3+3];update_intervals(rr,'HT29 → U2OS+A549',direction);yy=1-group+(np.arange(len(rr))-(len(rr)-1)/2)*.19
    a.plot([ht29_random[group]]*2,[yy.min()-.12,yy.max()+.12],c='#999999',ls='--',lw=.75)
    for k,(y,r) in enumerate(zip(yy,rr)):
        a.barh(y,r['value'],.16,color='#6D8B50' if k==4 else '#9271AF' if k==3 else colors[k]);a.errorbar(r['value'],y,xerr=[[r['value']-r['low']],[r['high']-r['value']]],fmt='none',ecolor='#354550',elinewidth=.7,capsize=1.5)
    sig=ci_tests[(ci_tests.task=='HT29 → U2OS+A549')&(ci_tests.direction==direction)&(ci_tests.p_holm_figure<.05)]
    for lev,r in enumerate(sig.sort_values('baseline').itertuples()):
        k=4 if r.baseline=='CellCLIP' else 3 if r.baseline.startswith('CLOOME') else 1 if r.baseline in ['CP + Ridge','CellProfiler','Raw L1000'] else 2;x=max(v['high'] for v in rr)+3+lev*4
        bracket,=a.plot([x-1,x,x,x-1],[yy[0],yy[0],yy[k],yy[k]],lw=.55,c='#354550')
        label=a.text(x+.5,(yy[0]+yy[k])/2,r.stars_figure,fontsize=10,va='center')
        annotation_objects.append((a,bracket,label))
# Keep all horizontal significance brackets and stars inside the page.
right=max([float(max(line.get_xdata())) for line in a.lines]+[float(t.get_position()[0])+3 for t in a.texts])
a.set_xlim(0,5*np.ceil((right+1)/5));a.set_xticks(np.arange(0,a.get_xlim()[1]+.1,10))
draw_spatial(f,h,globals(),offset=0)
inserts=[]
save(f,4,inserts)
