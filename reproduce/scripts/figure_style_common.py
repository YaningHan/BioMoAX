from figure_fonts import FONT_FAMILY, REGULAR_FONT, BOLD_FONT, register_fonts
"""Shared typography at final 468-pt manuscript width; vector output only."""
from pathlib import Path
from collections import Counter
import io,json
import numpy as np
import pandas as pd
import pymupdf as fitz

R=Path(__file__).resolve().parents[1]
WIDTH=468.
BODY=9.
DETAIL=8.5
TITLE=10.

def normalize(fig, scale=1.):
    from matplotlib.text import Text
    for t in fig.findobj(match=Text):
        if not t.get_text():continue
        size=BODY
        if t in fig.texts:
            s=t.get_text().strip()
            if s in ['Query','Reference']:size=BODY
            elif t.get_fontsize()>=11 or t.get_fontweight() in ['bold',700] or s in ['A','B','C'] or s.startswith(('A   ','B   ','C   ')):size=TITLE
            elif t.get_fontsize()<9:size=DETAIL
        t.set_fontfamily(FONT_FAMILY);t.set_fontsize(size*scale)
    for leg in fig.legends:
        for t in leg.get_texts():t.set_fontsize(DETAIL*scale)
    for ax in fig.axes:
        ax.title.set_fontsize(BODY*scale)
        ax.xaxis.label.set_fontsize(BODY*scale);ax.yaxis.label.set_fontsize(BODY*scale)
        for t in ax.get_xticklabels()+ax.get_yticklabels():t.set_fontsize(BODY*scale)

def reflow_vertical_stars(ax, peaks, annotations):
    """Allocate statistical brackets in physical points, independent of y units."""
    height=ax.figure.get_figheight()*72*ax.get_position().height
    counts=Counter(group for line,label,group in annotations)
    reserves={group:6 if counts.get(group,0)==0 else 18+12*(counts[group]-1) for group in peaks}
    upper=max(v*height/max(height-reserves[group],height*.18) for group,v in peaks.items())
    upper=max(10,5*np.ceil(upper/5));ax.set_ylim(0,upper)
    step=20 if upper>65 else 10;ax.set_yticks(np.arange(0,upper+.1,step))
    levels=Counter()
    for line,label,group in annotations:
        lev=levels[group];levels[group]+=1
        y=peaks[group]+(6+lev*12)*upper/height
        line.set_ydata([y-2*upper/height,y,y,y-2*upper/height])
        label.set_y(y+5*upper/height);label.set_va('center');label.set_fontsize(BODY)

def finalize_fig2(fig, stats):
    fig.set_size_inches(WIDTH/72,408/72)
    normalize(fig)
    groups={'U2OS':(0,'A'),'A549':(1,'A'),'I2I':(0,'B'),'E2E':(1,'B'),'I2E':(2,'B'),'E2I':(3,'B')}
    method_names=['BioMoAX','CellProfiler','DeepProfiler','CLOOME','CellCLIP']
    for ax in fig.axes:
        title=ax.get_title()
        if title in groups:
            j,group=groups[title]
            top=78+(j if group=='A' else j//2)*108
            left=.09 if group=='A' else .435+(j%2)*.295
            width=.245 if group=='A' else .235
            ax.set_position([left,(408-top-60)/408,width,60/408])
            ax.set_title(title,fontsize=BODY,pad=5);ax.xaxis.labelpad=3
        elif title in method_names:
            j=method_names.index(title)
            ax.set_position([.035+j*.195,(408-332-64)/408,.175,64/408])
            ax.set_box_aspect(1);ax.set_title(title,fontsize=BODY,pad=5)
        elif ax.images:
            pos=ax.get_position();ax.set_position([pos.x0,(408-38-16)/408,.030,16/408])
    for t in fig.texts:
        x,y=t.get_position();label=t.get_text().strip()
        top=29 if label in ['Query','Reference'] else 302 if label in ['C','MOA neighborhoods'] else 12
        t.set_position((x,1-top/408))
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D
    for artist in fig.artists:
        if isinstance(artist,FancyArrowPatch):
            a,b=artist._posA_posB;artist.set_positions((a[0],1-46/408),(b[0],1-46/408))
        elif isinstance(artist,Line2D):artist.set_ydata([1-270/408,1-32/408])
    for leg in fig.legends:leg.set_bbox_to_anchor((.50,1-282/408))
    for ax in fig.axes:
        if ax.get_title() not in ['I2I','E2E','I2E','E2I']:continue
        direction=ax.get_title().lower()
        lines=[l for l in ax.lines if len(l.get_xdata())==4 and len(l.get_ydata())==4]
        labels=[t for t in ax.texts if t.get_text().strip() in ['*','**','***','ns']]
        assert len(lines)==len(labels)
        peaks={j:float(stats[(stats.direction==direction)&(stats.cell==cell)].ci_high.max()) for j,cell in enumerate(['U2OS','A549'])}
        annotations=[(line,label,int(round(np.mean([min(line.get_xdata()),max(line.get_xdata())])))) for line,label in zip(lines,labels)]
        if annotations:reflow_vertical_stars(ax,peaks,annotations)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    plots={a.get_title():a for a in fig.axes if a.get_title() in groups}
    assert not plots['U2OS'].xaxis.label.get_window_extent(renderer).overlaps(plots['A549'].title.get_window_extent(renderer))
    for label in ['I2I','E2E']:
        box=plots[label].title.get_window_extent(renderer)
        assert not any(box.overlaps(a.get_window_extent(renderer)) for a in fig.axes if a.images)

def normalize_fig1(doc):
    """Reletter imported vector schematics; retain mathematical subscripts."""
    p=doc[0]
    spans=[s for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'].strip()]
    regular=REGULAR_FONT;bold=BOLD_FONT
    fonts={False:fitz.Font(fontfile=str(regular)),True:fitz.Font(fontfile=str(bold))}
    updates=[]
    for s in spans:
        if s['size']<6.7 or ('Italic' in s['font'] and s['text'].strip()=='L'):continue
        text=s['text'].replace('\xa0',' ').replace('\xad','-')
        is_bold='bold' in s['font'].lower()
        size=TITLE if s['size']>=9.8 else DETAIL if s['bbox'][1]>245 and s['size']<8 else BODY
        b=fitz.Rect(s['bbox']);width=fonts[is_bold].text_length(text,fontsize=size)
        x=b.x0 if s['size']>=9.8 or (s['bbox'][0]>=370 and s['bbox'][1]<210) else (b.x0+b.x1-width)/2
        x=max(4,min(x,WIDTH-4-width))
        p.add_redact_annot(b,fill=False)
        updates.append((text,(x,s['origin'][1]),size,is_bold,s['color']))
    p.apply_redactions(images=0,graphics=0)
    p.insert_font(fontname='UniformArial',fontfile=str(regular));p.insert_font(fontname='UniformArialBold',fontfile=str(bold))
    for text,pos,size,bold,color in updates:
        rgb=tuple(((color>>shift)&255)/255 for shift in [16,8,0])
        p.insert_text(pos,text,fontsize=size,fontname='UniformArialBold' if bold else 'UniformArial',color=rgb)

def scale_to_width(doc,width=WIDTH):
    out=fitz.open();p=out.new_page(width=width,height=doc[0].rect.height*width/doc[0].rect.width)
    p.show_pdf_page(p.rect,doc,0)
    return out

def font_receipt(doc,number):
    p=doc[0]
    spans=[s for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'].strip()]
    record=dict(figure=number,width_pt=p.rect.width,height_pt=p.rect.height,
                font_sizes=sorted(set(round(s['size'],2) for s in spans)),fonts=sorted(set(s['font'] for s in spans)))
    (R/f'build/statistics/Fig{number}_typography.json').write_text(json.dumps(record,indent=2))
