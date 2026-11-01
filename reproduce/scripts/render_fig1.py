from figure_fonts import FONT_FAMILY, REGULAR_FONT, BOLD_FONT, register_fonts
from pathlib import Path
import pymupdf as fitz
R=Path(__file__).resolve().parents[1]
a=fitz.open(R/'inputs/fig1_panels/Fig1_panel_A.pdf')
b=fitz.open(R/'inputs/fig1_panels/Fig1_panel_B_common_inputs.pdf')
# Explicit grouping for all three MOA retrieval contexts. Keep the source panel
# untouched; perform the vector-only annotation edit at render time.
panel=b[0]
labels=[s for block in panel.get_text('dict')['blocks'] for line in block.get('lines',[]) for s in line['spans'] if 'Shared' in s['text'] and 'MOA' in s['text']]
assert len(labels)==1
panel.add_redact_annot(fitz.Rect(labels[0]['bbox']),fill=(1,1,1))
panel.apply_redactions(images=0,graphics=0)
# Clear the former faint rule and the lower ends of the internal separators.
panel.draw_rect(fitz.Rect(216,330,834,372),color=None,fill=(1,1,1),overlay=True)
color=(.25,.43,.54)
shape=panel.new_shape()
shape.draw_polyline([fitz.Point(223,331),fitz.Point(223,341),fitz.Point(827,341),fitz.Point(827,331)])
shape.draw_line(fitz.Point(525,341),fitz.Point(525,348))
shape.finish(color=color,width=1.8);shape.commit()
label='Shared MOA';size=18
x=525-fitz.get_text_length(label,fontname='hebo',fontsize=size)/2
panel.insert_text((x,366),label,fontname='hebo',fontsize=size,color=color,overlay=True)

w=468.;gap=8
ha=a[0].rect.height*w/a[0].rect.width;hb=b[0].rect.height*w/b[0].rect.width
d=fitz.open();p=d.new_page(width=w,height=ha+gap+hb)
p.show_pdf_page(fitz.Rect(0,0,w,ha),a,0)
p.draw_line((5,ha+gap/2),(w-5,ha+gap/2),color=(.81,.85,.87),width=.4)
p.show_pdf_page(fitz.Rect(0,ha+gap,w,ha+gap+hb),b,0)
from figure_style_common import normalize_fig1,font_receipt
normalize_fig1(d);font_receipt(d,1)
d.save(R/'build/figures/Fig1.pdf',garbage=4,deflate=True)
