"""Alternative layout of the existing Fig2 artists; no statistical changes."""
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

def save_layout_trial(fig, out, colors, statistics):
    plots_a=[a for a in fig.axes if a.get_title() in ('U2OS','A549')]
    plots_b=[a for a in fig.axes if a.get_title() in ('I2I','E2E','I2E','E2I')]
    plots_c=[a for a in fig.axes if a.get_title() in ('BioMoAX','CellProfiler','DeepProfiler','CLOOME','CellCLIP')]
    keep=plots_a+plots_b+plots_c
    image_a=[a.images[0].get_array().copy() for a in fig.axes
             if a not in keep and a.images and a.get_position().y0 > .85]
    image_b=[(a.images[0].get_array().copy(),a.images[0].get_cmap()) for a in fig.axes
             if a not in keep and a.images and a.get_position().y0 < .85]
    for a in list(fig.axes):
        if a not in keep:a.remove()
    for obj in list(fig.texts)+list(fig.legends)+list(fig.artists):obj.remove()
    fig.set_size_inches(11,8.5)
    for j,a in enumerate(plots_a):
        a.set_position([.075,.715-j*.235,.245,.155])
        a.set_ylabel('Hit rate (%)',fontsize=10)
        a.set_xlabel('Top K',fontsize=10)
    for a in plots_b:
        j=['I2I','E2E','I2E','E2I'].index(a.get_title())
        a.set_position([.435+(j%2)*.295,.715-(j//2)*.235,.235,.155])
        a.set_ylabel('Top-5 hit rate (%)' if j%2==0 else '',fontsize=10)
    for j,a in enumerate(plots_c):
        a.set_position([.035+j*.195,.075,.175,.25])
    def heading(x,y,letter,title):
        fig.text(x,y,letter,fontsize=14,weight='bold')
        fig.text(x+.027,y,title,fontsize=12,weight='bold')
    heading(.035,.965,'A','Compound identity retrieval')
    heading(.395,.965,'B','Shared-MOA retrieval')
    def icon(x,data,cmap=None):
        ax=fig.add_axes([x,.900,.030,.039])
        ax.imshow(data,cmap=cmap)
        if data.ndim==2:
            ax.set_xticks(np.arange(-.5,data.shape[1],1),minor=True)
            ax.set_yticks(np.arange(-.5,data.shape[0],1),minor=True)
            ax.grid(which='minor',color='white',lw=.5)
            ax.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
            for s in ax.spines.values():s.set_visible(False)
        else:ax.axis('off')
    fig.text(.10,.945,'Query',ha='center',fontsize=9,weight='bold')
    fig.text(.275,.945,'Reference',ha='center',fontsize=9,weight='bold')
    icon(.085,image_a[0]);icon(.260,image_a[1])
    fig.text(.525,.945,'Query',ha='center',fontsize=9,weight='bold')
    fig.text(.850,.945,'Reference',ha='center',fontsize=9,weight='bold')
    for x,(data,cmap) in zip([.490,.530,.815,.855],image_b):icon(x,data,cmap)
    for start,end in [(.13,.25),(.585,.785)]:
        fig.add_artist(FancyArrowPatch((start,.919),(end,.919),transform=fig.transFigure,
            arrowstyle='->',mutation_scale=11,lw=1.2,color='#587D96'))
    heading(.035,.363,'C','MOA neighborhoods')
    fig.text(.5,.405,'',fontsize=9)
    handles=[Line2D([],[],color=colors[m],lw=2,marker='o',ms=3,label=m)
             for m in ['BioMoAX','CellProfiler','DeepProfiler','CLOOME','CellCLIP']]
    handles += [Line2D([],[],color='#FF8956',lw=2,label='L1000'),
                Line2D([],[],color='#888888',ls='--',label='Random')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.52,.405),ncol=7,
               frameon=False,fontsize=9,columnspacing=1.1,handlelength=1.4)
    fig.add_artist(Line2D([.365,.365],[.47,.95],transform=fig.transFigure,color='#D6E0E6',lw=.7))
    from figure_style_common import finalize_fig2,font_receipt
    finalize_fig2(fig,statistics)
    path=out/'Fig2.pdf'
    fig.savefig(path)
    import pymupdf as fitz
    with fitz.open(path) as doc:font_receipt(doc,2)
    fig.savefig(path.with_suffix('.png'),dpi=160)
    print(path)
