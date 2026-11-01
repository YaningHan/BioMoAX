"""Shared Fig2-style filled/outlined bars for figure previews."""
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.patches import Patch

def bar_colors(color):
    base=np.asarray(to_rgb(color))
    return base*.48+.52,base*.72

def style_bars(fig):
    for ax in fig.axes:
        current=None
        for container in ax.containers:
            if isinstance(container,BarContainer):
                for patch in container.patches:
                    fill,current=bar_colors(patch.get_facecolor())
                    patch.set_facecolor(fill);patch.set_edgecolor(current);patch.set_linewidth(1.1)
                if container.errorbar is not None:
                    color_error(container.errorbar,current)
            elif isinstance(container,ErrorbarContainer) and current is not None:
                color_error(container,current)

def color_error(container,color):
    _,caps,segments=container.lines
    for line in caps:line.set_color(color)
    for line in segments:line.set_color(color)

def legend_patch(color,label):
    fill,edge=bar_colors(color)
    return Patch(facecolor=fill,edgecolor=edge,linewidth=1.1,label=label)
