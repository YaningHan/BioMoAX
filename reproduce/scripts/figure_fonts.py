"""Portable fonts; licensed Arial files may be supplied, but are not distributed."""
from pathlib import Path
import os
from matplotlib import font_manager

regular = os.environ.get('BIOMOAX_FONT_REGULAR')
bold = os.environ.get('BIOMOAX_FONT_BOLD')
if bool(regular) != bool(bold):
    raise ValueError('Set both BIOMOAX_FONT_REGULAR and BIOMOAX_FONT_BOLD')
if regular:
    REGULAR_FONT, BOLD_FONT = Path(regular).resolve(), Path(bold).resolve()
    if not REGULAR_FONT.is_file() or not BOLD_FONT.is_file():
        raise FileNotFoundError('Configured font file is missing')
else:
    for family in ['Liberation Sans', 'DejaVu Sans']:
        try:
            REGULAR_FONT = Path(font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False))
            BOLD_FONT = Path(font_manager.findfont(font_manager.FontProperties(family=family, weight='bold'), fallback_to_default=False))
            break
        except ValueError:
            continue

FONT_FAMILY = font_manager.FontProperties(fname=str(REGULAR_FONT)).get_name()

def register_fonts():
    for path in [REGULAR_FONT, BOLD_FONT]:
        font_manager.fontManager.addfont(str(path))
