"""Reproduce the current paper from bundled fixed observations and coordinates."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent

def check_inputs():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Changed reproduction input: {item["path"]}')
    print('INPUTS_PASS', len(manifest['files']), flush=True)

def call(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / 'scripts/isolated_run.py'), script, *arguments], cwd=ROOT, check=True)

def figures():
    for number in range(1, 6):
        call(f'render_fig{number}.py', *(['--best-sampling-preview'] if number == 3 else []))

def validate_figures():
    import pymupdf as fitz
    report = []
    for number in range(1, 6):
        document = fitz.open(ROOT / f'build/figures/Fig{number}.pdf')
        page = document[0]
        if abs(page.rect.width - 468) > .01:
            raise ValueError(f'Unexpected Figure {number} width')
        if number in [2, 3, 4]:
            threshold = {2: 300, 3: 220, 4: 300}[number]
            raster = [list(rect) for image in page.get_images(full=True) for rect in page.get_image_rects(image[0]) if rect.y0 > threshold]
            if raster:
                raise ValueError(f'Raster images in Figure {number} scatter region: {raster}')
        report.append(dict(figure=number, width_pt=page.rect.width, height_pt=page.rect.height))
    (ROOT / 'validation/figures.json').write_text(json.dumps(dict(passed=True, figures=report), indent=2) + '\n')
    print('VECTOR_FIGURES_PASS', flush=True)

def paper():
    if not shutil.which('latexmk') or not shutil.which('pdflatex'):
        raise RuntimeError('Paper compilation requires latexmk and pdfLaTeX from a TeX distribution')
    destination = ROOT / 'build/paper'
    shutil.copytree(REPOSITORY / 'paper', destination, dirs_exist_ok=True)
    rendered = ROOT / 'build/figures'
    available = all((rendered / f'Fig{n}.pdf').is_file() for n in range(1, 6))
    if available:
        for number in range(1, 6):
            shutil.copy2(rendered / f'Fig{number}.pdf', destination / f'figures/fig{number}.pdf')
    with (ROOT / 'validation/paper_compile.log').open('w') as log:
        subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], cwd=destination, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('PAPER_PASS', destination / 'main.pdf', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['all', 'figures', 'statistics', 'paper', 'check-inputs', 'validate'], default='all', nargs='?')
    args = parser.parse_args()
    for name in ['build/figures', 'build/statistics', 'validation']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    check_inputs()
    if args.command in ['all', 'figures']:
        figures()
        validate_figures()
    if args.command in ['all', 'statistics']:
        call('verify_statistics.py')
    if args.command in ['all', 'paper']:
        paper()
    if args.command == 'validate':
        validate_figures()
    check_inputs()

if __name__ == '__main__':
    main()
