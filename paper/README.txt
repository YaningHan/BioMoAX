BioMoAX - Overleaf source package

Main document: main.tex
Compiler: pdfLaTeX
Bibliography: BibTeX using the included splncs04.bst

Upload this ZIP as an existing project in Overleaf. The main.tex file is at the ZIP root.
Sections are in sections/. Figures are in figures/ as fig1.pdf through fig5.pdf.
The user-provided LNCS class and bibliography style are included without modification.
The title/abstract/code page is separate; the main text occupies 10 pages.
References and the appendix follow the main text. The complete PDF has 14 pages.

This package contains the English manuscript only. No training data, checkpoints,
history directories, temporary figures or compilation caches are required.
All project file dependencies use paths inside this package.

Local command: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
