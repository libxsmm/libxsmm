#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)

# output directory
if [ "" != "$1" ]; then
  DOCDIR=$1
  shift
else
  DOCDIR=documentation
fi

# temporary file
TMPFILE=$(mktemp fileXXXXXX)
mv ${TMPFILE} ${TMPFILE}.tex

# dump pandoc template for latex, and adjust the template
pandoc -D latex \
| sed \
  -e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
  -e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' > \
  ${TMPFILE}.tex

# cleanup markup and pipe into pandoc using the template
# LIBXSMM documentation
iconv -t utf-8 README.md \
| sed \
  -e 's/\[\[..*\](..*)\]//g' \
  -e 's/\[!\[..*\](..*)\](..*)//g' \
  -e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
  -e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
  -e 's/----*//g' \
| tee >( pandoc \
  --latex-engine=xelatex --template=${TMPFILE}.tex --listings \
  -f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
  -V documentclass=scrartcl \
  -V title-meta="LIBXSMM Documentation" \
  -V author-meta="Hans Pabst, Alexander Heinecke" \
  -V classoption=DIV=45 \
  -V linkcolor=black \
  -V citecolor=black \
  -V urlcolor=black \
  -o ${DOCDIR}/libxsmm.pdf) \
| tee >( pandoc \
  -f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
  -o ${DOCDIR}/libxsmm.html) \
| pandoc \
  -f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
  -o ${DOCDIR}/libxsmm.docx

# cleanup markup and pipe into pandoc using the template
# CP2K recipe
iconv -t utf-8 ${HERE}/documentation/cp2k.md \
| sed \
  -e 's/\[\[..*\](..*)\]//g' \
  -e 's/\[!\[..*\](..*)\](..*)//g' \
  -e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
  -e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
  -e 's/----*//g' \
| tee >( pandoc \
  --latex-engine=xelatex --template=${TMPFILE}.tex --listings \
  -f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
  -V documentclass=scrartcl \
  -V title-meta="CP2K with LIBXSMM" \
  -V author-meta="Hans Pabst" \
  -V classoption=DIV=45 \
  -V linkcolor=black \
  -V citecolor=black \
  -V urlcolor=black \
  -o ${DOCDIR}/cp2k.pdf) \
| pandoc \
  -f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
  -o ${DOCDIR}/cp2k.docx

# remove temporary file
rm ${TMPFILE}.tex
