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
pandoc -D latex | sed \
  -e 's/\(\\documentclass\[.\+\]{.\+}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
  -e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' > \
  ${TMPFILE}.tex

# cleanup markup and pipe into pandoc using the template
# LIBXSMM documentation
sed \
  -e 's/https:\/\/raw\.githubusercontent\.com\/hfp\/libxsmm\/master\///' \
  -e 's/\[!\[.\+\](https:\/\/travis-ci.org\/hfp\/libxsmm.svg?branch=.\+)\](.\+)//' \
  -e 's/\[\[.\+\](.\+)\]//' -e '/!\[.\+\](.\+)/{n;d}' \
  -e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
  -e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
  README.md | tee >( \
pandoc \
  --latex-engine=xelatex --template=${TMPFILE}.tex --listings \
  -f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
  -V documentclass=scrartcl \
  -V title-meta="LIBXSMM Documentation" \
  -V author-meta="Hans Pabst, Alexander Heinecke" \
  -V classoption=DIV=45 \
  -V linkcolor=black \
  -V citecolor=black \
  -V urlcolor=black \
  -o ${DOCDIR}/libxsmm.pdf) | \
pandoc \
  -f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
  -o ${DOCDIR}/libxsmm.docx

# cleanup markup and pipe into pandoc using the template
# CP2K recipe
sed \
  -e 's/https:\/\/raw\.githubusercontent\.com\/hfp\/libxsmm\/master\///' \
  -e 's/\[!\[.\+\](https:\/\/travis-ci.org\/hfp\/libxsmm.svg?branch=.\+)\](.\+)//' \
  -e 's/\[\[.\+\](.\+)\]//' -e '/!\[.\+\](.\+)/{n;d}' \
  -e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
  -e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
  ${HERE}/documentation/cp2k.md | tee >( \
pandoc \
  --latex-engine=xelatex --template=${TMPFILE}.tex --listings \
  -f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
  -V documentclass=scrartcl \
  -V title-meta="CP2K with LIBXSMM" \
  -V author-meta="Hans Pabst" \
  -V classoption=DIV=45 \
  -V linkcolor=black \
  -V citecolor=black \
  -V urlcolor=black \
  -o ${DOCDIR}/cp2k.pdf) | \
pandoc \
  -f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
  -o ${DOCDIR}/cp2k.docx

# remove temporary file
rm ${TMPFILE}.tex
