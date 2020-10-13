#!/bin/bash
#
# Compile and build Markdown paper into PDF
# -------------------------------------------------------------------------------------------------

INPUT_TXT="paper.md"
INPUT_BIB="paper.bib"
OUTPUT_PDF="paper.pdf"
ENGINE="xelatex"
# OPTS="-V geometry:margin=1in --variable classoption=twocolumn"
OPTS="-V geometry:margin=1in"

# pandoc --filter pandoc-citeproc --bibliography=paper.bib --pdf-engine=xelatex -s paper.md -o paper.pdf
pandoc --filter pandoc-citeproc ${OPTS} --bibliography=${INPUT_BIB} --pdf-engine=${ENGINE} -s ${INPUT_TXT} -o ${OUTPUT_PDF}
