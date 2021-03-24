#! /bin/bash

#conda install -c conda-forge sphinx sphinx-argparse sphinx_rtd_theme m2r2

rm -rf _build
rm -rf _autosummary

make clean html
open _build/html/index.html
