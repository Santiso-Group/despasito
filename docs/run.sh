#! /bin/bash

#conda install sphinx sphinx-argparse sphinx_rtd_theme

rm -rf _build
rm -rf _autosummary

make clean html
open _build/html/index.html
