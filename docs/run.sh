#! /bin/bash

rm -rf _build
rm -rf _autosummary

make clean html
open _build/html/index.html
