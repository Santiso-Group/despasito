"""
DESPASITO
DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output
"""

import os
import glob
from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize

fpath = os.path.join("despasito", "equations_of_state", "saft", "compiled_modules")
extensions = []

cython_list = glob.glob(os.path.join(fpath, "*.pyx"))
for cyext in cython_list:
    name = os.path.split(cyext)[-1].split(".")[-2]
    cy_ext_1 = Extension(
        name=os.path.join(fpath,name).replace(os.sep,"."), 
        sources=[cyext], 
        include_dirs=[fpath, np.get_include()],
        )
    extensions.extend(
        cythonize(
            [cy_ext_1],
            compiler_directives={
                'language_level': 3,
                'cdivision': False,
                "boundscheck": True
            }))

if __name__ == "__main__":
    setup(
        name='despasito',
        ext_modules=extensions,
    )
