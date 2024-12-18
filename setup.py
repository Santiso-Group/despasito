"""
DESPASITO
DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output
"""

import os
from setuptools import Extension, setup
import glob
import numpy as np

short_description = __doc__.split("\n")
fpath = os.path.join("despasito", "equations_of_state", "saft", "compiled_modules")
extensions = []

try:
    from Cython.Build import cythonize
    flag_cython = True
except Exception:
    print(
        'Cython not available on your system. Dependencies will be run with numba.'
    )
    flag_cython = False

if flag_cython:
    cython_list = glob.glob(os.path.join(fpath, "*.pyx"))
    for cyext in cython_list:
        name = os.path.split(cyext)[-1].split(".")[-2]
        cy_ext_1 = Extension(
            name=name, 
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

setup(ext_modules=extensions)
