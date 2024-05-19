"""
DESPASITO
DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output
"""
import sys
import os
from setuptools import find_packages, Extension, setup
import versioneer
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

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

setup(
    name="despasito",
    author="Jennifer A Clark",
    author_email="jennifer.clark@gnarlyoak.com",
    description=short_description[0],
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD-3-Clause",
    packages=find_packages(),
    include_package_data=True,
    setup_requires=["numpy", "scipy",] + pytest_runner,
    ext_package=fpath,
    ext_modules=extensions,
    extras_require={
        "extra": ["cython"],
        "tests": ["pytest"],
    },
    install_requires=[
        "numpy",
        "scipy",
        "numba",
    ],
    python_requires=">=3.6, <3.12",
    zip_safe=False,
)
