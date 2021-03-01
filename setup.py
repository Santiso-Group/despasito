"""
DESPASITO
DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output
"""
import sys
import os
from setuptools import find_packages
import versioneer
from numpy.distutils.core import Extension, setup
from numpy.distutils.fcompiler import get_default_fcompiler
import numpy as np
import glob

short_description = __doc__.split("\n")
fpath = os.path.join("despasito", "equations_of_state", "saft", "compiled_modules")
extensions = []

if sys.version_info.minor > 8:
    raise ValueError(
        "DESPASITO cannot run on python versions greater than 3.8 due to incompadibilities between python 3.9 and numba."
    )

try:
    from Cython.Build import cythonize
    flag_cython = True
except:
    print('Cython not available on your system. Proceeding without C-extentions.')
    flag_cython = False

if flag_cython:
    cython_list = glob.glob(os.path.join(fpath,"*.pyx"))
    for cyext in cython_list:
        name = os.path.split(cyext)[-1].split(".")[-2]
        cy_ext_1 = Extension(name=name, sources=[cyext], include_dirs=[fpath])
        extensions.extend(cythonize([cy_ext_1],compiler_directives={'language_level': 3}))

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])

if get_default_fcompiler() != None:
    fortran_list = glob.glob(os.path.join(fpath, "*.f90"))
    for fext in fortran_list:
        name = os.path.split(fext)[-1].split(".")[-2]
        ext1 = Extension(name=name, sources=[fext], include_dirs=[fpath])
        extensions.append(ext1)
else:
    print("Fortran compiler is not found, default will use numba")

# try Extension and compile
# !!!! Note that we have fortran modules that need to be compiled with "f2py3 -m solv_assoc -c solve_assoc.f90" and the same with solve_assoc_matrix.f90

setup(
    # Self-descriptive entries which should always be present
    name="despasito",
    author="Jennifer A Clark",
    author_email="jennifer.clark@gnarlyoak.com",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD-3-Clause",
    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),
    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    # Allows `setup.py test` to work correctly with pytest
    setup_requires=["numpy", "scipy"] + pytest_runner,
    ext_package=fpath,
    ext_modules=extensions,
    extras_require={"extra": ["pytest", "numba", "cython"]},
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "cython",
    ],  # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    python_requires=">=3.6, <=3.8.8",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    zip_safe=False,
)
