from distutils.core import Extension, setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import Cython.Compiler.Options
import numpy

Cython.Compiler.Options.cimport_from_pyx = True


def tests():
    return TestLoader().discover('test', pattern='test_*.py')
print('#'*30)
print(numpy.get_include())
setup(
    name="rtta",
    packages=["rtta"],
    version="0.0.1",
    description="Real time incremental Technical Analysis Library in Python",
    long_description="It is a real time incremetnal Technical Analyissi library to perform incremental time series computaitons.  It is built on the numpy library.",
    author="Adam DePrince",
    author_email="adamdeprince@gmail.com",
    maintainer="Adam DePrince",
    maintainer_email="adamdeprince@gmail.com",
    install_requires=[
        "numpy",
        ],
    keywords=['technical analysis', 'realtime', 'real time', 'real time technical analysis', 'numpy'],
    license='GNU AGPLv3',
    classifiers=[
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
    ],
    project_urls = {},
    ext_modules=cythonize(
        ["rtta/trend.pyx",],
        language_level=3,
        nthreads=4,
        include_path=[numpy.get_include()]),
    
    package_data = {
        'rtta': ['*.pxd'],
    },
    
)
