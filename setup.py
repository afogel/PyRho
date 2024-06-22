from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    
    The purpose of this class is to postpone importing pybind11 until it is actually
    installed, so that the `get_include()` method can be invoked. This is necessary
    because we can't import `pybind11` before it is installed.
    """
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'rhoR',
        sources=['src/rhoR.cpp'],  # Replace with the correct path to your cpp file
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            # Path to RcppArmadillo headers (You may need to update this path)
            '/usr/local/include/armadillo',
        ],
        language='c++',
        extra_compile_args=['-std=c++11'],
        libraries=['armadillo'],  # Link against the Armadillo library
    ),
]

setup(
    name='ShaffersRho',
    version='0.1',
    author='Ariel Fogel',
    author_email='ariel@pillar.security',
    description='A package for calculating ShaffersRho using Python and C++. Original code by Cody Marquardt, Shaffer et al. (2016).',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    install_requires=[
        'numpy',
        'pybind11>=2.2',
        'setuptools',
    ],
    cmdclass={'build_ext': build_ext},
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
