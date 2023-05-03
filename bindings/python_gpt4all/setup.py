from distutils.core import setup
from distutils.extension import Extension
from setuptools import setup, find_packages

from Cython.Build import cythonize

def read_requirements():
    with open('requirements.txt') as req:
        return [line.strip() for line in req.readlines() if line.strip() and not line.startswith('#')]

llmodel_extension = Extension(
    name="pyllmodel",
    sources=["pyllmodel.pyx"],
    libraries=["llmodel"],
    library_dirs=["../../llmodel/build/"],
    include_dirs=["../../llmodel"],
    extra_link_args=['-Wl,-rpath,../../llmodel/build'],
)
setup(
    name="python_gpt4all",
    version="1.0",
    description="Python bindings for GPT4All",
    author="Richard Guo",
    author_email="richard@nomic.ai",
    ext_modules=cythonize([llmodel_extension]),
    packages=find_packages(),
    install_requires=read_requirements(),
)