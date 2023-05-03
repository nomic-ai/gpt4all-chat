from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize

llmodel_extension = Extension(
    name="gpt4all.pyllmodel",
    sources=["gpt4all/pyllmodel.pyx"],
    libraries=["llmodel"],
    library_dirs=["../../llmodel/build/"],
    include_dirs=["../../llmodel"],
    extra_link_args=['-Wl,-rpath,../../llmodel/build'],
)

ext_modules = cythonize([llmodel_extension])

setup(
    name="gpt4all",
    version="0.1.0",
    description="Python bindings for GPT4All",
    author="Richard Guo",
    author_email="richard@nomic.ai",
    packages=find_packages(),
    ext_modules=ext_modules
)