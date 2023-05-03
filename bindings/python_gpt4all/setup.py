from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

llmodel_extension = Extension(
    name="pyllmodel",
    sources=["pyllmodel.pyx"],
    libraries=["llmodel"],
    library_dirs=["../../llmodel/build/"],
    include_dirs=["../../llmodel"],
    extra_link_args=['-Wl,-rpath,../../llmodel/build'],
)
setup(
    name="pyllmodel",
    ext_modules=cythonize([llmodel_extension])
)