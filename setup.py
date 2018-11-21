from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[ Extension("fast_wrangle",
	["fast_wrangle.pyx"],
	libraries=["m"],
	extra_compile_args = ["-ffast-math"]) ]

setup(
	name="fast_wrangle",
	cmdclass={"build_ext": build_ext},
	ext_modules=ext_modules)

# from distutils.core import setup
# from Cython.Build import cythonize

# setup(name="fast_wrangle", ext_modules=cythonize('fast_wrangle.pyx'),)
