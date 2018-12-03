echo 'Compiling cy_wrangle ...'
python3 setup.py build_ext --inplace
cython cy_wrangle.pyx -a
