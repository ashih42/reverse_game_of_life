echo 'Compiling fast_wrangle...'
python3 setup.py build_ext --inplace
cython fast_wrangle.pyx -a
