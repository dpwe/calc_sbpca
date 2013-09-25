CFLAGS=-I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy


# ---- Link --------------------------- 
_autocorr_py.so: autocorr_py.o
	gcc -bundle -flat_namespace -undefined suppress -o _autocorr_py.so autocorr_py.o

# ---- gcc C compile ------------------
autocorr_py.o: autocorr_py.c
	gcc ${CFLAGS} -O3 -c autocorr_py.c

