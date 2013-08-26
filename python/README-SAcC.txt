README for SAcC.py

2013-08-26 Dan Ellis dpwe@ee.columbia.edu

This is a Python port of the SAcC pitch tracker.  With default configuration, 
is almost identical to the compiled Matlab target available at:

http://labrosa.ee.columbia.edu/projects/SAcC/ (version 1.73)

The differences arise from numerical differences, and slight changes in the 
way the HMM decode transition matrix is defined, but should not be material 
in performance.

The Python port (in mlp.py for the neural network, and SAcC.py for the rest 
of the code) is pure Python, whereas the original Matlab target used a 
compiled MEX function for the innermost autocorrelation calculation, which made 
it about twice as fast.  The source to the MEX routine is included as 
autocorr.c; this should be a direct replacement for the my_autocorr function 
in SAcC.py, for anyone who knows how to add a C extension to Python.

The enclosed sacc.txt and sacc.sph are the output of:

  python test_SAcC.py .

The aux/ directory contains the neural net and PCA definition files used, as 
well as several others that could be used with changes to the config array 
used in initialization.
