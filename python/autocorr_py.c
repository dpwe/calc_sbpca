/*
 * autocorr_py.c - calculate normalized auto-correlation
 *    Python extension version
 * 2013-08-26 Dan Ellis dpwe@ee.columbia.edu 
 *   after kslee@ee.columbia.edu, 3/3/2006
 */

/* see http://wiki.scipy.org/Cookbook/C_Extensions/NumPy_arrays */
#include <Python.h>
#include "arrayobject.h" 
#include <math.h>

//#define SQRT(x) sqrt(x)

double SQRT(double x) {
    if (x <= 0) 
	return 1;
    else 
	return sqrt(x);
}

/* 
 * calc_corr
 *
 * Calculates the block-wise autocorrelation of a channel of audio.
 * xp[n] is multiplied by xp[n+eta] and summed up over blocks of winL
 * points, hopped by frmL points, for nfrm points.  eta ranges from 
 * zero th lagL - 1.  ac
 * ac[frame*lagL+eta] = \sum_{n=0}^{winL-1} xp[frame*frmL+n]xp[frame*frmL+n+eta]
 * and to normalize this into a true cosine similarity, 
 * sc[frame*lagL+eta] = sqrt( (\sum_{n=0}^{winL-1} xp[frame*frmL+n]^2)
 *                            * (\sum_{n=0}^{winL-1} xp[frame*frmL+n+eta]^2) )
 * The algorithm actually runs one lag at a time, running through 
 * calculating and summing the products.  To avoid looking at each point 
 * more than once (per lag value), we keep a "running total", and update 
 * it for each frame by adding in the next frmL points, and subtracting 
 * away the sum of the frmL points that have been shifted out of the start 
 * of the window (so the contribution of points in the overlap between two 
 * successive windows remains unchanged in the sum).  Further, to avoid 
 * having to look back at the points from the beginning of the window, 
 * we keep track of the points that are going to fall into each 
 * "subtract away" portion as we see them the first time, then put those 
 * sums into a little ring FIFO buffer, so we can simply read out the 
 * right value to subtract for each block.
 * 
 * 2012-06-24 Dan Ellis dpwe@ee.columbia.edu, based on Kean Sub Lee's original.
 * I added the FIFO to avoid recalculating the portion to subtract away.
 */


static void calc_corr(
		      double xp[], /* Input vector */
		      double **acp, /* Autocorrelation Matrix [nfrms][maxlags] */
		      //double *sc, /* Scaling Factor Matrix for Normalizing AC */
		      int winL,   /* size of window vector */
		      int lagL,   /* size of lag */
		      int frmL,   /* size of a frame */
		      int nfrm,   /* Number of Frames */
		      int normalize /* flag to calc normalized xcorr */
		      )
{
    /* Calculating auto-correlation of rectangular-windowed input. */
    int eta, j, f;

    /*     fprintf(stderr, "calc_corr: xp 0x%lx ac 0x%lx sc 0x%lx wn %d lg %d fr %d nf %d\n", xp, ac, sc, winL, lagL, frmL, nfrm); */
    
    /* how many delays we need in the FIFO? */
    int histlen = (winL+frmL-1)/frmL;
    /* allocate the buffers that hold the past values to 
       subtract later on to keep the running sum in sync */
    double achist[histlen];
    double schist[histlen];
    /* how to divide the blocks of frmL into the parts that go into adjacent 
       elements of the history */
    int frmL1 = winL - (histlen - 1)*frmL;  /* points into older */
    int frmL2 = frmL - frmL1;             /* points into newer */

    /*	fprintf(stderr, "histlen=%d frmlL1=%d frmL2=%d\n", histlen, frmL1, frmL2); */

    double *e = (double*)malloc(nfrm*sizeof(double));

    for (eta=0;eta<lagL;eta++) {

	int hix = 0;

	double z1 = 0.0;
	double z2 = 0.0;

	double s1, s2;

	double *xp1 = xp;
	double *xp2 = xp + eta;

	/* startup: calculate the zero'th window by summing up 
	   the first winL points, and store the partial sums 
	   in each complete frmL subblock to preload the FIFO */
	for(f = 0; f < histlen-1; ++f) {
	    double a = *xp2++;
	    s1 = a * *xp1++; s2 = a*a;  /* initialize to 1st point */
	    for(j = 1; j < frmL; ++j) {
		a = *xp2++;
		s1 += a * *xp1++;
		s2 += a * a;
	    }
	    /* save in the FIFO */
	    achist[hix] = s1;
	    schist[hix] = s2;
	    /* wind on the FIFO */
	    hix = (hix + 1) % histlen;
	    /* accumulate in the actual autocorrelation for this frame */
	    z1 += s1;
	    z2 += s2;
	}
	/* The last part of the first window in general straddles
	   one of the frmL point blocks needed in the fifo, so 
	   calculate it separately, and store this partial result
	   in the fifo */
	double a = *xp2++;
	s1 = a * *xp1++; s2 = a * a;
	for(j = 1; j < frmL1; ++j) {
	    a = *xp2++;
	    s1 += a * *xp1++;
	    s2 += a * a;
	}
	/* finish off adding this to the running accumulators */
	z1 += s1;
	z2 += s2;
	/* .. giving us the full result for the first window */
	if (eta == 0)   e[0] = z2;
	if (eta > 0 && normalize)
	  acp[0][eta] = z1/SQRT(z1*e[0]);
	else
	  acp[0][eta] = z1;
	/* .. but also store as partial sums in the most recent 
	   value in the fifo */
	achist[hix] = s1;
	schist[hix] = s2;
	/* These sums will be completed in the first time through 
	   the loop below. */

	/* OK, all the remaining frames use the regular pattern 
	   of adding the new parts into the accumulators, and 
	   subtracting out the expired parts by reading them 
	   from the FIFO */
	for (f=1;f<nfrm;f++) {
	    /* for each frame, we sum up the next frmL points.
	       But we do them in two blocks, depending on which 
	       FIFO bin they go into */
	    /* first block */
	    double a = *xp2++;
	    s1 = a * *xp1++; s2 = a * a;
	    for (j=1;j<frmL2;j++) {
		a = *xp2++;
		s1 += a * *xp1++;
		s2 += a * a;
	    }
	    /* complete the sums in the current FIFO value */
	    achist[hix] += s1;
	    schist[hix] += s2;
	    /* .. so now we can wind on the FIFO */
	    hix = (hix + 1) % histlen;
	    /* add these new points into our accumulators */
	    z1 += s1;
	    z2 += s2;
	    /* now the second half */
	    a = *xp2++;
	    s1 = a * *xp1++; s2 = a * a;
	    for (j=1;j<frmL1;j++) {
		a = *xp2++;
		s1 += a * *xp1++;
		s2 += a * a;
	    }			
	    /* add these into the window, and take out the 
	       points to remove read from the FIFO at the same time */
	    z1 += s1 - achist[hix];
	    z2 += s2 - schist[hix];
	    /* now we've used those old FIFO points, they can become
	       the newest entries: write in the partial sums */
	    achist[hix] = s1;
	    schist[hix] = s2;
	    if (eta == 0)   e[f] = z2;
	    /* finally, write the actual raw autocorrelation and normalizing 
               constants from the accumulators */
	    if (eta > 0 && normalize)
	      acp[f][eta] = z1/SQRT(e[f]*z2);
	    else
	      acp[f][eta] = z1;
	}
	
    }
    free(e);
}


/* utilities from http://wiki.scipy.org/Cookbook/C_Extensions/NumPy_arrays */

double **ptrvector(long n) { 
   double **v;
   v=(double **)malloc((size_t) (n*sizeof(double)));
   if (!v)   {
      printf("In **ptrvector. Allocation of memory for double array failed.");
      exit(0); }
   return v;
}

double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin) { 
   double **c, *a;
   int i,n,m;

   n=arrayin->dimensions[0];
   m=arrayin->dimensions[1];
   c=ptrvector(n);
   a=(double *) arrayin->data; /* pointer to arrayin data as double */
   for ( i=0; i<n; i++) {
      c[i]=a+i*m; }
   return c;
}

/* ==== Check that PyArrayObject is a double (Float) type and a matrix ==============
    return 1 if an error and raise exception */ 
int  not_doublematrix(PyArrayObject *mat)  {
    if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2)  {
        PyErr_SetString(PyExc_ValueError,
			"In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
        return 1;  }
    return 0;
}

 /* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
    int i,n;
    
    n=arrayin->dimensions[0];
    return (double *) arrayin->data;  /* pointer to arrayin data as double */
}

/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
    return 1 if an error and raise exception */ 
int  not_doublevector(PyArrayObject *vec)  {
    if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
        PyErr_SetString(PyExc_ValueError,
            "In not_doublevector: array must be of type Float and 1 dimensional (n).");
        return 1;  }
    return 0;
}

static PyObject *
autocorr_py_autocorr(PyObject *self, PyObject *args)
{
    PyArrayObject *xpin, *acout, *scout;
    double *xp;
    double **acp;
    int i,j,npts,m, dims[2];
    int frmL, lagL, nfrm, winL, normalize;

    /* parse input args */
    if (!PyArg_ParseTuple(args, "O!iiiii", 
			  &PyArray_Type, &xpin, &frmL, &nfrm, &lagL, &winL, 
			  &normalize))
	return NULL;
//    fprintf(stderr, "xpin=0x%lx frmL=%d lagL=%d nfrm=%d winL=%d\n", 
//	    xpin, frmL, lagL, nfrm, winL);
    
    if (xpin == NULL)  return NULL;

    /* Check that object input is 'double' type and a vector
       Not needed if python wrapper function checks before call to this routine */
    if (not_doublevector(xpin)) return NULL;
     
    /* Get the dimension of the input */
    npts = xpin->dimensions[0];
    
    int lastpt = (nfrm - 1 - 1)*frmL + winL + (frmL - 1) + (lagL - 1);
    if (lastpt >= npts) {
	/* FREAKOUT */
	//char msg[64];
	//sprintf(msg,"autocorr mex: need %d points but only %d passed", 
	//	lastpt+1, npts);
	//mexErrMsgTxt(msg);
	return NULL;
    }

    /* Set up output matrices */
    dims[0] = nfrm;
    dims[1] = lagL;
    /* output of raw autocorrelations */
    acout = (PyArrayObject *)PyArray_FromDims(2, dims, NPY_DOUBLE);
         
    /* Change contiguous arrays into C *arrays   */
    xp = pyvector_to_Carrayptrs(xpin);
    acp = pymatrix_to_Carrayptrs(acout);
     
    /* run calculation */
    calc_corr(xp, acp, winL, lagL, frmL, nfrm, normalize);

    /* return the results */
    return PyArray_Return(acout);
}

/* standard hooks to Python, per http://docs.python.org/2/extending/extending.html */
	
static PyMethodDef AutocorrMethods[] = {
    {"autocorr",  autocorr_py_autocorr, METH_VARARGS},
    {NULL, NULL}        /* Sentinel */
};


/* ==== Initialize the C_test functions ====================== */
// Module name must be _autocorrmodule in compile and linked 
void init_autocorr_py()  {
    (void) Py_InitModule("_autocorr_py", AutocorrMethods);
    import_array();  // Must be present for NumPy.  Called first after above line.
}
