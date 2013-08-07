/*
 * autocorr.c - calculate normalized auto-correlation
 *
 * kslee@ee.columbia.edu, 3/3/2006
 */

#include "mex.h"
#include <math.h>

#define SQRT(x) sqrt(x)

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
		      double *ac, /* Autocorrelation Matrix */
		      double *sc, /* Scaling Factor Matrix for Normalizing AC */
		      int winL,   /* size of window vector */
		      int lagL,   /* size of lag */
		      int frmL,   /* size of a frame */
		      int nfrm    /* Number of Frames */
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
	ac[eta] = z1;
	sc[eta] = SQRT(ac[0]*(z2));
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
	    /* finally, write the actual raw autocorrelation and normalizing 
               constants from the accumulators */
	    ac[f*lagL+eta] = z1;
	    sc[f*lagL+eta] = SQRT(ac[f*lagL]*z2);
	}
	
    }
    
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int winL, lagL, frmL, nfrm;
    double *xp, *ac, *sc;

    /* fprintf(stderr, "autocorr entered\n"); */

    /* Do some error checking */
    if (nrhs < 5)
	mexErrMsgTxt("Five input arguments required. ");

    /* Get the length of window, lag and frame*/
    winL = mxGetScalar(prhs[4]);
    nfrm = mxGetScalar(prhs[2]);
    lagL = mxGetScalar(prhs[3]);
    frmL = mxGetScalar(prhs[1]);
	
    /* Create a matrix for the return argument */
    plhs[0] = mxCreateDoubleMatrix(nfrm*lagL,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nfrm*lagL,1,mxREAL);


    /* Assign pointers to the parameters */
    xp = mxGetPr(prhs[0]);
    ac = mxGetPr(plhs[0]);
    sc = mxGetPr(plhs[1]);	

    int xpn = mxGetN(prhs[0]);
    int xpm = mxGetM(prhs[0]);

    /* fprintf(stderr, "xp size = %d x %d = %d (last val %f)\n", 
	    xpn, xpm, xpn*xpm, xp[(xpn*xpm) - 1]); */

    /* Check that number of frames doesn't exceed input data size */
    int npts = xpn*xpm;
    /* this is derived from the largest-index access line in calc_corr, i.e.
       z1 += xp[(f-1)*frmL+winL+j]*xp[(f-1)*frmL+winL+j+eta]; */
    int lastpt = (nfrm - 1 - 1)*frmL + winL + (frmL - 1) + (lagL - 1);
    if (lastpt >= npts) {
	/* FREAKOUT */
	char msg[64];
	sprintf(msg,"autocorr mex: need %d points but only %d passed", 
		lastpt+1, npts);
	mexErrMsgTxt(msg);
    }

    /* Calculating the auto-correlation with hanning windowed input */
    calc_corr(xp, ac, sc, winL, lagL, frmL, nfrm );

    /* fprintf(stderr, "autocorr done\n"); */
}

