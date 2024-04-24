/*_______________________________________________________________________
 *
 * > INPUT
 * x - adjacency matrix (requirements: sparse, symmetric, zero-diagonal)
 * lengths - vector of path lengths to compute (requirements: type double, values >= 2, unique and sorted in increasing order)
 * models - vector of CH models to compute (requirements: type double, possible values [0,1,2,3]);
 *          legend: 0 = RA, 1 = CH1, 2 = CH2, 3 = CH3
 * T - [optional] scalar value indicating the number of threads;
 *     if not given, it is set to the maximum (equal to the OMP_NUM_THREADS environment variable);
 *     if higher than the maximum, it is set to the maximum
 *     if 1 or lower, no parallel computation;
 *
 * > OUTPUT
 * scores - cell array of size equal to [lengths,models], for each path length and CH model it contains the scores matrix;
 *          the order of paths lengths and CH models is the same as in the input variables "lengths" and "models"
 *
 * Usage:
 * scores = CH_scores_mex(x, lengths, models, T);
 * scores = CH_scores_mex(x, lengths, models);
 *
 * Example:
 * To compute path lengths [2,3] and all the CH models [RA,CH1,CH2,CH3] with the maximum number of threads:
 * scores = CH_scores_mex(x, [2,3], [0,1,2,3]);
 * the output variable "scores" will be a 2x4 cell array, each element is a scores matrix of the same size as the adjacency matrix
 *
 * Compile in Windows:
 * Go to MATLAB "Add-Ons" and install "MATLAB Support for MinGW-w64 C/C++ Compiler"
 * Build the MEX function using the following MATLAB command (change the MinGW path if needed):
 * mex C:\ProgramData\MATLAB\SupportPackages\R2020b\3P.instrset\mingw_w64.instrset\lib\gcc\x86_64-w64-mingw32\6.3.0\libgomp.a CH_scores_mex.c CFLAGS='$CFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp'
 *
 * Compile in Linux:
 * Build the MEX functions using the following MATLAB commands:
 * mex CH_scores_mex.c CFLAGS='$CFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp'
 *
 * 02-05-2022 - coded by Alessandro Muscoloni
 *_______________________________________________________________________*/

#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <omp.h>
#define P_MAX_START 100

void find_paths_rec(mwIndex *ir, mwIndex *jc,
        mwSize L, mwSize length_max, mwIndex *length_to_idx, mwSize length,
        mwIndex **paths_cell, mwSize *P_MAX, mwSize *P, mwIndex *path, mxLogical *in_path,
        mwIndex ns, mwIndex nv, mxDouble *deg)
{
    /* add the visited node nv to the path */
    path[length] = nv;
    in_path[nv] = true;
    length += 1;	/* current path length */
    
    if ((length>=2) && (length_to_idx[length-2]<L) &&               /* check if the current path length is among the lengths requested */
       ((deg[ns]<deg[nv]) || ((deg[ns]==deg[nv]) && (ns<nv))))      /* check if node ns is responsible over node nv for storing the paths */
    {
        /* if the allocated memory for a paths matrix is saturated, double the amount */
        mwIndex paths_idx = nv*L + length_to_idx[length-2];
        if (P[paths_idx]+1 > P_MAX[paths_idx])
        {
            P_MAX[paths_idx] *= 2;
            paths_cell[paths_idx] = (mwIndex*) realloc(paths_cell[paths_idx], (length-1)*P_MAX[paths_idx]*sizeof(mwIndex));     /* NB: use realloc because mxRealloc is not compatible with parallel threads */
        }
        
        /* store the intermediate nodes (length-1) of the current path */
        for (mwIndex i=0; i<length-1; i++)
        {
            paths_cell[paths_idx][P[paths_idx]*(length-1)+i] = path[i];
        }
        P[paths_idx] += 1;
    }
    
    /* if maximum path length not reached, continue the recursion from each neighbour not in the path */
    if (length<length_max)
    {
        for (mwIndex i=jc[nv]; i<jc[nv+1]; i++)
        {
            if (!in_path[ir[i]])
            {
                find_paths_rec(ir, jc, L, length_max, length_to_idx, length, paths_cell, P_MAX, P, path, in_path, ns, ir[i], deg);
            }
        }
    }
    
    /* remove the visited node nv from the path */
    in_path[nv] = false;
}

/*_______________________________________________________________________*/

void compute_scores_from_source_node(mwIndex *ir, mwIndex *jc, mwSize N,
        mxDouble *lengths, mwSize L, mwSize length_max, mwIndex *length_to_idx,
        mwIndex **paths_cell, mwSize *P_MAX, mwSize *P, mwIndex *path, mxLogical *in_path,
        mwIndex ns, mxDouble *deg, mwIndex *nodes_lc, mxDouble *deg_i, mxDouble *deg_e,
        mxDouble *models, mwSize M, mxDouble **scores_cell)
{
    mwIndex i, j, l, m, nd, n, p, paths_idx, nodes_lc_size, length;
    mxDouble deg_gmean, deg_i_gmean, deg_i1_gmean, deg_e1_gmean, pow_exp;
    
    /* find paths from source node to all other nodes */
    in_path[ns] = true;     /* add the source node to the path */
    for (i=jc[ns]; i<jc[ns+1]; i++)
    {
        /* start recursion from each neighbour */
        find_paths_rec(ir, jc, L, length_max, length_to_idx, 0, paths_cell, P_MAX, P, path, in_path, ns, ir[i], deg);
    }
    in_path[ns] = false;    /* remove the source node from the path */
    
    /* for each path length requested and for each destination node, compute CH scores */
    for (l=0; l<L; l++)
    {
        length = (mwIndex) lengths[l];
        pow_exp = (mxDouble) 1/(length-1);
        for (nd=0; nd<N; nd++)
        {   
            // if (nd == ns){
            //     paths_idx=0;
            // }
            if ((deg[ns]<deg[nd]) || ((deg[ns]==deg[nd]) && (ns<nd)))   /* check if node ns is responsible over node nd for computing the CH scores */
            {
                /* find local community nodes (intermediate nodes involved in any path) */
                paths_idx = nd*L + l;
                nodes_lc_size = 0;  /* number of local community nodes */
                for (p=0; p<P[paths_idx]; p++)
                {
                    for (i=0; i<length-1; i++)
                    {
                        n = paths_cell[paths_idx][p*(length-1)+i];  /* local community node */
                        if (!in_path[n])
                        {
                            in_path[n] = true;              /* here in_path is used as logical array for the local community nodes */
                            nodes_lc[nodes_lc_size] = n;
                            nodes_lc_size += 1;
                        }
                    }
                }
                
                /* for each local community node, compute internal and external degree */
                for (j=0; j<nodes_lc_size; j++)
                {
                    n = nodes_lc[j];    /* local community node */
                    deg_i[n] = 0;       /* internal degree: neighbours that are local community nodes */
                    deg_e[n] = deg[n];  /* external degree: neighbours that are not local community nodes or source/destination nodes */
                    for (i=jc[n]; i<jc[n+1]; i++)
                    {
                        if (in_path[ir[i]])
                        {
                            deg_i[n] += 1;
                        }
                        if ((ir[i]==ns) || (ir[i]==nd))
                        {
                            deg_e[n] -= 1;
                        }
                        // if ir[i]
                    }
                    deg_e[n] -= deg_i[n];
                }
                
                /* reset in_path variable */
                for (j=0; j<nodes_lc_size; j++)
                {
                    in_path[nodes_lc[j]] = false;
                }
                
                /* compute CH scores */
                for (p=0; p<P[paths_idx]; p++)
                {
                    deg_gmean = 1;      /* geometric mean of degree of intermediate nodes */
                    deg_i_gmean = 1;    /* geometric mean of internal degree of intermediate nodes */
                    deg_i1_gmean = 1;   /* geometric mean of 1 + internal degree of intermediate nodes */
                    deg_e1_gmean = 1;   /* geometric mean of 1 + external degree of intermediate nodes */
                    for (i=0; i<length-1; i++)
                    {
                        n = paths_cell[paths_idx][p*(length-1)+i];
                        deg_gmean *= deg[n];
                        deg_i_gmean *= deg_i[n];
                        deg_i1_gmean *= (1 + deg_i[n]);
                        deg_e1_gmean *= (1 + deg_e[n]);
                    }
                    deg_gmean = pow(deg_gmean, pow_exp);
                    deg_i_gmean = pow(deg_i_gmean, pow_exp);
                    deg_i1_gmean = pow(deg_i1_gmean, pow_exp);
                    deg_e1_gmean = pow(deg_e1_gmean, pow_exp);
                    
                    for (m=0; m<M; m++)
                    {
                        if (models[m]==0)       /* RA */
                        {
                            scores_cell[m*L+l][ns*N+nd] += 1 / deg_gmean;
                        }
                        else if (models[m]==1)  /* CH1 */
                        {
                            scores_cell[m*L+l][ns*N+nd] += deg_i_gmean / deg_gmean;
                        }
                        else if (models[m]==2)  /* CH2 */
                        {
                            scores_cell[m*L+l][ns*N+nd] += deg_i1_gmean / deg_e1_gmean;
                        }
                        else if (models[m]==3)  /* CH3 */
                        {
                            scores_cell[m*L+l][ns*N+nd] += 1 / deg_e1_gmean;
                        }
                    }
                }
                
                /* make scores matrices symmetric */
                for (m=0; m<M; m++)
                {
                    scores_cell[m*L+l][nd*N+ns] = scores_cell[m*L+l][ns*N+nd];
                }
            }
            /* reset number of current paths */
            P[paths_idx] = 0;
        }
    }
}

/*_______________________________________________________________________*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* parse input variables */
    mwIndex *ir = mxGetIr(prhs[0]);             /* sparse matrix representation: neighbours of node n are ir[i] with jc[n] <= i < jc[n+1] */
    mwIndex *jc = mxGetJc(prhs[0]);
    mwSize N = mxGetN(prhs[0]);                 /* number of nodes */

    // printf("N: %d\n", N);
    // printf("jc[0]: %d\n", jc[0]);
    // printf("jc[1]: %d\n", jc[1]);
    // printf("jc[N-1]: %d, ir[N-1]: %d\n", jc[N-1], ir[jc[N-1]]);
    // printf("jc[N]: %d, ir[N]: %d\n", jc[N], ir[jc[N]]);
    mxDouble *lengths = mxGetPr(prhs[1]);       /* path lengths */
    mwSize L = mxGetNumberOfElements(prhs[1]);  /* number of path lengths */
    mwSize length_max = (mwSize) lengths[L-1];  /* maximum path length */
    mxDouble *models = mxGetPr(prhs[2]);        /* CH models */
    mwSize M = mxGetNumberOfElements(prhs[2]);  /* number of CH models */
    mwSize T;                                   /* number of threads */
    if (nrhs == 4)
    {
        T = (mwSize) mxGetScalar(prhs[3]);
        T = T < 1 ? 1 : T;
        T = T > omp_get_max_threads() ? omp_get_max_threads() : T;
    }
    else
    {
        T = omp_get_max_threads();
    }
        
    /* initialize support variables */
    mwIndex n, l, length;
    mxDouble *deg = (mxDouble*) mxCalloc(N, sizeof(mxDouble));  /* node degree */
    for (n=0; n<N; n++)
    {
        deg[n] = jc[n+1] - jc[n];
    }
    mwIndex *length_to_idx = (mwIndex*) mxCalloc(length_max-1, sizeof(mwIndex));    /* vector mapping each path length >= 2 to its index in the original "lengths" vector (index L if not present) */
    for (length=2; length<length_max+1; length++)
    {
        length_to_idx[length-2] = L;
    }
    for (l=0; l<L; l++)
    {
        length_to_idx[(mwIndex) lengths[l]-2] = l;
    }
    
    /* create Matlab cell array output variable and allocate memory for scores matrices */
    mwIndex m, scores_idx;
    mwSize dims[2];
    dims[0] = L;
    dims[1] = M;
    plhs[0] = mxCreateCellArray(2, dims);                                       /* L*M cell array output variable */
    mxDouble **scores_cell = (mxDouble**) mxCalloc(L*M, sizeof(mxDouble*));     /* support variable storing pointers to L*M scores matrices */
    mxArray *scores_matrix;
    for (l=0; l<L; l++)
    {
        for (m=0; m<M; m++)
        {
            scores_idx = m*L + l;
            scores_matrix = mxCreateDoubleMatrix(N, N, mxREAL);                 /* create N*N scores matrix */
            mxSetCell(plhs[0], scores_idx, scores_matrix);                      /* set the scores matrix as element of the cell array output variable */
            scores_cell[scores_idx] = (mxDouble*) mxGetPr(scores_matrix);       /* store pointer to the scores matrix */
        }
    }
    
    /* parallel section */
    #pragma omp parallel if(T>1) num_threads(T) shared(ir, jc, N, lengths, L, length_max, length_to_idx, deg, models, M, scores_cell) private(n, l)
    {
        /* allocate memory for support variables for each thread */
        mwIndex **paths_cell = (mwIndex**) mxCalloc(L*N, sizeof(mwIndex*));     /* pointers to L*N paths matrices */
        mwSize *P_MAX = (mwSize*) mxCalloc(L*N, sizeof(mwSize));                /* for each of L*N paths matrices, number of paths for which memory is currently allocated */
        mwSize *P = (mwSize*) mxCalloc(L*N, sizeof(mwSize));                    /* for each of L*N paths matrices, number of current paths */
        mwIndex *path = (mwIndex*) mxCalloc(length_max, sizeof(mwIndex));       /* array of nodes in the current path (except source node) */
        mxLogical *in_path = (mxLogical*) mxCalloc(N, sizeof(mxLogical));       /* logical array indicating for each node if it is in the current path */
        mwIndex *nodes_lc = (mwIndex*) mxCalloc(N, sizeof(mwIndex));            /* array of local community nodes */
        mxDouble *deg_i = (mxDouble*) mxCalloc(N, sizeof(mxDouble));            /* internal degree of nodes */
        mxDouble *deg_e = (mxDouble*) mxCalloc(N, sizeof(mxDouble));            /* external degree of nodes */
        mwIndex paths_idx;
        for (l=0; l<L; l++)
        {
            for (n=0; n<N; n++)
            {
                paths_idx = n*L + l;
                P_MAX[paths_idx] = P_MAX_START;
                P[paths_idx] = 0;
                paths_cell[paths_idx] = (mwIndex*) calloc((lengths[l]-1)*P_MAX[paths_idx], sizeof(mwIndex));   /* allocate memory to store the intermediate nodes (length-1) of P_MAX_START paths */
            }
        }
        
        /* parallel for loop: for each source node, compute scores to other nodes */
        #pragma omp for schedule(dynamic)
        for (n=0; n<N; n++)
        {
            compute_scores_from_source_node(ir, jc, N, lengths, L, length_max, length_to_idx,
                    paths_cell, P_MAX, P, path, in_path, n, deg, nodes_lc, deg_i, deg_e, models, M, scores_cell);
        }
        
        /* free memory for each thread */
        for (l=0; l<L; l++)
        {
            for (n=0; n<N; n++)
            {
                paths_idx = n*L + l;
                free(paths_cell[paths_idx]);
            }
        }
        mxFree(paths_cell);
        mxFree(P_MAX);
        mxFree(P);
        mxFree(path);
        mxFree(in_path);
        mxFree(nodes_lc);
        mxFree(deg_i);
        mxFree(deg_e);
    }

    /* free memory */
    mxFree(deg);
    mxFree(length_to_idx);
    mxFree(scores_cell);
}
