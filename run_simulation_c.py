import numpy as np
from scipy.io import savemat, loadmat
from scipy.sparse import csr_matrix, coo_matrix
from concurrent.futures import ProcessPoolExecutor
import functools
import time
import os
import scipy.sparse.csgraph as spgraph
from scipy.stats import spearmanr, pointbiserialr, rankdata

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
import compute_scores

def prediction_evaluation(scores, labels):
    """
    Evaluate predictions with various metrics.

    :param scores: Numerical scores for the samples
    :param labels: Binary labels indicating the positive and negative samples
    :return: Dictionary containing precision, AUP, AUPR, AUC, and point-biserial correlation coefficient
    """
    # Input validation
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    assert scores.ndim == 1 and labels.ndim == 1, "scores and labels must be 1D vectors"
    assert np.all(np.isfinite(scores)), "scores must be finite"
    assert set(labels) <= {0, 1}, "labels must be binary"
    assert len(scores) == len(labels), "scores and labels must have the same length"

    n1 = np.sum(labels == 1)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError('labels cannot be all ones or all zeros')

    # Calculate precision at last threshold
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    p = precision[-1]

    # Calculate area under the precision-recall curve
    aupr = auc(recall, precision)

    # Calculate AUP
    aup = np.trapz(precision[labels.argsort()[::-1]], dx=1/n1) if n1 > 1 else 0

    # Calculate AUC
    auc_score = roc_auc_score(labels, scores)

    # Calculate point-biserial correlation coefficient
    m1 = np.mean(scores[labels == 1])
    m0 = np.mean(scores[labels == 0])
    s = np.std(scores)
    pbcorr = (m1 - m0) / s * np.sqrt(n1 * n0 / (len(scores) * (len(scores) - 1)))

    return {
        'Precision': p,
        'AUP': aup,
        'AUPR': aupr,
        'AUC': auc_score,
        'Point-Biserial Correlation Coefficient': pbcorr
    }


def find_paths_rec(ir, jc, L, length_max, length_to_idx, length, paths_cell, P, path, in_path, ns, nv, deg):
    # Add the visited node nv to the path
    path[length] = nv
    # print(len(in_path))
    in_path[nv] = True
    length += 1  # Current path length

    # Check if the current path length is among the lengths requested
    # And check if node ns is responsible over node nv for storing the paths
    if length >= 2 and length_to_idx[length - 2] < L and (deg[ns] < deg[nv] or (deg[ns] == deg[nv] and ns < nv)):
        paths_cell[nv][P[nv]] = path[:length - 1]
        P[nv] += 1

    # If maximum path length not reached, continue the recursion from each neighbour not in the path
    if length < length_max:
        for i in range(jc[nv], jc[nv + 1]):
            if not in_path[ir[i]]:
                find_paths_rec(ir, jc, L, length_max, length_to_idx, length, paths_cell, P, path, in_path, ns, ir[i],
                               deg)

    # Remove the visited node nv from the path
    in_path[nv] = False
    # path = np.zeros(lengths.max(), dtype=np.intp)


def compute_scores_from_source_node(A, ir, jc, N, lengths, L, length_max, length_to_idx, paths_cell, P, path, in_path,
                                    ns, deg, scores_cell, deg_e_mean):
    # Initialize in_path for the source node
    in_path[ns] = True
    t1 = time.time()
    # Start recursion from each neighbour
    for i in range(jc[ns], jc[ns + 1]):
        find_paths_rec(ir, jc, L, length_max, length_to_idx, 0, paths_cell, P, path, in_path, ns, ir[i], deg)
    t2 = time.time()
    in_path[ns] = False  # Remove the source node from the path
    for l in range(L):
        length = int(lengths[l])
        pow_exp = 1 / (length - 1)
        for nd in range(N):
            if P[nd] == 0:
                continue
            if deg[ns] < deg[nd] or (deg[ns] == deg[nd] and ns < nd):
                deg_i = np.zeros(N, dtype=float)
                paths = []
                for p in range(P[nd]):
                    if set(paths_cell[nd, p]) not in paths:
                        deg_i[paths_cell[nd, p]] += 1
                        paths.append(set(paths_cell[nd, p]))

                deg_e = deg - deg_i - np.sum(A[[ns, nd]], axis=0)
                scores_cell[nd] = np.sum(
                    1 / np.sqrt((deg_e[paths_cell[nd][:P[nd], 0]] + 1) * (deg_e[paths_cell[nd][:P[nd], 1]] + 1)))

def parallel_compute_scores(n, ir, jc, N, lengths, L, length_max, length_to_idx, deg):
    path = np.zeros(lengths.max(), dtype=np.intp)
    in_path = np.zeros(N, dtype=bool)
    # deg_i = np.zeros(N, dtype=float)
    # deg_e = np.zeros(N, dtype=float)
    P = np.zeros((N), dtype=np.intp)
    paths_cell = np.zeros((N, 2*N, 2), dtype=np.intp)
    scores = np.zeros(N)
    deg_e_mean = np.zeros(N)
    compute_scores_from_source_node(A, ir, jc, N, lengths, L, length_max, length_to_idx,
                                    paths_cell, P, path, in_path, n, deg, scores, deg_e_mean)
    # print(f"Done! Process ID: {os.getpid()}, n: {n}, Time: {time.time()}")
    return n, scores


iters = 10
models = [1]
# num_threads = 128
x = loadmat("/mnt/ssd/yingtao/evaluate_V3_V2_V1/ATLAS/matrices/citation_arxivhepph2003.mat")["x"]
matrices = loadmat("/mnt/ssd/yingtao/evaluate_V3_V2_V1/ATLAS/sparsified_matrices/citation_arxivhepph2003_linkrem10.mat")["matrices"]
# print(x.shape)
# x = loadmat("Yeast_DIP_net.mat")["x_lcc"]
# matrices = loadmat("network_perturbed_10percent_GSP_GSN_Yeast.mat")["L_net"]
# GS = loadmat("list_pairs_10percent_GSP_GSN_Yeast.mat")["L_pairs"]
S = []
for i in range(iters):
    A = np.array(matrices[i][0].todense())
    print(A.shape)
    N = A.shape[0]
    lengths = np.array([3])
    length_max = 3
    L = len(lengths)
    sparse_A = csr_matrix(A)

    ir = sparse_A.indices
    jc = sparse_A.indptr
    deg = np.sum(A, axis=1)

    # length_to_idx = np.full(length_max - 1, L, dtype=np.intp)
    # length_to_idx[[lengths[lengths >= 2] - 2]] = np.arange(np.sum(lengths >= 2))
    # print(f"length_to_idx: {length_to_idx}")

    t1 = time.time()



    scores_cell = compute_scores.compute_scores(ir, jc, N, lengths, L, length_max, models, len(models))
    t2 = time.time()
    print(f"time: {t2 - t1}")
    print(f"{type(scores_cell)}")
    print(scores_cell.shape)
    scores_cell = np.array(scores_cell)
    scores_cell = scores_cell.reshape((-1, N, N))
    # print(sum(scores_cell[0]))
    # exit()

    # func = functools.partial(parallel_compute_scores, ir=ir, jc=jc, N=N, lengths=lengths, L=L,
    #                          length_max=length_max,
    #                          length_to_idx=length_to_idx, deg=deg)
    # results = []
    # for i in range(N):
    #     results.append(func(i))
    # print("Done!")
    # scores_cell = np.zeros((N, N))
    # for n, result in results:
    #     scores_cell[n] = result
    #     # Process or integrate results here

    # scores_cell += scores_cell.transpose(1, 0)


    print(scores_cell)
    print(np.mean(scores_cell[0, :, :]))
    exit()


    print(round(t2 - t1, 2))

    # Subranking
    s = A * 1 / (1 + scores_cell)
    s = spgraph.shortest_path(csr_matrix(s), directed=False)
    rho, _ = spearmanr(s)
    # e1, e2 = triangle_indices(A)
    e1, e2 = np.triu_indices_from(A, k=1)
    indices = np.ravel_multi_index((e1, e2), dims=A.shape)
    scores = np.column_stack((scores_cell.flat[indices], rho.flat[indices]))
    _, scores = np.unique(scores, return_inverse=True, axis=0)

    # tiedrank
    scores = rankdata(scores, method='average')
    print(e1[:10], e2[:10])
    print(scores[:10])
    break
    scores = coo_matrix((scores, (e1, e2)), shape=(x.shape[0], x.shape[1])).todense()

    GSN = GS[i][0]
    GSP = GS[i][1]
    labels = np.zeros(len(GSN) + len(GSP))
    labels[len(GSN):] = 1
    predictions = np.concatenate(np.array([scores[GSN[:, 0], GSN[:, 1]], scores[GSP[:, 0], GSP[:, 1]]]),
                                 axis=-1).reshape(-1)
    print(predictions.shape, labels.shape)
    results = prediction_evaluation(predictions, labels)
    break


