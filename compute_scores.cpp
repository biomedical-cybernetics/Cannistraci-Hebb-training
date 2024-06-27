// Compile: python setup.py build_ext --inplace
// Use: import compute_scores
// scores_cell = compute_scores.compute_scores(ir, jc, N, lengths, L, length_max, models, len(models))

#include "compute_scores.h"
#include <chrono>

#define P_MAX_START 100
using namespace std;
namespace py = pybind11;
using namespace std::chrono; // Use the std::chrono namespace

void find_paths_rec(const vector<int>& ir, const vector<int>& jc,
                    int L, int length_max, const vector<int>& length_to_idx, int length,
                    vector<vector<int>>& paths_cell, vector<int>& P_MAX, vector<int>& P, vector<int>& path, vector<bool>& in_path,
                    int ns, int nv, const vector<double>& deg) {
    path[length] = nv;
    in_path[nv] = true;
    length += 1;

    if ((length >= 2) && (length_to_idx[length - 2] < L) && ((deg[ns] < deg[nv]) || ((deg[ns] == deg[nv]) && (ns < nv)))) {
        int paths_idx = nv * L + length_to_idx[length - 2];
        if (P[paths_idx] + 1 > P_MAX[paths_idx]) {
            P_MAX[paths_idx] *= 2;
            paths_cell[paths_idx].resize((length - 1) * P_MAX[paths_idx]);
        }

        for (int i = 0; i < length - 1; i++) {
            paths_cell[paths_idx][P[paths_idx] * (length - 1) + i] = path[i];
        }
        P[paths_idx] += 1;
    }

    if (length < length_max) {
        for (int i = jc[nv]; i < jc[nv + 1]; i++) {
            if (!in_path[ir[i]]) {
                find_paths_rec(ir, jc, L, length_max, length_to_idx, length, paths_cell, P_MAX, P, path, in_path, ns, ir[i], deg);
            }
        }
    }

    in_path[nv] = false;
}

void compute_scores_from_source_node(const vector<int>& ir, const vector<int>& jc, int N,
                                     const vector<double>& lengths, int L, int length_max, const vector<int>& length_to_idx,
                                     vector<vector<int>>& paths_cell, vector<int>& P_MAX, vector<int>& P, vector<int>& path, vector<bool>& in_path,
                                     int ns, const vector<double>& deg, vector<int>& nodes_lc, vector<double>& deg_i_v1, vector<double>& deg_i_v2, vector<double>& deg_i_v3,
                                     vector<double>& deg_e_v1, vector<double>& deg_e_v2, vector<double>& deg_e_v3, const vector<double>& models, int M, py::array_t<double> scores_cell) {
    int i, j, l, m, nd, n, p, paths_idx, nodes_lc_size, length;
    double deg_e1_gmean, deg_e2_gmean, deg_e3_gmean, pow_exp;
    bool a_found, b_found, c_found, d_found;

    auto scores = scores_cell.mutable_unchecked<2>();

    in_path[ns] = true;
    for (i = jc[ns]; i < jc[ns + 1]; i++) {
        find_paths_rec(ir, jc, L, length_max, length_to_idx, 0, paths_cell, P_MAX, P, path, in_path, ns, ir[i], deg);
    }
    in_path[ns] = false;

    for (l = 0; l < L; l++) {
        length = (int)lengths[l];
        pow_exp = (double)1 / (length - 1);
        for (nd = 0; nd < N; nd++) {
            paths_idx = 0;
            if ((deg[ns] < deg[nd]) || ((deg[ns] == deg[nd]) && (ns < nd))) {
                paths_idx = nd * L + l;
                nodes_lc_size = 0;
                for (p = 0; p < P[paths_idx]; p++) {
                    for (i = 0; i < length - 1; i++) {
                        n = paths_cell[paths_idx][p * (length - 1) + i];
                        if (!in_path[n]) {
                            in_path[n] = true;
                            nodes_lc[nodes_lc_size] = n;
                            nodes_lc_size += 1;
                        }
                    }
                }

                for (j = 0; j < nodes_lc_size; j++) {
                    int count = 0;
                    n = nodes_lc[j];
                    deg_i_v1[n] = 0;
                    deg_i_v2[n] = 0;
                    deg_i_v3[n] = 0;
                    deg_e_v1[n] = deg[n];
                    deg_e_v2[n] = deg[n];
                    deg_e_v3[n] = deg[n];
                    for (i = jc[n]; i < jc[n + 1]; i++) {
                        if (in_path[ir[i]]) {
                            deg_i_v2[n] += 1;
                            a_found = b_found = c_found = d_found = false;
                            for (int ii = jc[ns]; ii < jc[ns + 1]; ii++) {
                                if (ir[ii] == ir[i]) a_found = true;
                                if (ir[ii] == n) b_found = true;
                            }
                            for (int jj = jc[nd]; jj < jc[nd + 1]; jj++) {
                                if (ir[jj] == ir[i]) c_found = true;
                                if (ir[jj] == n) d_found = true;
                            }
                            if (((!a_found && !b_found) || (!c_found && !d_found)) && length > 2) {
                                count += 1;
                            } else {
                                /* Version 1 & 3 */
                                deg_i_v1[n] += 1;
                                deg_i_v3[n] += 1;
                            }
                        }
                        if ((ir[i] == ns) || (ir[i] == nd)) {
                            deg_e_v1[n] -= 1;
                            deg_e_v2[n] -= 1;
                            deg_e_v3[n] -= 1;
                        }
                    }
                    deg_e_v1[n] -= deg_i_v1[n];
                    deg_e_v2[n] -= deg_i_v2[n];
                    deg_e_v3[n] -= deg_i_v3[n];
                    deg_e_v3[n] -= count;
                }
                for (j = 0; j < nodes_lc_size; j++) {
                    in_path[nodes_lc[j]] = false;
                }
                for (p = 0; p < P[paths_idx]; p++) {
                    deg_e1_gmean = deg_e2_gmean = deg_e3_gmean = 1;
                    for (i = 0; i < length - 1; i++) {
                        n = paths_cell[paths_idx][p * (length - 1) + i];
                        deg_e1_gmean *= (1 + deg_e_v1[n]);
                        deg_e2_gmean *= (1 + deg_e_v2[n]);
                        deg_e3_gmean *= (1 + deg_e_v3[n]);
                    }
                    deg_e1_gmean = pow(deg_e1_gmean, pow_exp);
                    deg_e2_gmean = pow(deg_e2_gmean, pow_exp);
                    deg_e3_gmean = pow(deg_e3_gmean, pow_exp);
                    for (m = 0; m < M; m++) {
                        if (models[m] == 0) {
                            scores(m * L + l, ns * N + nd) += 1 / deg_e1_gmean;
                        } else if (models[m] == 1) {
                            scores(m * L + l, ns * N + nd) += 1 / deg_e2_gmean;
                        } else if (models[m] == 2) {
                            scores(m * L + l, ns * N + nd) += 1 / deg_e3_gmean;
                        }
                    }
                }
                for (m = 0; m < M; m++) {
                    scores(m * L + l, nd * N + ns) = scores(m * L + l, ns * N + nd);
                }
            }
            P[paths_idx] = 0;
        }
    }
}

py::array_t<double> compute_scores(const vector<int>& ir, const vector<int>& jc, int N, const vector<double>& lengths, int L, int length_max, const vector<double>& models, int M) {
    auto start = high_resolution_clock::now();
    vector<double> deg(N, 0);
    for (int n = 0; n < N; n++) {
        deg[n] = jc[n + 1] - jc[n];
    }

    vector<int> length_to_idx(length_max - 1, L);
    for (int l = 0; l < L; l++) {
        length_to_idx[lengths[l] - 2] = l;
    }

//    std::vector<std::vector<double>> scores_cell(L * M, std::vector<double>(N * N, 0.0));
    py::array_t<double> scores_cell({L * M, N * N});

    int num_threads = omp_get_max_threads();

    // Timing parallel section start
    auto parallel_start = high_resolution_clock::now();

    #pragma omp parallel if(num_threads > 1) num_threads(num_threads) shared(ir, jc, N, lengths, L, length_max, length_to_idx, deg, models, M, scores_cell)
    {
        vector<vector<int>> paths_cell(L * N);
        vector<int> P_MAX(L * N, P_MAX_START);
        vector<int> P(L * N, 0);
        vector<int> path(length_max, 0);
        vector<bool> in_path(N, false);
        vector<int> nodes_lc(N, 0);
        vector<double> deg_i_v1(N, 0), deg_i_v2(N, 0), deg_i_v3(N, 0);
        vector<double> deg_e_v1(N, 0), deg_e_v2(N, 0), deg_e_v3(N, 0);
        for (int l = 0; l < L; l++) {
            for (int n = 0; n < N; n++) {
                int paths_idx = n * L + l;
                P_MAX[paths_idx] = P_MAX_START;
                P[paths_idx] = 0;
                paths_cell[paths_idx].resize((lengths[l] - 1) * P_MAX[paths_idx]);
            }
        }
        #pragma omp for schedule(dynamic)
        for (int n = 0; n < N; n++) {
            compute_scores_from_source_node(ir, jc, N, lengths, L, length_max, length_to_idx, paths_cell, P_MAX, P, path, in_path, n, deg, nodes_lc, deg_i_v1, deg_i_v2, deg_i_v3, deg_e_v1, deg_e_v2, deg_e_v3, models, M, scores_cell);
        }
    }

    // End timing parallel section
    auto parallel_end = high_resolution_clock::now();
    auto parallel_duration = duration_cast<milliseconds>(parallel_end - parallel_start);
    std::cout << "Parallel section took: " << parallel_duration.count() << " milliseconds" << std::endl;

    // End timing
    auto end = high_resolution_clock::now();
    auto total_duration = duration_cast<milliseconds>(end - start);
    std::cout << "Total function took: " << total_duration.count() << " milliseconds" << std::endl;

    return scores_cell;
}

