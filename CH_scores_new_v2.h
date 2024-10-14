#ifndef CH_SCORES_NEW_V2
#define CH_SCORES_NEW_V2

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>

namespace py = pybind11;

py::array_t<double> CH_scores_new_v2(const std::vector<int>& ir, const std::vector<int>& jc, int N,
                    const std::vector<double>& lengths, int L, int length_max,
                    const std::vector<double>& models, int M);

#endif