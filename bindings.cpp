#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "CH_scores_new_v2.h"

namespace py = pybind11;

PYBIND11_MODULE(CH_scores, m) {
    m.def("CH_scores_new_v2", &CH_scores_new_v2, "Compute CH scores New V2",
          py::arg("ir"), py::arg("jc"), py::arg("N"),
          py::arg("lengths"), py::arg("L"), py::arg("length_max"),
          py::arg("models"), py::arg("M"));
}