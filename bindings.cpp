#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "compute_scores.h"

namespace py = pybind11;

PYBIND11_MODULE(compute_scores, m) {
    m.def("compute_scores", &compute_scores, "Compute CH scores",
          py::arg("ir"), py::arg("jc"), py::arg("N"),
          py::arg("lengths"), py::arg("L"), py::arg("length_max"),
          py::arg("models"), py::arg("M"));
}