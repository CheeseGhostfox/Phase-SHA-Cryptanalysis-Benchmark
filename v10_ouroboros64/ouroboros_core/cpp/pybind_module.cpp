/**
 * Optional PyBind11 binding module for Ouroboros-64 C++17 OpenMP Engine
 * (Built automatically when pybind11 is available, e.g. on Colab Linux).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "crystal_core.hpp"

namespace py = pybind11;
using namespace ouroboros;

PYBIND11_MODULE(_ouroboros_pybind, m) {
    m.doc() = "Phase-SHA V10 Ouroboros-64 C++17 OpenMP Spacetime Crystal Core";

    m.def("max_threads", []() -> int {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    });

    m.def("eval_cluster_flips_batch",
        [](
            uint32_t R,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> H_target,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> W_base,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> S_base,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> C_base,
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> cluster_idx,
            uint32_t alpha,
            uint32_t beta,
            uint32_t gamma,
            bool allow_boundary_flip
        ) {
            uint32_t B_cryst = static_cast<uint32_t>(W_base.shape(0));
            uint32_t K_cands = static_cast<uint32_t>(cluster_idx.shape(0) / B_cryst);
            uint32_t M_max = static_cast<uint32_t>(cluster_idx.shape(1));
            auto out_dE = py::array_t<int64_t>(B_cryst * K_cands);
            auto out_nE = py::array_t<uint64_t>(B_cryst * K_cands);
            {
                py::gil_scoped_release nogil;
                eval_cluster_flips_batch(
                    B_cryst, K_cands, M_max, R,
                    H_target.data(), W_base.data(), S_base.data(), C_base.data(),
                    cluster_idx.data(),
                    out_dE.mutable_data(), out_nE.mutable_data(),
                    alpha, beta, gamma, allow_boundary_flip
                );
            }
            return py::make_tuple(out_dE, out_nE);
        },
        py::arg("R"), py::arg("H_target"), py::arg("W_base"), py::arg("S_base"),
        py::arg("C_base"), py::arg("cluster_idx"),
        py::arg("alpha") = 1u, py::arg("beta") = 1u, py::arg("gamma") = 1u,
        py::arg("allow_boundary_flip") = false
    );
}
