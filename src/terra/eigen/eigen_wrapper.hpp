#pragma once

#ifdef __NVCOMPILER
  #pragma diag_push
  #pragma diag_suppress code_is_unreachable
#endif

#if defined( __CUDACC__ )
#pragma nv_diag_push
#pragma nv_diag_suppress   111 // code_is_unreachable
#pragma nv_diag_suppress 20011 // __host__ annotation is ignored
#pragma nv_diag_suppress 20012 // annotation ignored on defaulted function
#pragma nv_diag_suppress 20013 // invalid_host_device_call
#pragma nv_diag_suppress 20014 // __device__ annotation is ignored
#pragma nv_diag_suppress 20015 // invalid_constexpr_call
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE

#include "../../../extern/eigen-3.4.0/Eigen/Eigen"

#pragma clang diagnostic pop

#if defined( __CUDACC__ )
#pragma nv_diag_pop
#endif

#pragma diag_pop
