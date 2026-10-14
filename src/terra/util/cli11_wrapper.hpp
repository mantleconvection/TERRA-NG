
#pragma once

#ifdef __NVCOMPILER
  #pragma diag_push
  #pragma diag_suppress code_is_unreachable
#endif

#if defined( __CUDACC__ )
#pragma nv_diag_push
#pragma nv_diag_suppress   111 // code_is_unreachable
#endif

#include "../../../extern/cli11-2.5.0/CLI11.hpp"

#if defined( __CUDACC__ )
#pragma nv_diag_pop
#endif

#pragma diag_pop
