// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "../hopper/flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM128
// Template parameters: <Arch, T, kHeadDim, kHeadDimV, Split, PagedKVNonTMA, Has_softcap, PackGQA>
template void run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, false, false, false, false>(Flash_fwd_params &params, cudaStream_t stream);
#endif
