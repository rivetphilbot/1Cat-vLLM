# SPDX-License-Identifier: Apache-2.0
"""Build only the GLM kernels (glm_sparse_mla_cuda) into glm_attn_v100/ with the current interpreter's torch.

    python glm-attn-v100/build_glm_ext.py

Much faster than setup.py, which also builds every kernel copied from flash-attention-v100. Run it with the
same Python/torch that serves (the extension is ABI-tied to that torch build).
"""
import os
import shutil

from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build", "glm_sparse_mla_cuda")
os.makedirs(BUILD, exist_ok=True)
ext = load(
    name="glm_sparse_mla_cuda",
    sources=[os.path.join(ROOT, "kernel", "glm_sparse_mla.cu")],
    extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_70,code=sm_70"],
    build_directory=BUILD,
    verbose=False,
)
dst = os.path.join(ROOT, "glm_attn_v100", os.path.basename(ext.__file__))
shutil.copy2(ext.__file__, dst)
print("installed", dst, "precision_version", ext.precision_version)
