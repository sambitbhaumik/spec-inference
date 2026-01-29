from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="cuda_ops",
    ext_modules=[
        CUDAExtension(
            "cuda_ops",
            [
                "sampling_kernel.cu",
                "kv_ops.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
