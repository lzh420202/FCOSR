from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='my_tools',
    version='0.1',
    ext_modules=[
        cpp_extension.CUDAExtension('tools_cuda', [
            'src/tools_cuda.cpp',
            'src/tools_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })