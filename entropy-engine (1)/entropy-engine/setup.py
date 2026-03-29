# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='entropy_engine_cuda',
    ext_modules=[
        CUDAExtension(
            name='entropy_engine_cuda',
            sources=['csrc/engine.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
            libraries=['cuda']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)