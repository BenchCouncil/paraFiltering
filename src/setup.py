# python setup.py build_ext --inplace
from setuptools import setup, Extension

mul_filter = [
    Extension(
        "boost_filter_refine",
        ["filter_refine_boost.cpp"],
        libraries=['boost_python310', 'boost_numpy310'],  # 根据实际库文件名称填写
        extra_compile_args=['-std=c++17'],
        include_dirs=['/usr/local/include'],
        library_dirs=['/usr/local/lib'],  # 根据实际库文件路径填写
        language='c++',
        extra_link_args=['-fopenmp']
    )
]

setup(
    name='hjp_filter_refine_boost_for_amazon',
    ext_modules=mul_filter
)
