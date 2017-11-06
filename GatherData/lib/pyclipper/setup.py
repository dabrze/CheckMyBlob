import numpy
from distutils.core import setup
from distutils.extension import Extension
import setup_clipper as custom_setup


if custom_setup.has_cython:
    clipper_ext = Extension(
        "pyclipper.pyclipper",
        ["src/clipper.pyx", "src/_clipper.cpp"],
        language="c++",
        extra_compile_args=['-O3']
    )
else:
    clipper_ext = Extension(
        "pyclipper.pyclipper",
        ["src/clipper.cpp", "src/_clipper.cpp"],
        language="c++",
        extra_compile_args=['-O3']
    )

print('NumPy include:', numpy.get_include())

setup(
    name="pyclipper",
    version='0.0.1',
    description="pyclipper bindings to Clipper",
    author=["Przemyslaw Porebski, Kowiel Marcin"],
    maintainer="Przemyslaw Porebski",
    maintainer_email="przemek@iwonka.med.virginia.edu",
    cmdclass={
        'build_ext': custom_setup.build_clipper_ext,
    },
    packages=['pyclipper'],
    ext_modules=[clipper_ext],
    include_dirs=[numpy.get_include()],
    requires=['numpy'],
)
