# meson.build
project('rtta', 'cython') # Or just 'python' if you have other source types
py = import('python').find_installation()
dep_py = py.dependency()

# Get the NumPy include directory
incdir_numpy = run_command(py, ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'], check : true ).stdout().strip()
inc_np = include_directories(incdir_numpy)

py.extension_module(
    'rtta',  # Replace with your module name
    'src/rtta.pyx', # Replace with the path to your .pyx file
    dependencies: dep_py,
    include_directories: inc_np,	
    install: true,
    # include_directories: ['src'], # Optional, if you have header files in a 'src' directory
)
