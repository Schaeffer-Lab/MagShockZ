##########################################################################################
# System specific configuration for OSIRIS
#   System:    Linux / darwin
#   Compilers: GNU family ( gcc, gfortran ), Nvidia (nvcc) 
##########################################################################################

##########################################################################################
# OSIRIS configuration

DISABLE_PGC = 1
# DISABLE_TILES = 1
DISABLE_QED = 1
DISABLE_QEDCYL = 1
DISABLE_SHEAR = 1
DISABLE_RAD = 1
DISABLE_XXFEL = 1
DISABLE_NEUTRAL_SPIN = 1
DISABLE_GR = 1
DISABLE_CYLMODES = 1
DISABLE_OVERDENSE = 1
DISABLE_OVERDENSE_CYL = 1
# APPEND_BRANCH = 1
ENABLE_CUDA = 1

# Set the following to a system optimized timer, or leave commented to use a default one
TIMER = __POSIX_TIMER__

# SIMD
# Uncomment one of the following lines to enable SSE / AVX optimized code 
#SIMD = SSE
SIMD = AVX2
#SIMD = MIC
# Numeric precision (SINGLE|DOUBLE)
#PRECISION = SINGLE
PRECISION = DOUBLE
SIMD_FLAGS = -march=core-avx2 
#SIMD_FlAGS = -xavx2
#SIMD_FLAGS = -xHost
#SIMD_FLAGS = -msse4.2

##########################################################################################
# Compilers

# When compiling AVX code the following must be used to i) enable AVX code generation (-mavx)
# and ii) use the clang integrated assembler instead of the GNU based system assembler.

F90 = /usr/bin/gfortran-11 -Wa,-q -cpp

# Use this to specify specific options for compiling .f03 files
F03 = $(F90)
# F03 = $(F90) -std=f2003

cc  = /usr/bin/gcc-11 -Wa,-q -I/usr/lib/x86_64-linux-gnu/openmpi/include
CC  = /usr/bin/gcc-11 -Wa,-q -I/usr/lib/x86_64-linux-gnu/openmpi/include
cxx = $(cc) -std=c++0x
NVCC = /usr/bin/nvcc

# Fortran preprocessor
FPP = /usr/bin/gcc-11 -C -E -x assembler-with-cpp

# This flag supresses some hyper-vigilant compilier warnings that have been deemed harmless. The list of warnings
# can be found is ./source/config.mk.warnings. If you are having some strange issues that you need to debug
# comment out DISABLE_PARANOIA and/or read the ./source/config.warnings file to allow the warnings if you think
# they may help you find the issue.
DISABLE_PARANOIA = YES

##########################################################################################
# Fortran flags

# External name mangling
UNDERSCORE = FORTRANSINGLEUNDERSCORE

# Flag to enable compilation of .f03 files (not needed for gfortran)
# F03_EXTENSION_FLAG = 

# ------------------------------- Compilation Targets ------------------------------------

# -fno-range-check is required because of the random module. 
# gfortran has a bug that considers -2147483648 to be outside the valid
# int32 range

# -pipe makes gfortran use pipes for internal process communication (instead of files)
#       which speeds up the ocmpilation process significantly

# -ffree-line-length-none removes all constraints on line size
F90FLAGS_all = -pipe -fall-intrinsics -ffree-line-length-none -fno-range-check -Wl,-Bdynamic 

# OpenMP Support
F90FLAGS_all += -fopenmp -pthread
FPP += -fopenmp -pthread
cc += -fopenmp -pthread
CC += -fopenmp -pthread


# Production
# Intel Core i7 flags

F90FLAGS_production = $(F90FLAGS_all) -Ofast -march=native -ffpe-summary=none 

# Debug

# -std=f95 is too picky

F90FLAGS_debug      = $(F90FLAGS_all) -g -Og -fbacktrace -fbounds-check \
                      -Wall -fimplicit-none -pedantic \
                      -Wimplicit-interface -Wconversion  -Wsurprising \
                      -Wunderflow  -ffpe-trap=invalid,zero,overflow
                      
#-ffpe-trap=underflow,denormal is usually too picky. For example, it may raise an exception when
# converting double precision values to single precision for diagnostics output if the
# value is outside of valid single precision range (e.g. 1e-40). It may also raise 
# an exception if a wave amplitude gets very low in a PML region. Note that in these 
# situations the value is (correctly) rounded to 0.

# Profile with Shark
F90FLAGS_profile    = -g $(F90FLAGS_production)


##########################################################################################
# C flags

CFLAGS_production = -Ofast -march=native -std=c99

CFLAGS_debug      = -Og -g -Wall -pedantic -march=native 
#-fsanitize=address

CFLAGS_profile    = -g $(CFLAGS_production) 

##########################################################################################
# NVCC flags

NVCCFLAGS_all = -I/usr/include -std=c++11 -dc

#debug
NVCCFLAGS_debug      = $(NVCCFLAGS_all) -O0 -G -g

# production
NVCCFLAGS_production = $(NVCCFLAGS_all) -O3 -dlto -use_fast_math -extra-device-vectorization

# profile
NVCCFLAGS_profile    = $(NVCCFLAGS_production) -g

# for additional compilation information, add the following options to NVCCFLAGS_debug
# -opt-info inline -res-usage --ptxas-options="-v"

# note, you must also add -res-usage to CUDA_CLINKFLAGS_debug below if specified above

# you can also add --nvlink-options="-v" to CUDA_CLINKFLAGS_debug

##########################################################################################
# Linker flags

# Add conda rpath for runtime library resolution (especially for Python/OpenSSL)
LDFLAGS = -pthread -lpthread -Wl,-rpath,$(CONDA_PREFIX)/lib

##########################################################################################
# Libraries
# MPI
# MPI_FCOMPILEFLAGS = $(shell $(OPENMPI_ROOT)/bin/mpif90 --showme:compile)
# MPI_FLINKFLAGS    = $(shell $(OPENMPI_ROOT)/bin/mpif90 --showme:link)
MPI_FCOMPILEFLAGS = $(shell /usr/bin/mpif90.openmpi --showme:compile)
MPI_FLINKFLAGS    = $(shell /usr/bin/mpif90.openmpi --showme:link)


# HDF5
H5_FCOMPILEFLAGS = -I/usr/include/hdf5/openmpi
H5_FLINKFLAGS    = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -L/usr/lib/x86_64-linux-gnu -lcurl -lcrypto -lsz -ldl -lz -lm
H5_HAVE_PARALLEL = 1

# FFTW
#FFTW_ROOT = /usr/local/fftw-3.3.10
FFTW_FCOMPILEFLAGS = -I/usr/include
# # flags for single precision
FFTW_FLINKFLAGS = -L/usr/lib/x86_64-linux-gnu -lfftw3f
# flags for double precision
# FFTW_FLINKFLAGS = -L/usr/lib/x86_64-linux-gnu -lfftw3

# CUDA
# CUDA_ROOT = /usr/local/cuda-11.5
CUDA_CLINKFLAGS_debug = -dlink
CUDA_CLINKFLAGS_production = -dlink
#-dlto
CUDA_FLINKFLAGS = -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcurand -dlto -lstdc++
# __DEBUG__ = 1

CONDA_ROOT = $(CONDA_PREFIX)
PY_FCOMPILEFLAGS = -I$(CONDA_ROOT)/include/python3.14
PY_FLINKFLAGS = -Wl,-rpath,$(CONDA_ROOT)/lib -L$(CONDA_ROOT)/lib -lpython3.14 -lssl -lcrypto

# Or maybe these linking flags
# PY_FLINKFLAGS = -Wl,-rpath,$(CONDA_ROOT)/lib $(CONDA_ROOT)/lib/libpython3.9.dylib
