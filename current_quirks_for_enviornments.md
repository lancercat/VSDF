This document lists the quirks when setting up the enviorment.

As the rolling nature of arch-based systems, problems come and go.

We document problems and solutions when they come, and remove when they go.

----------------------------------------------------------------------------
## problem
Found in 02-Jun-2022, Manjaro and Arch breaks CUDA by upgrading to gcc12
## Solution
  1. Install GCC11: pacman -S gcc11
  2. Link them to CUDA path:
```
   sudo rm /opt/cuda/bin/gcc
   sudo ln -s /usr/bin/gcc-11 "/opt/cuda/bin/gcc"
   sudo rm /opt/cuda/bin/g++
   sudo ln -s /usr/bin/g++-11 "/opt/cuda/bin/g++"
```
