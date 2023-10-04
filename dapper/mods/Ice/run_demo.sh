#!/bin/bash

export OMP_NUM_THREADS=1
source "/home/ivo/.local/opt/firedrake/firedrake/bin/activate"
export PYTHONPATH="/home/ivo/Code/pyIce"
mpiexec -n 8 python demo.py

