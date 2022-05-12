#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:03:56 2022

@author: ivo
"""

from IPython import get_ipython
ip = get_ipython()
ip.run_cell(("!OMP_NUM_THREADS=1; "
             "source /home/ivo/.local/opt/firderake/firedrake/bin/activate; "
             "export PYTHONPATH=/home/ivo/Code/pyIce; "
             "mpiexec -n 2 python demo.py"))