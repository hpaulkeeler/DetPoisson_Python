# DetPoissonPython

Randomly simulates a determinantally-thinned Poisson point process on a rectangle.

A determinantally-thinned (Poisson) point process is essentially a discrete determinantal point process whose underlying state space is a single realization of a (Poisson) point process defined on some bounded continuous space. This is a repulsive point process, where the repulsion depends on the kernel and average density of points. For more details, see the paper by Blaszczyszyn and Keeler[1].

I wrote the simulation of the Poisson myself. To simulate the (discrete) determinantal point process, I modified the code in sample_dpp.py from this repository:

https://github.com/mbp28/determinantal-point-processes

It should be noted that there are a number of repositories that do simulate (discrete) determinantal point process in Python, including:

https://github.com/javiergonzalezh/dpp

https://github.com/mbp28/determinantal-point-processes

https://github.com/guilgautier/DPPy

https://github.com/ChengtaoLi/dpp

https://github.com/mehdidc/dpp

References: [1] Blaszczyszyn and Keeler, Determinantal thinning of point processes with network learning applications, 2018
.
