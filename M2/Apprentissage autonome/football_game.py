# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:44:09 2021

@author: Sandro
"""
import math
from scipy.optimize import linprog
obj = [-1, -2]
lhs_ineq = [[ 2, 1],[2,3]]
rhs_ineq = [5,10]
lhs_eq=[[-1,1]]
rhs_eq=[1]
bnd = [(0, math.inf), (0, math.inf)]
opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,\
A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,\
method="revised simplex")
print(opt)
print(opt.x)