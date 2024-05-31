# Author: Anar Abdullayev

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def FEM(num_points): 
    
    """
    Purpose: This function provides the FEM solution for the following problem: 
    
                -u'' + p(x)u'+ q(x)u = f(x)    in    Ω = (a,b),     
                 u'(a) = u_n, u(b) = u_d.
    Parameters
    ----------
    num_points: number of points in the mesh
    
    Outputs
    ----------
    exact: Exact solution of the problem
    
    approximation: FEM solution of the problem
    
    coefficients: Solution of the stiffness-load system
    
    nodes: Generated mesh
    
    grad_exact: Derivative of the exact solution
    
    grad_approximation: Derivative of FEM solution
    
    """

    # Boundary conditions
    
    a = -1
    
    b = 2

    u_n = eval("math.exp(-1)*(math.sin(-1) + math.cos(-1))")

    u_d = eval("math.exp(2)*math.sin(2)")
    
    # Particular solution
    
    def particular(x):
        
        return u_d + u_n*(x-b)

    # Coefficient and source functions 

    def p(x): 

        return (1 - 2*x)/(1 + x**2)

    def q(x): 

        return np.exp(-x)/(1 + x**2)
    
    def f(x):
        
        return (np.exp(x)*np.cos(x)*(-1-2*x-2*x**2) + np.sin(x)*(1 + np.exp(x)*(1-2*x)))/(1 + x**2)
    
    # Source function of the corresponding ODE with homogeneous boundary conditions

    def g(x):

        return f(x) - u_n*(p(x) + q(x)*(x-2)) - u_d*q(x)
    
    # Generate mesh -> x0,x1,x2,...,xn

    #nodes = np.round(np.arange(-1,2+0.1,0.1),2) - activate this if a step size is specified

    nodes = np.linspace(a,b,num_points) # activate this if the number of points is specified

    n = len(nodes) - 1 # n or number of points - 1

    # Step size

    h = np.zeros(n)

    for j in range(0,n):

        h[j] = nodes[j+1] - nodes[j]
        
    mesh_size = np.max(h)

    # Hat functions

    phi_left = lambda x, x0, h: (x - x0)/h

    phi_right = lambda x, x1, h: (x1 - x)/h

    # Numerical integration -> int_{a}^{b} f*phi dx --- below is written for load vector only

    def simpson_phi(f,phi,h,a,b): 

        if phi == phi_left:

            return ((b-a)/6)*(f(a)*phi(a,a,h) + 4*f((a+b)/2)*phi((a+b)/2,a,h) + f(b)*phi(b,a,h))

        elif phi == phi_right:

            return ((b-a)/6)*(f(a)*phi(a,b,h) + 4*f((a+b)/2)*phi((a+b)/2,b,h) + f(b)*phi(b,b,h))

    def gauss_phi(f,phi,h,a,b):

        if phi == phi_left:

            return ((b-a)/2)*(f((a+b)/2 - (b-a)/(2*math.sqrt(3)))*phi((a+b)/2 - (b-a)/(2*math.sqrt(3)),a,h) + f((a+b)/2 + (b-a)/(2*math.sqrt(3)))*phi((a+b)/2 + (b-a)/(2*math.sqrt(3)),a,h))

        elif phi == phi_right:

            return ((b-a)/2)*(f((a+b)/2 - (b-a)/(2*math.sqrt(3)))*phi((a+b)/2 - (b-a)/(2*math.sqrt(3)),b,h) + f((a+b)/2 + (b-a)/(2*math.sqrt(3)))*phi((a+b)/2 + (b-a)/(2*math.sqrt(3)),b,h))

    # Numerical integration -> int_{a}^{b} f(x) dx --- below is written for stiffness matrix only

    def simpson(f, a, b, args=()):

        return ((b-a)/6)*(f(a,*args) + 4*f((a+b)/2, *args) + f(b, *args))

    def gauss(f, a, b, args=()):

        return ((b-a)/2)*(f((a+b)/2 - (b-a)/(2*math.sqrt(3)), *args) + f((a+b)/2 + (b-a)/(2*math.sqrt(3)), *args))

    # Load vector(F)

    load_vector = np.zeros(n+1)

    load_vector[0] = simpson_phi(g,phi_right,h[0],nodes[0],nodes[1])

    load_vector[n] = 0

    for i in range(0,n-1):

        load_vector[i+1] = simpson_phi(g,phi_left,h[i],nodes[i],nodes[i+1]) + simpson_phi(g,phi_right,h[i+1],nodes[i+1],nodes[i+2])

    # Stiffness matrix(K)

    stiffness_matrix = np.zeros((n+1,n+1))

    k_ii = lambda x, x_j: 1 + (x-x_j)*p(x) + q(x)*(x-x_j)**2 # 1 + (x - x_{i-1})p + (x - x_{i-1})^{2}q

    k_ij = lambda x, x_j, x_i: 1 + (x-x_j)*p(x) + q(x)*(x-x_j)*(x-x_i) # 1 + (x - x_{i-1})p + (x - x_{i-1})(x - x_{i})q

    stiffness_matrix[0,0] = simpson(k_ii, nodes[0], nodes[1], args=(nodes[1],))/(h[0]**2)

    stiffness_matrix[n,n] = 1

    for k in range(1,n): 

        stiffness_matrix[k,k] = simpson(k_ii, nodes[k-1], nodes[k], args=(nodes[k-1],))/(h[k-1]**2) + simpson(k_ii, nodes[k], nodes[k+1], args=(nodes[k+1],))/(h[k]**2)

        stiffness_matrix[k,k-1] = (-1)*simpson(k_ij, nodes[k-1], nodes[k], args=(nodes[k-1],nodes[k],))/(h[k-1]**2)

        stiffness_matrix[k-1,k] = (-1)*simpson(k_ij, nodes[k-1], nodes[k], args=(nodes[k],nodes[k-1],))/(h[k-1]**2)

    # Solve KU = F 

    coefficients = np.linalg.solve(stiffness_matrix, load_vector)

    # FEM solution + particular solution
    
    """
    The FEM solution corresponds to the ODE with homogeneous boundary conditions. Therefore, to obtain a numerical solution 
    for the given ODE with nonhomogeneous boundary conditions, we must add a particular solution to the FEM solution.
    
    """
    def approximation(nodes, coefficients, x):

        for l in range(0,n):

            if x >= nodes[l] and x <= nodes[l+1]:

                return phi_right(x,nodes[l+1],h[l])*coefficients[l] + phi_left(x,nodes[l],h[l])*coefficients[l+1] + particular(x)

    # Exact solution

    def exact(x):

        return np.sin(x)*np.exp(x)

    # Derivative of exact and FEM solutions - these will be used in error analysis
    
    def grad_exact(x):

        return (np.sin(x) + np.cos(x))*np.exp(x)
    
    def grad_approximation(nodes, coefficients, x):

        for l in range(0,n):

            if x >= nodes[l] and x <= nodes[l+1]:

                return (-1/h[l])*coefficients[l] + (1/h[l])*coefficients[l+1] + u_n

    
    return exact, approximation, coefficients, nodes, grad_exact, grad_approximation


def plot_solution(n, xpoints): 
    
    """
    Purpose: This function plots both the exact and FEM solutions for a mesh with n points.
    
    Parameters
    ----------
    n: number of points in the mesh
    
    xpoints: an array of points at which the exact and FEM solutions will be evaluated
    
    """ 
    
    exact, approximation, coefficients, nodes, *_ = FEM(n)
    
    femsol = np.array([approximation(nodes, coefficients, x) for x in xpoints])
    
    # Exact Solution vs FEM Solution plot

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    ax.plot(xpoints,exact(xpoints),"r-",label = "Exact solution")

    ax.plot(xpoints,femsol, 'k--', label = "FEM solution")

    ax.plot(nodes,exact(nodes),'o', mfc='none', color='black', ms = 6, label = "Nodal points")


    #ax.set_title('Exact Solution vs FEM Solution')

    ax.set_xlabel("$x$")

    ax.set_ylabel("$u(x)$, $u_{h}(x)$")

    ax.set_label("d")

    ax.legend()

    #ax1.set_aspect('equal')
    
    #fig.savefig(f'FEM{n}.png', bbox_inches='tight')
    
    plt.show()
    
    return 
    

def error_analysis(n, xpoints):

    """
    Purpose: This function plots the L^2 error, H^1 error, and L^infinity error, and provides the following table:
    
    -----------------------------------------------------------
    N    L2_Error   Rate   H1_Error   Rate   Inf_Error   Rate
    -----------------------------------------------------------
    -      ----      --      ----      --       ---       --
    ------------------------------------------------------------    
    
    Parameters
    ----------
    n: an array containing numbers that represent the number of points in a mesh
    
    xpoints: an array of points, spaced by step size h, at which the exact and FEM solutions will be evaluated
    
    Outputs
    ----------
    L2_error, L2_rate, H1_error, H1_rate, Linfty_error, Linfty_rate
    
    """
    
    L2_error = np.zeros(len(n))
    L2_rate = np.zeros(len(n))
    H1_error = np.zeros(len(n))
    H1_rate = np.zeros(len(n))
    Linfty_error = np.zeros(len(n))
    Linfty_rate = np.zeros(len(n))
    
    for i in range(len(n)):
        
        exact, approximation, coefficients, nodes, grad_exact, grad_approximation = FEM(n[i])
        
        exact_half, approximation_half, coefficients_half, nodes_half, grad_exact_half, grad_approximation_half = FEM(2*n[i]-1)
        
        exactsol, exactsolhalf = exact(xpoints), exact_half(xpoints)
        
        grad_exactsol, grad_exactsolhalf = grad_exact(xpoints), grad_exact_half(xpoints)
    
        femsol = np.array([approximation(nodes, coefficients, x) for x in xpoints])
    
        femsolhalf = np.array([approximation_half(nodes_half, coefficients_half, x) for x in xpoints])
        
        grad_femsol = np.array([grad_approximation(nodes, coefficients, x) for x in xpoints])
    
        grad_femsolhalf = np.array([grad_approximation_half(nodes_half, coefficients_half, x) for x in xpoints])
        
        # L2 error with convergence rate

        L2_error[i] = np.linalg.norm(exactsol - femsol,2)

        L2_errorhalf = np.linalg.norm(exactsolhalf - femsolhalf,2)

        L2_rate[i] = (1/np.log(2))*np.log(L2_error[i]/L2_errorhalf)

        # H1 error with convergence rate

        H1_error[i] = np.sqrt((L2_error[i])**2 + (np.linalg.norm(grad_exactsol - grad_femsol,2))**2)

        H1_errorhalf = np.sqrt((L2_error[i])**2 + (np.linalg.norm(grad_exactsolhalf - grad_femsolhalf,2))**2)

        H1_rate[i] = (1/np.log(2))*np.log(H1_error[i]/H1_errorhalf)

        # L infinity error with convergence rate

        Linfty_error[i] = np.linalg.norm(exactsol - femsol,np.inf)

        Linfty_errorhalf = np.linalg.norm(exactsolhalf - femsolhalf,np.inf)

        Linfty_rate[i] = (1/np.log(2))*np.log(Linfty_error[i]/Linfty_errorhalf)   
        
    # Error plots

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    ax.plot(n, L2_error, "r:", label = "$||e_{h}||_{L^{2}}$", linewidth = 1)
    ax.plot(n, H1_error, "b-.", label = "$||e_{h}||_{H^{1}}$", linewidth = 1)
    ax.plot(n, Linfty_error, "g--", label = "$||e_{h}||_{L^{\infty}}$", linewidth = 1)

    #ax.set_title('Error plots')

    ax.set_xlabel("$n$")

    ax.set_ylabel(r"$||u-u_h||_{L^{2}}$, $||u-u_h||_{H^{1}}$, $||u-u_h||_{L^{\infty}}$")

    ax.set_label("d")

    ax.legend()

    #ax1.set_aspect('equal')
    
    #fig.savefig(f'FEM_error.png', bbox_inches='tight')
    
    plt.show()
    
    # Construction of table
    
    columns = ["$N$", "$||e_{h}||_{L^{2}}$", "$L^2 rate$", "$||e_{h}||_{H^{1}}$", "$H^{1} rate$", "$||e_{h}||_{L^{\infty}}$", "$L^{\infty} rate$"]
        
    column_entries = {
    "$N$": n,
    "$||e_{h}||_{L^{2}}$": L2_error,
    "$L^2 rate$": L2_rate,
    "$||e_{h}||_{H^{1}}$": H1_error,
    "$H^{1} rate$": H1_rate,
    "$||e_{h}||_{L^{\infty}}$": Linfty_error,
    "$L^{\infty} rate$": Linfty_rate        
    }
    
    error_table = pd.DataFrame(column_entries, columns=columns)

    return error_table


#print(plot_solution(4, np.round(np.arange(-1, 2 + 0.1, 0.1), 2)))

#print(plot_solution(8, np.round(np.arange(-1, 2 + 0.1, 0.1), 2)))

#print(error_analysis([4,8,16,32,64,128,256], np.round(np.arange(-1, 2 + 0.1, 0.1), 2)))
