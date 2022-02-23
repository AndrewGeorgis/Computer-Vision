import cv2
import numpy as np
import matplotlib.pyplot as plt
from CV20_lab1_Part2 import cv20_lab1_part2_utils as ut

#---------------------------------------------------------- GENERAL --------------------------------------------------------

def GaussianKernel(sigma=2):
    n = int((np.ceil(3*sigma) * 2 + 1))
    G_1D = cv2.getGaussianKernel(n, sigma)
    return np.outer(G_1D, G_1D)

def LoG_metric(point, Lxx, Lyy):  # point = [x, y, sigma]
    x = int(point[0])
    y = int(point[1])
    sigma = point[2]
    return (sigma**2) * np.absolute(Lxx[x, y] + Lyy[x, y])

# ----------------------------------------------------- HARRIS - STEPHENS --------------------------------------------------

def Harris_Stephens (Im, sigma, rho, k, theta, save_png=False):

    # Convert to gray scale
    # Normalize to [0,1]
    #i_norm = Im.astype(np.float)/255
    
    # Gaussian kernels
    G_sigma = GaussianKernel(sigma)
    G_rho = GaussianKernel(rho)
    
    # Smoothing
    Is = cv2.filter2D(Im, -1, G_sigma)
    
    # Gradients
    dIs_dx , dIs_dy = np.gradient(Is)
    dIs_dxdx = dIs_dx**2
    dIs_dxdy = dIs_dx*dIs_dy
    dIs_dydy = dIs_dx**2
    
    # Structure tensor
    J1 = cv2.filter2D(dIs_dxdx, -1, G_rho)
    J2 = cv2.filter2D(dIs_dxdy, -1, G_rho)
    J3 = cv2.filter2D(dIs_dydy, -1, G_rho)
    
    #λ+
    lamda_p = 0.5 * ( J1 + J3 + np.sqrt( (J1-J3)**2 + 4 * J2**2 ) )   

    #λ-
    lamda_n = 0.5 * ( J1 + J3 - np.sqrt( (J1-J3)**2 + 4 * J2**2 ) )
    
#     title = "lamda- eigenvalues for picture " + image_name
#     plt.title(title, fontsize=8)
#     plt.imshow(lamda_n, cmap = "gray_r")
#     plt.savefig(title+".png", dpi=200)

    # Cornerness criterion
    R = np.multiply(lamda_p, lamda_n) - k * (lamda_p + lamda_n)**2

    # Condition 1
    ns = np.ceil(3*sigma)*2 + 1
    B_sq = ut.disk_strel(ns)
    Cond1 = (R==cv2.dilate(R,B_sq))

    # Condition 2
    Cond2 = (R > theta*R.max())

    # Edges
    are_edges = np.logical_and(Cond1, Cond2)
    edges = np.empty([0,3])
    for i in range(are_edges.shape[0]):
        for j in range(are_edges.shape[1]):
            if are_edges[i,j]: 
                edges = np.append(edges, [[j,i,sigma]], axis=0)
    
    # Plot edges
    #my_title = 'Harris_Stephens of ' + image_name +', sigma=' + str(sigma) +', rho=' + str(rho) + ', k=' + str(k) + ', theta=' + str(theta)
    #ut.interest_points_visualization(color_image, edges, ax=None, title=my_title, save_png=save_png)
        
    return edges

# ----------------------------------------------------- HARRIS - LAPLACE --------------------------------------------------

def Harris_Laplace (Im, sigma, rho, k, theta, s, N, save_png=False):
    
    # Normalize to [0,1]
    #i_norm = Im.astype(np.float)/255
    
    # Sigma and rho values
    sigmas = [(s**i) * sigma for i in range(N)]
    rhos = [(s**i) * rho for i in range(N)]
    
    # Calculate Lxx, Lyy for all sigma values
    Lxx = {}
    Lyy = {}
    for i in range(N):
        # Gaussian kernel
        G = GaussianKernel(sigmas[i])
        Is = cv2.filter2D(Im, -1, G)
        # Lxx, Lyy
        dIs_dx, dIs_dy = np.gradient(Is)
        Lxx[sigmas[i]], _ = np.gradient(dIs_dx)
        _, Lyy[sigmas[i]] = np.gradient(dIs_dy)
    
    # Harris-Stephens edges
    HS_edges = {} 
    for i in range(N):
        HS_edges[sigmas[i]] = Harris_Stephens(Im, sigma = sigmas[i], rho = rhos[i], k=k, theta=theta)  
    
    # Harris-Laplace edges : eliminate HS_edges that dont satisfy the LoG metric restricton
    HL_edges = np.empty([0,3])    
    for i in range(0,N):
        edges = HS_edges[sigmas[i]]
        for j in range(edges.shape[0]):
            point = edges[j]
            point_T = [edges[j][1], edges[j][0], edges[j][2]]
            
            metric = LoG_metric(point_T, Lxx[point_T[2]], Lyy[point_T[2]])

            metric_l = 0
            if i-1 >= 0:
                point_l = (point_T[0], point_T[1], sigmas[i-1])
                metric_l = LoG_metric(point_l, Lxx[point_l[2]], Lyy[point_l[2]])

            metric_r = 0
            if i+1 < N:
                point_r = (point_T[0], point_T[1], sigmas[i+1])
                metric_r = LoG_metric(point_r, Lxx[point_r[2]], Lyy[point_r[2]])

            if metric > metric_r and metric > metric_l:
                HL_edges = np.append(HL_edges, [point], axis=0)
    
    # Plot results
    #my_title = 'Harris_Laplace of ' + image_name +', sigma=' + str(sigma) +', rho=' + str(rho) + ', k=' + str(k) +\
    #', theta=' + str(theta) + ', s=' + str(s) + ', N=' + str(N)
    #ut.interest_points_visualization(color_image, HL_edges, ax=None, title=my_title, save_png=save_png)
    
    return HL_edges