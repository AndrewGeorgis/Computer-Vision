import cv2
import numpy as np
import matplotlib.pyplot as plt
from CV20_lab1_Part2 import cv20_lab1_part2_utils as ut

#---------------------------------------------------------- GENERAL -----------------------------------------------------------

def GaussianKernel(sigma=2):
    n = int((np.ceil(3*sigma) * 2 + 1))
    G_1D = cv2.getGaussianKernel(n, sigma)
    return np.outer(G_1D, G_1D)

def LoG_metric(point, Lxx, Lyy):  # point = [x, y, sigma]
    x = int(point[0])
    y = int(point[1])
    sigma = point[2]
    return (sigma**2) * np.absolute(Lxx[x, y] + Lyy[x, y])

# ---------------------------------------------- SINGLESCALE BLOB DETECTOR ----------------------------------------------------

def SingleScale_Blobs (Im, sigma, theta, approx=False, save_png=False):
    
    # Read image
    
    
    # Convert to gray scale
    # i_gray = cv2.cvtColor(image_name, cv2.COLOR_BGR2GRAY)
    
    # Normalize to [0,1]
    #i_norm = Im.astype(np.float)/255
    
    # Creation of Gaussian kernel
    G = GaussianKernel(sigma)
    
    # Smoothing
    Is = cv2.filter2D(Im, -1, G)
    
    if not approx:
        dIs_dx, dIs_dy = np.gradient(Is)
        Lxx, Lxy = np.gradient(dIs_dx)
        Lxy, Lyy = np.gradient(dIs_dy)
    else:
        Lxx,Lyy,Lxy = approx_Hessian(Is, sigma)
    
    # Importance criterion
    if not approx:
        R = Lxx*Lyy - Lxy**2
    else:
        R = Lxx*Lyy - (0.9*Lxy)**2    

    # Condition 1
    ns = np.ceil(3*sigma)*2 + 1
    B_sq = ut.disk_strel(ns)
    Cond1 = (R==cv2.dilate(R,B_sq))

    # Condition 2
    Cond2 = (R > theta*R.max())

    # Blobs
    are_blobs = np.logical_and(Cond1, Cond2)

    blobs = np.empty([0,3])
    for i in range(are_blobs.shape[0]):
        for j in range(are_blobs.shape[1]):
            if are_blobs[i,j]: 
                blobs = np.append(blobs, [[j,i,sigma]], axis=0)
                
    # Plot blobs
    if not approx:
        using_approximation = ' '
    else:
        using_approximation = ' (Hessian approximation) '
    #my_title = 'Blob_Detection' + using_approximation + 'of ' + image_name +', sigma=' + str(sigma) + ', theta=' + str(theta)
    #ut.interest_points_visualization(color_image, blobs, ax=None, title=my_title, save_png=save_png)
    
    return blobs

#--------------------------------------------------------- MULTISCALE BLOB DETECTOR ----------------------------------------------

def MultiScale_Blobs (Im, sigma=2, theta=0.2, s=1.5, N=4, approx=False, save_png=False):
    
    # Read image
    
    
    # Convert to gray scale
    # i_gray = cv2.cvtColor(image_name, cv2.COLOR_BGR2GRAY)
    
    # Normalize to [0,1]
    #i_norm = Im.astype(np.float)/255
    
    # Sigma values
    sigmas = [(s**i) * sigma for i in range(N)]
    
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
    
    # Hessian blobs
    blobs = {} 
    for i in range(N):
        blobs[sigmas[i]] = SingleScale_Blobs(Im, sigma=sigmas[i], theta=theta, approx=approx)

    # Multichannel blobs : eliminate blobs that dont satisfy the LoG metric restricton
    multi_blobs = np.empty([0,3])    
    for i in range(0, N):
        i_blobs = blobs[sigmas[i]]
        
        for j in range(i_blobs.shape[0]):
            point = i_blobs[j]
            point_T = [i_blobs[j][1], i_blobs[j][0], i_blobs[j][2]]
            
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
                multi_blobs = np.append(multi_blobs, [point], axis=0)
    
    # Plot results
    if not approx:
        using_approximation = ' '
    else:
        using_approximation = ' (Hessian approximation) '
    #my_title = 'Multiscale_Blob_Detection' + using_approximation + 'of ' + image_name +', sigma=' + str(sigma) + ', theta=' + str(theta)\
   # + ', s=' + str(s) + ', N=' + str(N)
    #ut.interest_points_visualization(color_image, multi_blobs, ax=None, title=my_title, save_png=save_png)
    
    return multi_blobs

#------------------------------------------- HESSIAN APPROXIMATION ------------------------------------------------------------------

def approx_Hessian(Is, sigma):
    
    Dxx = np.zeros(Is.shape)
    Dyy = np.zeros(Is.shape)
    Dxy = np.zeros(Is.shape)
    
    # Second derivative approximation kernel size
    n = int((np.ceil(3*sigma) * 2 + 1))

    # Padding
    n = int((np.ceil(3*sigma) * 2 + 1))
    I_pad = np.pad(Is, pad_width=n, mode='symmetric')

    # Calculate Integral Image
    temp = np.cumsum(I_pad, axis=0)
    S = np.cumsum(temp, axis=1)

    # Calculate Dxx:

    h = int(4*np.floor(n/6) + 1)
    w = int(2*np.floor(n/6) + 1)

    # Get kernel starting potition and weights ([n,n] is the starting position, after padding)
    boxes, weights = get_box_filter([n,n], h, w, 'Dxx')

    for y in range(Is.shape[1]):
        for x in range(Is.shape[0]):
            for b in range(weights.size):
                Dxx[x,y] += weights[b] * (S[boxes[b]['A'][0]+x, boxes[b]['A'][1]+y] + S[boxes[b]['C'][0]+x, boxes[b]['C'][1]+y] \
                                         - S[boxes[b]['B'][0]+x, boxes[b]['B'][1]+y] - S[boxes[b]['D'][0]+x, boxes[b]['D'][1]+y])

    # Calculate Dyy:

    h = int(2*np.floor(n/6) + 1)
    w = int(4*np.floor(n/6) + 1)

    # Get kernel starting potition and weights ([n,n] is the starting position, after padding)
    boxes, weights = get_box_filter([n,n], h, w, 'Dyy')

    for y in range(Is.shape[1]):
        for x in range(Is.shape[0]):
            for b in range(weights.size):
                Dyy[x,y] += weights[b] * (S[boxes[b]['A'][0]+x, boxes[b]['A'][1]+y] + S[boxes[b]['C'][0]+x, boxes[b]['C'][1]+y] \
                                         - S[boxes[b]['B'][0]+x, boxes[b]['B'][1]+y] - S[boxes[b]['D'][0]+x, boxes[b]['D'][1]+y])

    # Calculate Dxy:

    h = int(2*np.floor(n/6) + 1)

    # Get kernel starting potition and weights ([n,n] is the starting position, after padding)
    boxes, weights = get_box_filter([n,n], h, 0, 'Dxy')

    for y in range(Is.shape[1]):
        for x in range(Is.shape[0]):
            for b in range(weights.size):
                Dxy[x,y] += weights[b] * (S[boxes[b]['A'][0]+x, boxes[b]['A'][1]+y] + S[boxes[b]['C'][0]+x, boxes[b]['C'][1]+y] \
                                         - S[boxes[b]['B'][0]+x, boxes[b]['B'][1]+y] - S[boxes[b]['D'][0]+x, boxes[b]['D'][1]+y])
             
             
    return Dxx, Dyy, Dxy

#---------------------------------------------------------------------------------------------------------------------------------------

def get_box_filter(point, h, w, kernel_type = 'Dxx'):

    [x, y] = point
    boxes = {}
    weights = []

    if kernel_type=='Dxx':
        
        weights = np.array([1, -3])
        
        # External box coordinates (weight = 1)
        ext_box = {}
        ext_box['A'] = np.array([x - (3*w)//2, y - h//2]) 
        ext_box['B'] = np.array([x + (3*w)//2, y - h//2])
        ext_box['C'] = np.array([x + (3*w)//2, y + h//2]) 
        ext_box['D'] = np.array([x - (3*w)//2, y + h//2]) 
        boxes[0] = ext_box
        
        # Internal box coordinates (weight = -3)
        int_box = {}
        int_box['A'] = np.array([x - w//2, y - h//2]) 
        int_box['B'] = np.array([x + w//2, y - h//2])
        int_box['C'] = np.array([x + w//2, y + h//2]) 
        int_box['D'] = np.array([x - w//2, y + h//2])
        boxes[1] = int_box

    elif kernel_type=='Dyy':
        
        weights = np.array([1, -3])

        # External box coordinates (weight = 1)
        ext_box = {}
        ext_box['A'] = np.array([x - w//2, y - (3*h)//2]) 
        ext_box['B'] = np.array([x + w//2, y - (3*h)//2])
        ext_box['C'] = np.array([x + w//2, y + (3*h)//2]) 
        ext_box['D'] = np.array([x - w//2, y + (3*h)//2]) 
        boxes[0] = ext_box
        
        # Internal box coordinates (weight = -3)
        int_box = {}
        int_box['A'] = np.array([x - w//2, y - h//2]) 
        int_box['B'] = np.array([x + w//2, y - h//2])
        int_box['C'] = np.array([x + w//2, y + h//2]) 
        int_box['D'] = np.array([x - w//2, y + h//2])
        boxes[1] = int_box
        
    elif kernel_type=='Dxy': # (h=w, ignore w)

        weights = np.array([1, -1, 1, -1])  #clockwise from box top left to box bottom left
        
        k = h - 1
        
        # Box 0
        box0 = {}
        box0['A'] = np.array([x-k-1, y-k-1])
        box0['B'] = np.array([x-1, y-k-1])
        box0['C'] = np.array([x-1, y-1]) 
        box0['D'] = np.array([x-k-1, y-1])
        boxes[0] = box0
        
        # Box 1
        box1 = {}
        box1['A'] = np.array([x+1, y-k-1])
        box1['B'] = np.array([x+k+1, y-k-1])
        box1['C'] = np.array([x+k+1, y-1])
        box1['D'] = np.array([x+1, y-1])
        boxes[1] = box1
        
        # Box 2
        box2 = {}
        box2['A'] = np.array([x+1, y+1]) 
        box2['B'] = np.array([x+k+1, y+1])
        box2['C'] = np.array([x+k+1, y+k+1])
        box2['D'] = np.array([x+1, y+k+1])
        boxes[2] = box2
        
        # Box 3
        box3 = {}
        box3['A'] = np.array([x-k-1, y+1])
        box3['B'] = np.array([x-1, y+1]) 
        box3['C'] = np.array([x-1, y+k+1])
        box3['D'] = np.array([x-k-1, y+k+1])
        boxes[3] = box3
        
    return boxes, weights