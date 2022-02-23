#!/usr/bin/env python
# coding: utf-8

# In[1]:


def edgedetection(image,sigma,theta_edge,lapl_type):
    n = int(2*np.ceil(3*sigma)+1) #kernel size
    gauss1D = cv2.getGaussianKernel(n, sigma) # Column vector
    gauss2D = gauss1D @ gauss1D.T # Symmetric gaussian kernel
    kern = np.array([
            [0,1,0],
            [1,1,1],
            [0,1,0]
        ], dtype=np.uint8)  # B 
    Is = cv2.filter2D(image,-1,gauss2D)
    # 1.2.2
    if lapl_type == 'Linear':
        #create LoG kernel we find the second order derivative of Gaussian
        #x = y = np.linspace(-n/2,n/2,n)
        #xv, yv = np.meshgrid(x, y)
        #Gxx = (xv**2)/(2*np.pi*pow(sigma,4)) * gauss2D
        #Gyy = (yv**2)/(2*np.pi*pow(sigma,4)) * gauss2D
        #LoG = Gxx + Gyy
        LoG = cv2.Laplacian(gauss2D,cv2.CV_64F)
        L = cv2.filter2D(image, -1, LoG)
    else:
        dilated_img = cv2.dilate(Is, kern)
        eroded_img = cv2.erode(Is, kern)
        L = dilated_img + eroded_img - 2 * Is # Non linear convolution
        
    #1.2.3
    
    _, X = cv2.threshold(L, 0, 1, cv2.THRESH_BINARY)
    dilated_X = cv2.dilate(X, kern)
    eroded_X = cv2.erode(X, kern)
    Y = dilated_X - eroded_X
    #print
    #plt.figure()
    #plt.title('zero-crossings image')
    #plt.imshow(Y,cmap='gray')
    # 1.2.4
    [fx,fy] = np.gradient(Is)
    gradientIs = np.sqrt(fx**2 + fy**2)
    max_of_grad = gradientIs.max()      
    D = (Y == 1) & (gradientIs > theta_edge*max_of_grad)
    return D
       
    

