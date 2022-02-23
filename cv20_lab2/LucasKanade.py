import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sc
from scipy.ndimage import zoom

def GaussianKernel(sigma=2):
    n = int((np.ceil(3*sigma) * 2 + 1))
    G_1D = cv2.getGaussianKernel(n, sigma)
    return np.outer(G_1D, G_1D)

def Interpolate(image, d_x, d_y):
    x = np.linspace(0,image.shape[1]-1,image.shape[1])
    y = np.linspace(0,image.shape[0]-1,image.shape[0])
    x_0, y_0 = np.meshgrid(x, y)
    return sc.map_coordinates(image, [np.ravel(y_0 + d_y), np.ravel(x_0 + d_x)], order = 1).reshape(image.shape)

def Lucas_Kanade(I1, I2, d_x0, d_y0, rho=1.5, epsilon=0.05, reps=10, plot_and_save=False, index=1, saveFolder=''):
    G_rho = GaussianKernel(rho)
    d_x = d_x0
    d_y = d_y0
    
    for i in range(reps):
        I1_d = Interpolate(I1, d_x, d_y)
        E = I2 - I1_d

        # A matrix
        I1_dx, I1_dy = np.gradient(I1)
        A1 = Interpolate(I1_dx, d_x, d_y)
        A2 = Interpolate(I1_dy, d_x, d_y)

        # M * u = N
        M_xx = cv2.filter2D(A1**2, -1, G_rho) + epsilon
        M_xy = cv2.filter2D(A1*A2, -1, G_rho)
        M_yy = cv2.filter2D(A2**2, -1, G_rho) + epsilon

        N_x = cv2.filter2D(A1*E, -1, G_rho)
        N_y = cv2.filter2D(A2*E, -1, G_rho)

        # Cramer for solving system
        D = M_xx * M_yy - M_xy**2
        D_x = N_x * M_yy - N_y * M_xy
        D_y = N_y * M_xx - N_x * M_xy

        u_x = D_x / D
        u_y = D_y / D

        d_x = d_x + u_x
        d_y = d_y + u_y
    
    if(plot_and_save):
        d_x_r = cv2.resize(d_x, None, fx=0.3, fy=0.3)
        d_y_r = cv2.resize(d_y, None, fx=0.3, fy=0.3)
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.quiver(-d_x_r, -d_y_r, angles='xy')
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.gca().invert_yaxis()
        plt.savefig(saveFolder+'OpticalFlow'+str(index)+'.png', dpi=200)
        
    return d_x, d_y

def Lucas_Kanade_Multiscale(I1, I2, d_x0, d_y0, rho=1.5, epsilon=0.05, reps=10, N=5, plot_and_save=False, index=1, saveFolder=''):
    d_x = d_x0
    d_y = d_y0
    scale = 1 / 2**(N-1)
    for i in range(N):
        dims_I1 = (int(I1.shape[1]*scale), int(I1.shape[0]*scale))
        dims_I2 = (int(I2.shape[1]*scale), int(I2.shape[0]*scale))
        I1_r = cv2.resize(I1, dims_I1, fx=scale, fy=scale)
        I2_r = cv2.resize(I2, dims_I2, fx=scale, fy=scale)
        d_x_r = cv2.resize(d_x, dims_I1, fx=scale, fy=scale)
        d_y_r = cv2.resize(d_y, dims_I1, fx=scale, fy=scale)     
        d_x, d_y = Lucas_Kanade(I1_r, I2_r, d_x_r, d_y_r, rho, epsilon, reps, plot_and_save, index, saveFolder)
        d_x = zoom(d_x,2)
        d_y = zoom(d_y,2)
        scale *= 2
    return d_x, d_y
        
def displ(d_x, d_y, threshold=1.):
    counter=0
    sum_x=0
    sum_y=0
    for i in range(d_x.shape[0]):
        for j in range(d_y.shape[1]):
            if(d_x[i,j]**2 + d_y[i,j]**2 > threshold):
                counter += 1
                sum_x += d_x[i,j]
                sum_y += d_y[i,j]
    #print(str(counter)+" out of "+str(d_x.shape[0]*d_x.shape[1])+" are above threshold value.")
    if(counter==0):
        counter=1
    return int(sum_x/counter), int(sum_y/counter)

def NewBoundingBox(displ_x, displ_y, box):
    new_box = [0,0,box[2], box[3]]
    new_box[0] = round(box[0] - displ_y)
    new_box[1] = round(box[1] - displ_x)
    
    return new_box

def SaveFrame(frame, box, title, saveFolder='Results/'):    
    start_point=(box[0],box[1])
    end_point=(box[0]+box[2],box[1]+box[3])
    color=(255,0,0)
    thickness=2
    box_frame = cv2.rectangle(frame, start_point, end_point, color, thickness) 
    
    _, ax = plt.subplots()

    ax.set_aspect('equal')
    ax.imshow(cv2.cvtColor(box_frame, cv2.COLOR_BGR2RGB))
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    
    plt.title(title, fontsize=8)
    plt.savefig(saveFolder+title+".png", dpi=200)
    
#box = [x,y,width,height]
def DetectItem(box, rho=1.5, epsilon=0.05, threshold=5, reps=10, multiscale = False, N=5, pngFolder='skinSamples/', frames=66, saveFolder=''):
    current_box = box
    for i in range(1,frames):
        
        # Frame paths
        I1_path = pngFolder+str(i)+'.png'
        I2_path = pngFolder+str(i+1)+'.png'
        
        # Read frames
        I1_col = cv2.imread(I1_path, cv2.IMREAD_COLOR)
        I2_col = cv2.imread(I2_path, cv2.IMREAD_COLOR)

        # Convert to gray scale
        I1 = cv2.cvtColor(I1_col, cv2.COLOR_BGR2GRAY)
        I2 = cv2.cvtColor(I2_col, cv2.COLOR_BGR2GRAY)

        # Normalize to [0,1]
        I1 = I1.astype(np.float)/255
        I2 = I2.astype(np.float)/255
        
        # Bounding boxes
        I1_item = I1[current_box[1]:(current_box[1]+current_box[3]), current_box[0]:(current_box[0]+current_box[2])]
        I2_item = I2[current_box[1]:(current_box[1]+current_box[3]), current_box[0]:(current_box[0]+current_box[2])]
        
        # Lucas - Kanade between 2 frames
        d_x0 = np.zeros(I1_item.shape)
        d_y0 = np.zeros(I1_item.shape)
        if(multiscale):
            d_x, d_y = Lucas_Kanade_Multiscale(I1_item, I2_item, d_x0, d_y0, rho, epsilon, reps, N)            
        else:
            d_x, d_y = Lucas_Kanade(I1_item, I2_item, d_x0, d_y0, rho, epsilon, reps, True, i+1, saveFolder)
        
        # Find item displacement
        displ_x, displ_y = displ(d_x, d_y, threshold)
        
        # New box coordinates
        new_box = NewBoundingBox(displ_x, displ_y, current_box)        
        current_box = new_box
        
        # Save new frame with the new bounding box
        SaveFrame(I2_col, new_box, str(i+1), saveFolder) 