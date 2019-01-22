import numpy as np
import matplotlib.pyplot as plt
import cv2

#convert rgb image to grayscale
def rgb2gray(im_in):
    grayArray = np.array([.299, .587, .114])
    return np.dot(im_in[..., :3], grayArray)

#return mean & std of image neighborhoods of size ***
def grid_stats2(im_in,numGrid):
    #create i,j meshgrid
    dims=im_in.shape
    i = np.floor(np.linspace(0, dims[0], numGrid + 1)).astype(int)
    j = np.floor(np.linspace(0, dims[1], numGrid + 1)).astype(int)
    iv, jv = np.meshgrid(i, j)
    grid_avg = np.zeros((numGrid, numGrid))
    grid_std = np.zeros((numGrid, numGrid))
    # calculate mean in each region
    for a in range(numGrid):
        for b in range(numGrid):
            grid_avg[a, b] = np.mean(im_in[iv[a, b]:iv[a, b + 1], jv[a, b]:jv[a + 1, b]])
            grid_std[a, b] = np.sd(im_in[iv[a, b]:iv[a, b + 1], jv[a, b]:jv[a + 1, b]])
    return grid_avg,grid_std,iv,jv

def get_pupil_mask(im_in,thresh,size):
    # TODO: use connected components and region thresholding to select ONLY largest component (ie pupil)
    _, mask = cv2.threshold(im_in, thresh, 1, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    #perform CLOSE-OPEN morphological operation to remove gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(pup_mask, cv2.MORPH_CLOSE, kernel)
    return mask

def get_bounds(vect_in,diff):
    vect_in_diff = np.absolute(np.convolve(diff, vect_in, 'valid'))
    vect_max = np.argwhere(vect_in_diff == 1)
    return vect_max[0], vect_max[-1]

# estimate pupil radius and position
def get_pupil_geom(im,fig):
    grid_size=20
    im_grid_mean,im_grid_sd,iv,jv=grid_stats2(im,grid_size)
    #find subscript, mean and standard deviation of min avg. intensity
    lowest_sub = np.unravel_index(im_grid_mean.argmin(), (grid_size, grid_size))
    lowest_mean = im_grid_mean.min()
    lowest_sd = im_grid_sd(lowest_sub[0],lowest_sub[1])
    #guess x_cen and y_cen from min intensity grid
    x_cen_guess = iv[lowest_sub[0],lowest_sub[1]]
    y_cen_guess = jv[lowest_sub[0],lowest_sub[1]]
    #compute threshold for pupil from min intensity grid
    thresh=lowest_mean + 1.96 * lowest_sd
    pupil_bin = get_pupil_mask(im_gray,thresh,(5,5))
    diff=[1,-1]
    #step 1 find left and right temporary bounds
    vect_temp = pupil_bin[x_cen_guess, :]
    peaks = get_bounds(vect_temp,diff)
    #step 2 find top and bottom bounds
    y_cen = int((peaks[0]+peaks[1])/2)
    vect_temp = pupil_bin[:,y_cen]
    vert_bounds = get_bounds(vect_temp,diff)
    #step 3 find left and right bounds
    x_cen = int((vert_bounds[0]+vert_bounds[1])/2)
    vect_temp = pupil_bin[x_cen,:]
    horz_bounds = get_bounds(vect_temp,diff)
    r = ((horz_bounds[1]-horz_bounds[0]) + (vert_bounds[1] - vert_bounds[0])) / 4
    
    return x_cen,y_cen,r

def get_iris_geom():
    pass



def main():
    #TODO: user input image path
    path = 'MMUIrisDatabase/MMU Iris Database/2/left/bryanl1.bmp'
    im = cv2.imread(path)
    im_gray = rgb2gray(im)
    r,x_pup,y_pup = get_pupil_geom(im_gray,fig=True)
    get_iris_geom(r,x_pup,y_pup,im_gray,fig=True)

if __name__=='__main__':
    main()