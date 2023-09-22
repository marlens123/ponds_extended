import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image
from scipy.signal import argrelmax



def kmeans_image(img, save_path=None, mode='rgb', k=3, method='sklearn', init=False, init_values=None, return_mode='numpy'):
    """
    Performs kmeans algorithm on given image and saves it to given folder.

    Parameters:
    -----------
        img : 
            Image to cluster
        save_path : str
            path where result should be saved. Must contain image name and extension (.png, .jpg...)
        mode : str
            either 'gray' or 'rgb'
        k : int
            number k of clusters
        method : str
            either 'sklearn' or 'cv2'
        init : Bool
            if cluster centers should be initialized
        init_values : tuple
            centroid intitialization values
        return_mode : str
            whether to return as numpy array ('numpy') or PIL Image ('pil')
    
    Returns:
    --------
        pil, labels: PIL.Image, numpy.ndarray
            resulting image and labels (for merging classes)
    
    """
    # reshape image to 2D array of pixels and grayscale value
    if mode == 'gray':
        pixel_values = img.reshape((-1, 1))

    # reshape image to 2D array of pixels and 3 color values
    elif mode == 'rgb':    
        pixel_values = img.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
    
    else:
        'Input mode must be either "gray" or "rgb"'

    # CV2 implementation: Random centroid initialization
    if(method=='cv2'):
        # define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Sklearn implementation
    elif(method=='sklearn'):
        # kmeans with defined centroid initialization
        if(init):
            kmeans = KMeans(n_clusters=k, init=init_values, n_init=1).fit(pixel_values)
        # kmeans with random initialization
        else:
            kmeans = KMeans(n_clusters=k).fit(pixel_values)
        
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

    # convert back to 8 bit values and flatten labels array
    centers = np.uint8(centers)
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image shape and convert to PIL for saving
    segmented_image = segmented_image.reshape((img.shape))

    if not save_path==None:
        cv2.imwrite(save_path, segmented_image)

    if return_mode == 'numpy':
        return segmented_image, labels
    
    elif return_mode == 'pil':
        pil = Image.fromarray(segmented_image)
        print(pil.mode)
        return pil, labels
    
    else: 
        print('return_mode must be one of "numpy" or "pil"')

    """
    if mode == 'gray':
        pil.convert('L')

    if not save_path==None:
        pil.save(save_path)

    return pil, labels
    """


def most_occuring_pixel_values(histogram, num=3):
    """
    Find the most occuring pixel values of a histogram retrieved from grayscale image.

    Parameters:
    ----------
        histogram: numpy.ndarray
            histogram showing distribution of grayscale pixels of image
    
    Returns:
        most_three: list
            the three most occuring pixel values, in ascending order
    """
    # Get the local max x-values
    local_max = argrelmax(histogram)
    x_local_max = local_max[0]
    # Get the y-values of the local max
    y_local_max = histogram[x_local_max]

    #########################################################

    # sort the local maxima in ascending order
    y_sorted = np.sort(y_local_max)
    max_list = []
    
    for i in range(num + 1):
        # y value local maximum
        y = y_sorted[-i]
        # x value local maximum
        idx = np.where(y_local_max == y)[0]
        x = x_local_max[idx][0]
        max_list.append(x)

    return max_list



def convert_to_centroids(pixel_vals):
    """
    Converts pixel values to kmeans centroids.

    Parameters:
    -----------
        pixel_vals : list
            the pixel values to be converted
    
    Returns:
    -------
        centroids : list
    """
    
    centroids = []
    for i in pixel_vals:
        centroids.append([i/255])
    
    return centroids
