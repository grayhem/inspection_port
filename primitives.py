# pylint: disable=E0401, E1101

"""
very simple composable effects and filters in the image domain
"""



from skimage import feature
from skimage import morphology
from skimage import filters

import cv2
import numpy as np


import trees

def neg_tree(frame):
    """
    draw a tree in negative
    """
    tree = trees.tree_edges(frame)
    tree = morphology.binary_dilation(tree)
    return color_mask(frame, tree)

def line_tree_thresh(frame):
    """
    draw the edges of a quadtree and then run adaptive thresholding
    """
    frame = neg_tree(frame)
    return thresh_colors(frame)


def draw_tree(frame):
    """
    draw the edges of a quadtree on the frame
    """
    tree = trees.tree_edges(frame)
    tree = morphology.binary_dilation(tree)
    return color_mask(frame, np.logical_not(tree))

def draw_gray_tree(frame):
    """
    use a grayscale copy of the frame to draw a quadtree on the original frame
    """
    tree = trees.tree_edges(grayscale(frame))
    tree = morphology.binary_dilation(tree)
    return color_mask(frame, np.logical_not(tree))



#---------------------------------------------------------------------------------------------------

# input to all of these is expected to be an rgb frame




def threshold_mask(frame, block_size=7):
    """
    use the adaptive threshold to generate a mask for an image.
    """
    return filters.threshold_adaptive(grayscale(frame), block_size=block_size)


def grayscale(frame):
    """
    convert an rgb frame to grayscale
    """

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#-----------------------------------

def median(frame):
    """
    convert an rgb frame into grayscale and replace pixels brighter
    than the median with white, and dimmer with black
    """
    frame = grayscale(frame)
    median_intensity = np.median(frame)
    frame = (frame > median_intensity).astype(np.uint8) * 255
    return frame

#-----------------------------------

def median_rows(frame):
    """
    same as median, but row-wise. because faster than col-wise.
    """
    frame = grayscale(frame)
    median_intensity = np.median(frame, 1).reshape(-1, 1)
    frame = (frame > median_intensity).astype(np.uint8) * 255
    return frame

#-----------------------------------

def rgb_max(frame):
    """
    for each pixel in the frame max out the color channel with the highest value and
    and zero out the other color channels.
    """
    pass

#-----------------------------------

def rgb_distance(frame):
    """
    normalize the euclidean norm between each RGB vector and the mean RGB
    vector to the range [0, 255].
    """

    # diff from mean
    diff = difference_from_mean(frame)
    # norm
    frame = np.linalg.norm(diff, axis=2).astype(np.float32)
    # normalize
    return normalize(frame)

#-----------------------------------

def normalize(frame):
    """
    normalize the frame to [0, 255]. preserves input shape
    """

    # minimum bound
    frame = frame - frame.min()
    # maximum bound
    frame = frame / frame.max()
    # [0, 255]
    return (frame * 255).astype(np.uint8)


#-----------------------------------

def difference_from_mean(frame):
    """
    difference vector between each pixel color and the mean. returns RGB.
    """
    mean_rgb = frame.reshape(-1, 3).mean(0)
    return np.abs(frame - mean_rgb).astype(np.uint8)

def normal_difference(frame):
    """
    difference from mean but normalized because i don't have function composition yet
    """

    diff = difference_from_mean(frame)
    return normalize(diff)

#-----------------------------------

def identity(frame):
    """
    single-arg identity function
    """
    return frame

#-----------------------------------

def hog(frame):
    """
    compute the histogram of oriented gradients and return the image output
    """
    frame = grayscale(frame)
    _, frame = feature.hog(frame, visualise=True)

    return frame

#-----------------------------------

def canny(frame):
    """
    canny edge detection
    """
    frame = feature.canny(grayscale(frame)).astype(np.uint8)
    return frame*255

#-----------------------------------

def gabor(frame, frequency=0.2, real=True):
    """
    return real or imaginary response to the gabor filter
    """

    real_frame, imag_frame = filters.gabor(grayscale(frame), frequency)

    if real:
        return real_frame
    else:
        return imag_frame

#-----------------------------------

def adaptive_threshold(frame):
    """
    apply adaptive thresholding. grayscale output.
    """
    block_size = 7
    frame = filters.threshold_adaptive(grayscale(frame), block_size=block_size)
    return frame.astype(np.uint8)*255

#-----------------------------------

def black_outlines(frame, sigma=1.75):
    """
    draw black outlines on the image wherever we detect edges
    """

    # get the edges
    edges = not_edges(frame, sigma=sigma)

    # and zero out the corresponding pixels
    frame = color_mask(frame, edges)    

    return frame

def not_edges(frame, sigma=1.75):
    """
    generate a boolean mask of negatives of dilated edges.
    """

    # find edges
    edges = feature.canny(grayscale(frame), sigma=sigma)

    # dilate them
    edges = morphology.binary_dilation(edges)

    # now invert
    return np.logical_not(edges)

def color_mask(frame, mask):
    """
    use a boolean mask to zero out pixels in an RGB frame
    """
    frame_shape = frame.shape
    frame = (frame.reshape(-1, 3)*mask.reshape(-1, 1)).reshape(frame_shape)
    return frame

def smooth_scale(frame, sigma=2, scale=3):
    """
    apply a gaussian smoothing filter and then scale by given factor
    """
    frame = filters.gaussian(frame, sigma=sigma, multichannel=True)
    return (frame*scale).astype(np.uint8)

def thresh_colors(frame, block_size=7):
    """
    generate adaptive thresholds and use them to mask a color image
    """

    # adaptive thresholding
    mask = threshold_mask(frame, block_size=block_size)

    # apply mask
    frame = color_mask(frame, mask)

    return frame


