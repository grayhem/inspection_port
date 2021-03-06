# pylint: disable=E0401, E1101, C0103

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

def draw_corner_tree(frame):
    """
    use a grayscale copy of the frame to draw a quadtree and block out upper left corners of nodes
    """
    tree = trees.tree_corners(grayscale(frame))
    # tree = morphology.skeletonize(tree)
    # tree = morphology.binary_dilation(tree)
    return color_mask(frame, np.logical_not(tree))

def draw_dot_tree(frame):
    """
    use a grayscale copy of the frame to draw a quadtree and put a dot at centers of nodes
    """
    tree = trees.tree_dots(grayscale(frame))
    selem = morphology.diamond(4, dtype=np.bool)
    tree = morphology.binary_dilation(tree, selem=selem)
    return color_mask(frame, np.logical_not(tree))

def build_skeleton(frame):
    """
    build a corner tree, skeletonize, dilate
    """
    tree = trees.tree_corners(frame)
    tree = morphology.skeletonize(tree)
    # tree = morphology.binary_dilation(tree)
    morphology.remove_small_objects(tree, min_size=20, connectivity=2, in_place=True)
    tree = morphology.binary_dilation(tree)
    return tree


def sobel(frame):
    """
    return the sobel importance of an rgb image
    """
    frame = grayscale(frame)
    frame = filters.sobel(frame)
    # print(frame.max())
    return normalize(frame)

def sobel_hv(frame):
    """
    compute horizontal/ vertical sobel intensities and convert to red/ blue values. green channel
    will be left zero.
    """
    output = np.zeros(frame.shape, dtype=np.uint8)
    frame = grayscale(frame)
    # dilation doesn't really improve much
    # morphology.dilation(normalize(np.abs(filters.sobel_h(frame))), out=output[:, :, 0])
    # morphology.dilation(normalize(np.abs(filters.sobel_v(frame))), out=output[:, :, 2])
    output[:, :, 0] = normalize(np.abs(filters.sobel_h(frame)))
    output[:, :, 2] = normalize(np.abs(filters.sobel_v(frame)))
    return output

def sobel_triple(frame):
    """
    compute horizontal/ vertical sobel intensities and convert to red/ blue values. green channel
    will get the un-directed sobel filter. very pleasing effect.
    """
    output = np.zeros(frame.shape, dtype=np.uint8)
    frame = grayscale(frame)
    output[:, :, 0] = normalize(np.abs(filters.sobel_h(frame)))
    output[:, :, 1] = normalize(filters.sobel(frame))
    output[:, :, 2] = normalize(np.abs(filters.sobel_v(frame)))
    return output


# how about box counting or some pyramid thing for a local fractal dimension
#---------------------------------------------------------------------------------------------------

# input to all of these is expected to be an rgb frame

def reflect_101(frame, border=75):
    """
    pad the frame with a reflection that doesn't duplicate the element on the original edge of the
    image
    """
    return cv2.copyMakeBorder(
        frame,
        border,
        border,
        border,
        border,
        cv2.BORDER_REFLECT_101)

def resize(frame, fx=1.5, fy=1.5):
    """
    scale the frame
    """
    return cv2.resize(frame, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)



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
    normalize the frame to [0, 255]. preserves input shape.
    """

    # minimum bound
    frame -= frame.min()
    # maximum bound
    frame /= frame.max()
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

def adaptive_threshold_mask(frame):
    """
    adaptive thresholding with boolean output
    """
    block_size = 15
    frame = filters.threshold_adaptive(grayscale(frame), block_size=block_size)
    return frame

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
    use a boolean mask to zero out pixels in an RGB or grayscale frame
    """
    frame_shape = frame.shape
    if len(frame_shape) == 3:
        frame = (frame.reshape(-1, 3)*mask.reshape(-1, 1)).reshape(frame_shape)
    else:
        frame = frame * mask
    return frame

def smooth_scale(frame, sigma=2, scale=3, multichannel=True):
    """
    apply a gaussian smoothing filter and then scale by given factor
    """
    frame = filters.gaussian(frame, sigma=sigma, multichannel=multichannel)
    return (frame*scale)

def thresh_colors(frame, block_size=17):
    """
    generate adaptive thresholds and use them to mask a color image
    """

    # adaptive thresholding
    mask = threshold_mask(frame, block_size=block_size)

    # apply mask
    frame = color_mask(frame, mask)

    return frame

def roll_channels_2d(frame, factor=1):
    """
    roll each channel by *factor* relative to the channel before it.
    """
    for channel in range(frame.shape[-1]-1):
        this_channel = frame[:, :, channel+1]
        this_channel = np.roll(this_channel, channel*factor, axis=0)
        this_channel = np.roll(this_channel, channel*factor, axis=1)
        frame[:, :, channel+1] = this_channel
    return frame

def dark_region_mask(frame, proportion=0.5):
    """
    return a mask covering regions darker than proportion * mean grayscale value with False
    """

    if frame.ndim == 2:
        # we have a BW frame so no need to do grayscale
        use_frame = frame.view()
    else:
        # color frame
        use_frame = grayscale(frame)
    threshold = proportion * use_frame.mean()

    return use_frame > threshold

def dark_regions(frame, proportion=0.5):
    """
    black out regions darker than proportion*mean
    """
    mask = dark_region_mask(frame, proportion=proportion)
    return color_mask(frame, mask)

def mask_together(pos_frame, neg_frame, mask):
    """
    zero out the pixels in pos_frame where mask is negative. zero out the pixels in neg_frame where
    mask is positive. add neg_frame to pos_frame and return pos_frame.
    """
    pos_frame = color_mask(pos_frame, mask)
    neg_frame = color_mask(neg_frame, np.logical_not(mask))
    if neg_frame.ndim == 3:
        return pos_frame + neg_frame
    else:
        return (pos_frame.reshape(-1, 3) + neg_frame.reshape(-1, 1)).reshape(pos_frame.shape)
