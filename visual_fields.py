# pylint: disable=E0401, E1101

"""
creating novel visuals from a webcam
"""

import sys
from collections import deque
from functools import partial

import cv2
import numpy as np
from skimage import feature
from skimage import morphology
from skimage import filters



# there's that dang ol void again
# make a filter that masks every pixel that is within a certain rgb distance from its last frame (block out everything not moving)
# make an object that has an iterator method. in that iterator it builds (at random intervals) new iterators that yield (for a certain 
# number of calls) a set of pixels drawing a shape like a star or upside down cross. these shapes grow brighter and then dim out.

# then in the render (which also compares with the last frame and blocks out unchanged areas) we call star_field.__next__() on the 
# while True loop. convolve some kind of filter with the star field and use the movement mask to combine it with the frame

# make the void fade in, in a region with smooth boundaries? somewhat difficult but will look cool

# let's break this into a few modules.


# quadqueue: quadtree without recursion.
# make a closure that takes a set of bounds and depth and checks the std (or w/e) inside them using a view on the (nonlocal) frame array
# it then modifies the output array, using a view with same bounds. (if it's a line-drawing thing) 
# if the node condition is met, return the bounds on the child partitions, with depth += 1.
# if it's a color-averaging thing and leaf condition is met, average the colors in the output frame. or the input frame.

# make a queue and put a set of bounds describing the entire image in it. give it depth 0.
# while true:
#   try:
#       pop bounds from queue
#   except queue empty:
#       break
#   else:
#       call the closure with the bounds
#       push whatever bounds it gave back into the queue

#---------------------------------------------------------------------------------------------------

def difference_render(func, device=0, dist=10):
    """
    render the difference between the prior frame and the current frame and map some function
    """

    cap = cv2.VideoCapture(device)
    _, last_frame = cap.read()

    while True:
        _, this_frame = cap.read()
        cv2.imshow(
            'frame',
            func(
                np.abs(
                    this_frame.astype(np.int) - last_frame.astype(np.int)).astype(np.uint8)))
        last_frame = np.roll(this_frame, dist)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#---------------------------------------------------------------------------------------------------

def special_halo_render(device=0):
    """
    render a cam feed with halo, with continually varying parameters given to it
    """
    cap = cv2.VideoCapture(device)

    frame_number = 0
    while True:
        ret, frame = cap.read()
        roll = 150 * np.sin(frame_number * 0.01 * np.pi) # varies from 0-50 with 200 frame frequency
        cv2.imshow('frame', halo(frame, roll=int(roll)))
        frame_number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#---------------------------------------------------------------------------------------------------

def special_halo_render_two(device=0):
    """
    render a cam feed with halo_two_dee, with continually varying parameters given to it
    """
    cap = cv2.VideoCapture(device)

    frame_number = 0
    magnitude = 15
    while True:
        _, frame = cap.read()
        # varies from -magnitude to +magnitude with period 200 frames
        roll_rows = magnitude * np.sin(frame_number * 0.01 * np.pi)
        roll_cols = magnitude * np.cos(frame_number * 0.01 * np.pi)
        cv2.imshow(
            'frame',
            halo_two_dee(
                frame,
                roll_rows=int(roll_rows),
                roll_cols=int(roll_cols)))
        frame_number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#---------------------------------------------------------------------------------------------------

def render(func, device=0):
    """
    render a cam feed with some arbitrary function mapped onto it
    """
    cap = cv2.VideoCapture(device)

    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', func(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


    """
    every few seconds flatten out the spatial component of the frame
    leaving a (u * v) x 3 array. subsample as needed and do pca.
    keep the basis vectors and project the color values of each
    frame onto them.
    """
 
#---------------------------------------------------------------------------------------------------

MAX_TREE_DEPTH = 6
MAX_LEAF_STD = 10
def color_tree(frame, depth=0):
    """
    subdivide the image (with a quadtree) until the standard deviation of each color channel in each
    partition is below a given threshold. then fill that partition with the mean color.
    
    runs like doo-doo.

    """

    # check the standard deviation (and depth)
    std = frame.reshape(-1, 3).std(0)
    if std.max() <= MAX_LEAF_STD or depth == MAX_TREE_DEPTH:
        # we have a leaf!
        mean = frame.reshape(-1, 3).mean(0)
        # set each color channel to the mean
        for dim in range(3):
            frame[:, :, dim] = mean[dim]
    else:
        # we have a node (sigh...)
        y, x, _ = frame.shape
        y = int(y * 0.5)
        x = int(x * 0.5)
        # note the passing-by-reference fuckery
        color_tree(frame[:y, :x], depth=depth+1)
        color_tree(frame[:y, x:], depth=depth+1)
        color_tree(frame[y:, :x], depth=depth+1)
        color_tree(frame[y:, x:], depth=depth+1)

    return frame

def tree_edges(frame):
    """
    subdivide the image with a quadtree until std of each color channel in the partition is below
    given threshold. return a bool mask corresponding to the edges of every node and leaf.
    """

    frame = np.atleast_3d(frame)
    mask = np.zeros(frame.shape[:-1], dtype=np.bool)    
    dim = frame.shape[-1]

    def quadtree(frame_partition, mask_partition, depth=0):
        """
        partition and add splitting lines to the mask.
        """
        # std = frame_partition.reshape(-1, 3).std(0)
        if depth == MAX_TREE_DEPTH:
            # leaf
            pass
        elif max([frame_partition[:, :, d].std() for d in range(dim)]) >= MAX_LEAF_STD:
            # node
            y, x, _ = frame_partition.shape
            y = int(y / 2)
            x = int(x / 2)
            # draw the lines on the mask
            mask_partition[:, x] = True
            mask_partition[y, :] = True
            # now recurse
            quadtree(frame_partition[:y, :x], mask_partition[:y, :x], depth=depth+1)
            quadtree(frame_partition[:y, x:], mask_partition[:y, x:], depth=depth+1)
            quadtree(frame_partition[y:, :x], mask_partition[y:, :x], depth=depth+1)
            quadtree(frame_partition[y:, x:], mask_partition[y:, x:], depth=depth+1)


    quadtree(frame, mask)

    return mask

def draw_tree(frame):
    """
    draw the edges of a quadtree on the frame
    """
    tree = tree_edges(frame)
    tree = morphology.binary_dilation(tree)
    return color_mask(frame, np.logical_not(tree))

def draw_gray_tree(frame):
    """
    use a grayscale copy of the frame to draw a quadtree on the original frame
    """
    tree = tree_edges(grayscale(frame))
    tree = morphology.binary_dilation(tree)
    return color_mask(frame, np.logical_not(tree))

# pca projection: every 100 frames or something do a PCA on the frame and get the first principal component.
# project all the pixels in the subsequent frames along that component instead of using grayscale.

def neg_tree(frame):
    """
    draw a tree in negative
    """
    tree = tree_edges(frame)
    tree = morphology.binary_dilation(tree)
    return color_mask(frame, tree)

def line_tree_thresh(frame):
    """
    draw the edges of a quadtree and then run adaptive thresholding
    """
    frame = neg_tree(frame)
    return thresh_colors(frame)


def edge_tree(frame):
    """
    color tree with black edges generated using original (sharp) information
    """

    # get a mask for the edges
    edge_mask = not_edges(frame)

    # build a color tree
    frame = color_tree(frame)

    # now apply the mask to the tree
    frame = color_mask(frame, edge_mask)
    return frame


def thresh_colors(frame, block_size=7):
    """
    generate adaptive thresholds and use them to mask a color image
    """

    # adaptive thresholding
    mask = threshold_mask(frame, block_size=block_size)

    # apply mask
    frame = color_mask(frame, mask)

    return frame

def threshold_mask(frame, block_size=7):
    """
    use the adaptive threshold to generate a mask for an image.
    """
    return filters.threshold_adaptive(grayscale(frame), block_size=block_size)

def thresh_tree_thresh(frame, block_size=7):
    """
    color tree masked with adaptive thresholds generated before and after the tree
    """

    # thresh the colors
    frame = thresh_colors(frame, block_size=block_size)

    # build a color tree
    frame = color_tree(frame)

    # now thresh that tree
    return thresh_colors(frame, block_size=block_size)

def tree_thresh(frame, block_size=7):
    """
    generate a color tree and then mask it with adaptive thresholds
    """
    frame = color_tree(frame)
    return thresh_colors(frame, block_size=block_size)

def not_cell_shading(frame):
    """
    this is definitely not cell shading.
    """

    # get a mask for the edges
    edge_mask = not_edges(frame)

    # build a color tree
    frame = color_tree(frame)

    # smooth the color tree
    frame = filters.gaussian(frame, sigma=3, multichannel=True)

    # now apply the mask to the tree
    frame = color_mask(frame, edge_mask)
    return frame


#---------------------------------------------------------------------------------------------------

# input to all of these is expected to be an rgb frame

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

def halo(frame, roll=30, thresh_param=50):
    """
    let's try and put a crazy-ass halo around things.
    roll=30 and thresh_param=50 with the straight hard threshold looks good.
    """

    # roll the frame a little
    roll_frame = np.roll(frame, roll).astype(np.int32)

    # get the difference (we'll turn this into the halo)
    difference = np.abs(frame.astype(np.int32) - roll_frame).astype(np.uint8)

    # print(difference.min())
    # print(difference.max())
    # return difference
    # smooth and enhance the difference
    # difference = smooth_scale(difference, sigma=2, scale=150)

    # use adaptive thresholding to find a mask describing the interesting regions in the difference
    # mask = threshold_mask(difference, block_size=thresh_param)
    mask = np.linalg.norm(difference, axis=2) > thresh_param

    # get the differences
    difference = color_mask(difference, mask)
    # return difference
    # and the original image
    frame = color_mask(frame, np.logical_not(mask))

    return frame + difference

def halo_two_dee(frame, roll_rows=30, roll_cols=30, thresh_param=50):
    """
    do the halo thing with a 2D roll.
    deceptively, this is doing something rather different than the regular halo, which operates on
    the flattened array.
    """
    # roll the frame a little (rows)
    roll_frame = np.roll(frame, roll_rows, axis=0).astype(np.int32)
    # cols
    roll_frame = np.roll(roll_frame, roll_cols, axis=1).astype(np.int32)

    # get the difference (we'll turn this into the halo)
    difference = np.abs(frame.astype(np.int32) - roll_frame).astype(np.uint8)

    # use adaptive thresholding to find a mask describing the interesting regions in the difference
    # mask = threshold_mask(difference, block_size=thresh_param)
    mask = np.linalg.norm(difference, axis=2) > thresh_param

    # get the differences
    difference = color_mask(difference, mask)
    # return difference
    # and the original image
    frame = color_mask(frame, np.logical_not(mask))

    return frame + difference


#---------------------------------------------------------------------------------------------------

def compose_one_arg(arg, *functions):
    """
    compose all the single argument functions passed in and give them the arg.
    """
    if len(functions) > 1:
        return functions[0](compose_one_arg(arg, *functions))
    else:
        return functions[0](arg)

#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # gray_window(int(sys.argv[1]))
        # render(draw_gray_tree, int(sys.argv[1]))
        # special_halo_render(int(sys.argv[1]))
        special_halo_render_two(int(sys.argv[1]))
        # difference_render(smooth_scale, int(sys.argv[1]))
    else:
        render(tree_thresh)

