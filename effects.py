# pylint: disable=E0401, E1101

"""
more complicated effects and filters
"""

import numpy as np

from skimage import morphology

import primitives
import trees



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
    difference = primitives.color_mask(difference, mask)
    # return difference
    # and the original image
    frame = primitives.color_mask(frame, np.logical_not(mask))

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
    difference = primitives.color_mask(difference, mask)
    # return difference
    # and the original image
    frame = primitives.color_mask(frame, np.logical_not(mask))

    return frame + difference






def edge_tree(frame):
    """
    color tree with black edges generated using original (sharp) information
    """

    # get a mask for the edges
    edge_mask = primitives.not_edges(frame)

    # build a color tree
    frame = trees.color_tree(frame)

    # now apply the mask to the tree
    frame = primitives.color_mask(frame, edge_mask)
    return frame


def draw_channel_trees(frame):
    """
    use the individual channels (r,g,b for instance) of the image to draw separate trees on the
    frame. each tree will be drawn in the color of the channel it represents.
    """

    tree_function = trees.tree_edges
    tree_function = trees.quad_queue
    # pass each channel to the quadtree function (and dilate right away)
    # tree_collection = [morphology.binary_dilation(tree_function(frame[:, :, d]))\
        # for d in range(frame.shape[-1])]
    tree_collection = [tree_function(frame[:, :, d])\
        for d in range(frame.shape[-1])]

    # concat into a multichannel image. type is still bool.
    tree = np.stack(tree_collection, axis=-1)

    # make a mask of all the trees together.
    all_mask = np.stack(tree.any(axis=-1))

    # this is the use case for putmask. first we'll zero out all channels.
    # seems like we should be able to use broadcast_to here but there's some shape rules i don't
    # grasp. we can just use a for loop instead...
    for d in range(frame.shape[-1]):
        np.putmask(frame[:, :, d], all_mask, 0)
    # now draw the new tree edges in there
    np.putmask(frame, tree, 255)

    return frame

def thresh_tree_thresh(frame, block_size=7):
    """
    color tree masked with adaptive thresholds generated before and after the tree
    """

    # thresh the colors
    frame = primitives.thresh_colors(frame, block_size=block_size)

    # build a color tree
    frame = trees.color_tree(frame)

    # now thresh that tree
    return primitives.thresh_colors(frame, block_size=block_size)

def tree_thresh(frame, block_size=7):
    """
    generate a color tree and then mask it with adaptive thresholds
    """
    frame = trees.color_tree(frame)
    return primitives.thresh_colors(frame, block_size=block_size)

def not_cell_shading(frame):
    """
    this is definitely not cell shading.
    """

    # get a mask for the edges
    edge_mask = primitives.not_edges(frame)

    # build a color tree
    frame = trees.color_tree(frame)

    # smooth the color tree
    frame = filters.gaussian(frame, sigma=3, multichannel=True)

    # now apply the mask to the tree
    frame = primitives.color_mask(frame, edge_mask)
    return frame

