# pylint: disable=E0401, E1101

"""
more complicated effects and filters
"""

import numpy as np

from skimage import morphology
from skimage import color
from skimage import filters

import primitives
import trees

from unseen import representations



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


def sobel_glow(frame, threshold=10):
    """
    use the 3-channel sobel filter (primitives.sobel_triple) to find edges. take the bright parts of
    that frame and replace pixels there in the original frame with them.
    """

    # get edges
    edges = primitives.sobel_triple(frame)

    # get a mask for the edges
    edge_mask = edges.max(2) < threshold

    # zero out the original frame where there are edges
    # frame = primitives.color_mask(frame, edge_mask)
    frame = primitives.mask_together(frame, edges, edge_mask)
    return frame
    # return frame + edges



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

    # pass each channel to the quadtree function (and dilate right away)
    tree_collection = [tree_function(frame[:, :, d])\
        for d in range(frame.shape[-1])]

    # concat into a multichannel image. type is still bool.
    tree = np.stack(tree_collection, axis=-1)

    # make a mask of all the trees together.
    all_mask = tree.any(axis=-1)

    # this is the use case for putmask. first we'll zero out all channels.
    # seems like we should be able to use broadcast_to here but there's some shape rules i don't
    # grasp. we can just use a for loop instead...
    for d in range(frame.shape[-1]):
        np.putmask(frame[:, :, d], all_mask, 0)
    # now draw the new tree edges in there
    np.putmask(frame, tree, 255)

    return frame


def sparkle_trees(frame):
    """
    use the individual channels (r,g,b for instance) of the image to draw separate trees on the
    frame. each tree will be drawn in the color of the channel it represents. then get rid of the
    white lines.
    """

    tree_function = trees.tree_edges

    # pass each channel to the quadtree function (and dilate right away)
    tree_collection = [tree_function(frame[:, :, d], max_tree_depth=6)\
        for d in range(frame.shape[-1])]

    # concat into a multichannel image. type is still bool.
    tree = np.stack(tree_collection, axis=-1)

    # make a mask of all the trees together.
    all_mask = tree.any(axis=-1)

    # and get rid of the whites
    no_whites_mask = np.logical_not(tree.all(axis=-1))
    all_mask = np.logical_and(all_mask, no_whites_mask)

    # this is the use case for putmask. first we'll zero out all channels.
    # seems like we should be able to use broadcast_to here but there's some shape rules i don't
    # grasp. we can just use a for loop instead...
    for d in range(frame.shape[-1]):
        np.putmask(frame[:, :, d], all_mask, 0)
        # now draw the new tree edges in there
        np.putmask(frame[:, :, d], np.logical_and(tree[:, :, d], no_whites_mask), 255)

    return frame

def skeleton_channels(frame):
    """
    convert image to another colorspace.
    split up the channels of the image and build a corner tree for each.
    skeletonize each channel.
    dilate each channel.
    roll each channel a little further in both directions than the one before.
    mask out the new image and draw the skeletons on it
    convert to rgb and return
    """

    tree_function = primitives.build_skeleton

    tree_collection = [tree_function(frame[:, :, d])\
        for d in range(frame.shape[-1])]

    # concat into a multichannel image. type is still bool.
    tree = np.stack(tree_collection, axis=-1)

    # roll the trees relative to one another
    primitives.roll_channels_2d(tree, factor=10)

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

def gray_skeleton(frame):
    """
    draw a skeletonized corner tree on the image using the grayscale
    """
    skeleton = primitives.build_skeleton(primitives.grayscale(frame))

    return primitives.color_mask(frame, np.logical_not(skeleton))

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

def dark_sobel(frame, proportion=0.5):
    """
    replace the dark regions of the image with triple sobel edges
    """
    sobel = primitives.sobel_triple(frame)
    mask = morphology.binary_dilation(primitives.dark_region_mask(frame, proportion=proportion))
    # keep the bright regions
    new_frame = primitives.color_mask(frame, mask)
    # replace dark with sobel edges
    # np.logical_not(mask, out=mask)
    # new_frame += primitives.color_mask(sobel*4, mask)
    new_frame += primitives.color_mask(sobel*4, np.logical_not(mask))

    return new_frame

# do a dot tree and compute radius of gyration. draw radius of gyration on the image w/o dot tree.

# grayscale an rgb frame and flatten it, then do symmetrized dots on the flattened signal and plot
# that with or without the original frame behind it
# do that for each channel independently, converting the boolean snowflake to a channel in the output
# image. use a not-binary dilation to smear the snowflakes out a little?
# vary parameters over time!

def gray_snowflake(frame, angle=60, top=250, lag=1, cutoff=2000):
    """
    grayscale the frame, flatten it, compute an SDP and return that.
    """
    frame = primitives.grayscale(frame)

    snowflake = representations.symmetrized_dots(
        np.sqrt(frame.flatten()[:cutoff]),
        angle=angle,
        top=top,
        lag=lag)

    # print(snowflake.shape)
    # snowflake = morphology.binary_dilation(snowflake)

    return (snowflake*255).astype(np.uint8)

def special_snowflake(frame, angle=60, top=240, lag=10, skip=20):
    """
    take each channel of the frame and do a snowflake on it
    """

    output = []
    for channel in range(frame.shape[-1]):

        snowflake = representations.symmetrized_dots(
            np.sqrt(frame[:, :, channel].flatten()[::skip]),
            angle=angle,
            top=top,
            lag=lag)
        # snowflake = morphology.binary_dilation(snowflake)
        snowflake = (snowflake*255).astype(np.uint8)
        output.append(snowflake)
    output = np.stack(output, axis=-1)
    # print(output.min())
    # print(output.max())
    output = primitives.smooth_scale(output, sigma=1, scale=1)
    output = primitives.normalize(output)
    # print(output.min())
    # print(output.max())
    # print(output.dtype)
    return output.astype(np.uint8)

