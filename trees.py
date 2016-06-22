# pylint: disable=E0401, E1101, C0103

"""
recursive structures in the image domain
"""

import numpy as np

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

def quad_queue(frame, max_leaf_std=10, max_tree_depth=6):
    """
    non-recursive implementation of the quadtree.
    this version will just draw the tree on a new frame.
    next version, i'll abstract the logic to arbitrary operations and output.
        use functools.partial with some set of boolean predicate functions with kwargs...
    """

    frame = np.atleast_3d(frame)        # this is a view
    mask = np.zeros(frame.shape[:-1], dtype=np.bool)    
    dim = frame.shape[-1]
    
    def fork(node):
        """
        given a set of specs describing a node in the tree:
        node = {
            "depth": 1,
            "min": (0, 0),      # (row, col)
            "max": (10, 10)
        }
        decide whether that node is a branch or leaf. 
        if branch, draw on output, make a list of new nodes and return them.
        if leaf, do nothing and return [].
        """

        new_nodes = []

        if node["depth"] > max_tree_depth:
            return new_nodes

        # if we're still here, let's grab a view on the frame    
        min_row, min_col = node["min"]
        max_row, max_col = node["max"]
        # partition = frame[min_row: max_row, min_col: max_col, :].view()
        partition = frame[min_row: max_row, min_col: max_col, :]    # don't really get it yet

        # check out the std of each channel in this partition
        std = [partition[:, :, d].std() for d in range(dim)]
        if max(std) > max_leaf_std:
            # define new partitions
            half_row = int((max_row - min_row)/2)
            half_col = int((max_col - min_col)/2)

            # view the part of the mask described by the parent node
            # mask_partition = mask[min_row: max_row, min_col: max_col].view()
            mask_partition = mask[min_row: max_row, min_col: max_col]   # i think this might have to be a view for assignment to work like we want it to

            # draw on the mask (partition)
            mask_partition[half_row, :] = True
            mask_partition[:, half_col] = True

            # now build the new node params
            new_nodes.append({
                "depth": node["depth"]+1,
                "min": (min_row, min_row+half_row),
                "max": (min_col, min_col+half_col)
            })
            new_nodes.append({
                "depth": node["depth"]+1,
                "min": (min_row+half_row, max_row),
                "max": (min_col, min_col+half_col)
            })
            new_nodes.append({
                "depth": node["depth"]+1,
                "min": (min_row, min_row+half_row),
                "max": (min_col+half_col, max_col)
            })
            new_nodes.append({
                "depth": node["depth"]+1,
                "min": (min_row+half_row, max_row),
                "max": (min_col+half_col, max_col)
            })

        return new_nodes




def color_tree(frame, depth=0, max_leaf_std=10, max_tree_depth=6):
    """
    subdivide the image (with a quadtree) until the standard deviation of each color channel in each
    partition is below a given threshold. then fill that partition with the mean color.
    
    runs like doo-doo.

    """

    # check the standard deviation (and depth)
    std = frame.reshape(-1, 3).std(0)
    if std.max() <= max_leaf_std or depth == max_tree_depth:
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





def tree_edges(frame, max_tree_depth=6, max_leaf_std=10):
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
        if depth == max_tree_depth:
            # leaf
            pass
        elif max([frame_partition[:, :, d].std() for d in range(dim)]) >= max_leaf_std:
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






