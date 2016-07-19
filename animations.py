# pylint: disable=E0401, E1101

"""
generators that animate some kind of basic scene. one frame per __next__ call.
these are pretty much just discrete event simulators.
"""

import sys
from collections import deque
from functools import partial, reduce

import cv2
import numpy as np

DEFAULT_COLORS = {
    "red" : np.array([255, 0, 0], dtype=np.uint8),
    "green" : np.array([0, 255, 0], dtype=np.uint8),
    "blue" : np.array([0, 0, 255], dtype=np.uint8),
    "white" : np.array([255, 255, 255], dtype=np.uint8)
}

class Void(object):
    """
    draw (in RGB) a black field with stars fading in and out at random locations and intervals.

    constructor arg [stars] is a json thing:
    {
        "regular_star" : {
            "probability" : 0.01,       # probability that this star will be made per star-making
                                            # call
            "num_tries" : 3,            # number of stars which could be generated per star-making
                                            # call
            "new_star_interval" : 5,    # try to make a new batch of stars every 
                                            # [new_star_interval] frames
            "args" : {
                "duration" : (20, 200), # (min, max) number of frames the star will last
                ...                     # whatever other args go to this particular star generator
            }

        }
    }
    """

    def __init__(self, shape, stars, seed=10):
        """
        shape : animate an RGB image of this shape
        stars : see above
        seed : integer seed for random number generator
        """

        # last dimension will always be 3 (RGB)
        self.shape = np.asarray(shape[:2]).append(3)

        # stars will be cycled through this queue as they're animated
        self.star_deque = deque()
        # the state of this random number generator is independent of any other one in use during
        # execution. so a given seed will give you a consistent animation from this object
        # regardless of other random animations going on at the same time.
        self.random_generator = np.random.RandomState(seed=seed)

        # these are all the stars we can use:
        self.star_functions = {
            "regular_stars" : regular_stars,
            "crosses" : crosses
        }
        if not all([this_star in self.star_generators for this_star in stars.keys()]):
            raise NameError("couldn't recognize all of the requested stars")

        self.star_parameters = stars

        

    def _build_new_stars(self, this_frame):
        """
        go through the star_parameters dict and try to create new stars as necessary.
        this_frame : which frame are we rendering?
        """

        for star_name, star_properties in self.star_parameters.items():
            if this_frame % star_properties["new_star_interval"] == 0:
                # on this frame we should try to create new instances of this star
                for _ in range(star_properties["num_tries"]):
                    if self.random_generator.uniform() < star_properties["probability"]:
                        self.star_deque.push(
                            self.star_generator(
                                self.star_functions[star_name],
                                star_properties["args"]))


    def frame_generator(self):
        """
        return an iterator which draws the scene and creates new stars as necessary
        """

        # which frame are we currently rendering? this controls which stars have a chance to
        # be created on this frame. overflowing to 0 will reset the creation pattern but it should
        # hardly be perceptible if you don't have high-probability, low frequency stars in the mix.
        frame_number = np.uint64(0)

        while True:
            frame = np.zeros(self.shape, dtype=np.uint8)
            self._build_new_stars(frame_number)
            for _ in range(len(self.star_deque)):
                # pop off each star iterator in turn
                this_star_iterator = self.star_deque.pop()
                try:
                    # try to draw the star on the frame
                    # frame = draw_star(this_star_iterator.__next__(), frame)
                    frame += this_star_iterator.__next__()
                except StopIteration:
                    # if there's no more drawing to do, that's fine.
                    pass
                else:
                    # if there might be, put it back.
                    self.star_deque.appendleft(this_star_iterator)
            frame_number += 1
            # done with the frame
            yield frame


    def star_generator(self, star_function, star_function_args):
        """
        yields a 3d numpy array and its location in the frame, wrapped in a dict. the numpy array
        is provided by the star function.

        {
            "star" : 3d ndarray,
            "location" : ndarray star minimum corner location as [row, col] pixel coordinates
        }

        star_function_args should contain "duration" and "size" at a minimum. they can be literal
        values or tuples of (min, max) specifying range for a random value.
        """

        pass

        #--------------------------------------
        
        # all of this part only has to be done once per star parameter specification. we can do it in another function called by the constructor method.
        # we can also use the same function (method for random generator) for size and duration.

    def _sanitize_args(self, arg_dict):
        """
        iterate over the arguments in a star arg dictionary and translate them into a useful
        standard format
        """
        # get a size first
        size = arg_dict["size"]
        if isinstance(size, tuple):
            star_function_args["size"] = self.random_generator.random_integers(
                low=size[0],
                high=size[1])

        # and get a duration
        duration = arg_dict["duration"]
        if isinstance(duration, tuple):
            duration = self.random_generator.random_integers(low=duration[0], high=duration[1])

        # now clean up the color specification
        arg_dict["color"] = color_lookup(arg_dict["color"])

    def _arg_from_maybe_range(self, maybe_range):
        """
        this is totally silly
        """
        if isinstance(maybe_range, tuple):
            return self.random_generator.random_integers(low=maybe_range[0], high=maybe_range[1])
        else:
            return maybe_range


        #--------------------------------------


        # get the basic star
        star = star_function(star_function_args)

        # now figure out its location (keeping in mind its shape and the shape of the frame)
        viable_area = self.shape - star.shape
        location = np.asarray([
            self.random_generator.random_integers(
                low=0,
                high=viable_area[0]),
            self.random_generator.random_integers(
                low=0,
                high=viable_area[1])])

        output = {"location": location}
        # not pretty

        for frame_number in range(duration):
            # we want the star to be at maximum value at duration/2 and minimum at 0 and duration.
            # this will scale linearly as a function of time.
            scale = (duration/2 - np.abs(frame_number - duration/2)) / (duration/2)
            output["star"] = (star * scale).astype(np.uint8)
            yield output



def draw_star_on_frame(star, frame, mode="replace"):
    """
    draw a star on the frame given, and return the frame.
    mode : {'add', 'replace', 'subtract'}
    """

    # what does the star generator output look like?

    return frame





def regular_stars(args):
    """
    draw a + at a random location. returns a dict:
    
    color can be specified as a string or 3 element ndarray (rgb, uint8)
    size and duration can be exact values (integer) or tuples of integers indicating a range for a
    uniform random number. size should be odd.
    """
    pass

    



def color_lookup(color):
    """
    given a string or ndarray representing a color, throw an error or return an appropriate ndarray.
    """
    if isinstance(color, str):
        return DEFAULT_COLORS[color]
    elif isinstance(color, np.ndarray):
        color = color.flatten()
        if color.shape == (3,):
            return color.astype(np.uint8)
        else:
            raise ValueError("color must be a 3-length ndarray")
    else:
        raise ValueError("color was not of a recognizable type")


def stick_render(stick_generator, frame_size=(480, 640)):
    """
    given a generator that yields an index array, fill in the indices provided by the generator in
    the frame with white and fill the rest black. then render the frame.
    """

    frame = np.zeros(frame_size, dtype=np.uint8)
    for indices in stick_generator(frame_size):
        # fill it with black
        frame.fill(0)
        np.put(frame, indices, 255)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def stick_frames(stick_generator, frame_size=(480, 640)):
    """
    stick_render but yields frames instead
    """
    frame = np.zeros(frame_size, dtype=np.uint8)
    for indices in stick_generator(frame_size):
        # fill it with black
        frame.fill(0)
        np.put(frame, indices, 255)
        yield frame


def marching_column(frame_size, offset=0):
    """
    basically the simplest animation. draw a column that moves across the frame.
    """

    rows, cols = frame_size
    # this is the index to the first col of each row in a frame this size
    indices = np.arange(rows)*cols
    # so, about the counter. we can cast it as an integer here but assignment will promote it to a
    # double. which means the animation might get jumpy if you let it run a long time. the other
    # expedient option would be to make an array with one uint in it and just let it overflow...
    # which means the animation will jump every time it overflows. w/e.
    frame_counter = offset
    while True:
        # don't forget to cast back to an integral type
        yield indices + int(frame_counter % cols)
        frame_counter += 1

def marching_row(frame_size, offset=0):
    """
    same as the marching col, but, well, if you need to read the docstring to understand this
    function i might not be able to help you
    """
    rows, cols = frame_size
    indices = np.arange(cols)
    frame_counter = offset
    while True:
        yield indices + int(frame_counter % rows)*cols
        frame_counter += 1

def stack_animations(frame_size, stick_generators):
    """
    given a list of animation generators, run them all at once. to use this, wrap it in a partial.
    """
    # oh, this is important. if you port this over to python2 then use functools.izip here or you're
    # in for a world of hurt.
    active_generators = [this_generator(frame_size) for this_generator in stick_generators]
    the_iterators = zip(*active_generators)
    for these_indices in the_iterators:
        all_index = reduce(np.union1d, these_indices)
        yield all_index

def lotta_rows_and_cols(num_rows, num_cols, frame_size=(480, 640)):
    """
    return a partial accepting one arg (frame size) combining a number of rows and columns at 
    ??? offsets.
    """
    # row_offsets = np.random.randint(low=0, high=frame_size[0], size=num_rows)
    # col_offsets = np.random.randint(low=0, high=frame_size[1], size=num_cols)
    row_offsets = np.linspace(0, frame_size[0], num_rows).astype(np.uint)
    col_offsets = np.linspace(0, frame_size[1], num_cols).astype(np.uint)
    row_generators = [partial(marching_row, offset=this_offset) for this_offset in row_offsets]
    col_generators = [partial(marching_column, offset=this_offset) for this_offset in col_offsets]
    all_generators = row_generators + col_generators
    return partial(stack_animations, stick_generators=all_generators)


# do a biased tree with arbitrary-sized bias matrix
# mutate the bias matrix over time
# allow bias matrix to change with depth

# webcam and animations as generators that just yield frames. then we can have our effect renderer
# take any kind of input source and rescale, lay on effects and framerate counter etc.

# use a pyramid decomposition of an input frame (like from a webcam) at some fixed depth as the
# bias matrix

def bias_tree_generator(frame, max_depth, bias, increment=1):
    """
    a generator to do the fun part of the bias tree animation. bring your own frame. the iterator
    will yield it back to you, but it will also modify the original memory in place. again,
    configure the bias matrix so that it visually corresponds to the pattern you want and we'll
    transpose it here. except i messed something up and we won't have to transpose..........
    the frame will be divided into a number of sections along each axis corresponding to the entries
    in the bias matrix.
    """

    # what shape is the bias matrix?
    bias_shape = bias.shape
    # flatten to facilitate indexing
    bias = np.asarray(bias).flatten()
    # normalize so that it sums to 1
    bias /= bias.sum()
    # now do a cumulative sum to facilitate indexing
    bias = np.cumsum(bias)

    def compute_sub_window(window):
        """
        given a window in the frame, pick a section at random and return its min and max indices.
        """
        # get a number from 0-1
        selection = np.random.uniform()
        # find where our choice slots in the bias matrix
        flat_index = np.searchsorted(bias, selection)
        # and turn that back into a multidimensional index
        n_index = np.unravel_index(flat_index, bias_shape)

        # now figure out where each sub-window starts and stops
        sub_window = []
        for dim in (0, 1):
            edges = np.linspace(
                window[dim, 0],
                window[dim, 1],
                num=bias_shape[dim]+1,
                endpoint=True,
                dtype=np.int)
            ends = edges[1:]
            starts = edges[:-1]
            all_sub_windows = list(zip(starts, ends))
            # now index into those potential sub-windows to get the one we selected at random
            sub_window.append(all_sub_windows[n_index[dim]])

        # now convert to an array
        sub_window = np.asarray(sub_window)
        return sub_window

    def descend_tree(window, my_depth):
        """
        at each node of the tree, pick a new child. increment values at all associated indices and
        recurse.
        """
        sub_window = compute_sub_window(window)
        frame[sub_window[0, 0]: sub_window[0, 1], sub_window[1, 0]: sub_window[1, 1]] += increment
        if my_depth < max_depth:
            descend_tree(sub_window, my_depth+1)

    # start from the entire frame-- [[minx, maxx], [miny, maxy]
    window = np.array([
        [0, frame.shape[0]],
        [0, frame.shape[1]]])

    # that's all it is
    while True:
        descend_tree(window, 0)
        yield frame

def n_bias_tree_sample(
        size=(600, 600),
        max_depth=6,
        bias=(
            (0.1, 0.7),
            (0.15, 0.05))):
    """
    recreate the original bias tree pattern with the new implementation
    """
    frame = np.zeros(size, dtype=np.uint8)
    # bias = np.asarray(bias)
    # bias = np.eye(3) + np.random.rand(3,3)
    # bias = np.eye(3)
    # bias = np.random.rand(3,4)
    print(bias)
    jenny = bias_tree_generator(frame, max_depth, bias)

    for _ in jenny:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def n_bias_tree_rgb(
        size=(600, 600),
        max_depth=6,
        increment=3):
    """
    same but colorized
    """
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    bias = [np.random.rand(3, 3) for _ in range(3)]
    for b in bias:
        print(b)
    channels = [bias_tree_generator(frame[:, :, d], max_depth, bias[d], increment=increment)\
        for d in range(3)]
    for _ in zip(*channels):
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def rgb_bias_tree_generator(
        size=(480, 640),
        max_depth=6,
        increment=3):
    """
    yield a bias tree frame in color
    """
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    bias = [np.random.rand(3, 3) for _ in range(3)]
    channels = [bias_tree_generator(frame[:, :, d], max_depth, bias[d], increment=increment)\
        for d in range(3)]
    for _ in zip(*channels):
        yield frame


def biased_tree(
        size=(400, 400),
        fill_branches=False,
        seed=10,
        max_depth=6,
        bias=(
            (0.1, 0.7),
            (0.15, 0.05))):
    """
    render a square frame. divide it into quarters recursively. at every tick, descend the tree at
    random, with each node being given a certain bias to be selected. if fill_branches, add 1 to all
    cells inside the selected quadrant before selecting a child to descend through. at the
    specified depth, add 1 to all cells in the last quadrant.
    bias should be entered as you want it visually-- due to the row/ col fuckery we will transpose
    it here.
    """

    frame = np.zeros(size, dtype=np.uint8)

    # seeding is optional
    if seed is not None:
        np.random.seed(seed)

    # transpose the bias matrix so that it visually corresponds to the row/ col standard
    # we will also flatten to facilitate indexing
    bias = np.asarray(bias).T.flatten()
    # normalize so that it sums to 1
    bias /= bias.sum()
    # now do a cumulative sum to facilitate indexing
    bias = np.cumsum(bias)

    def compute_sub_window(window):
        """
        given a window in the frame, pick a quadrant at random and return its min and max indices
        """
        # get a number from 0-1
        selection = np.random.uniform()
        # print(selection)
        # find where it slots in the bias matrix
        index = np.searchsorted(bias, selection)
        # print(index)
        # construct all possible sub-windows
        half_row = int(np.floor(window[0, 1] - window[0, 0]) * 0.5) + window[0, 0]
        half_col = int(np.floor(window[1, 1] - window[1, 0]) * 0.5) + window[1, 0]
        all_sub_windows = [
            [
                [window[0, 0], half_row],   # UL
                [window[1, 0], half_col]
            ],
            [
                [half_row, window[0, 1]],   # LL
                [window[1, 0], half_col]
            ],
            [
                [window[0, 0], half_row],   # UR
                [half_col, window[1, 1]]
            ],
            [
                [half_row, window[0, 1]],   # LR
                [half_col, window[1, 1]]

            ]]

        # now index into those sub-windows
        sub_window = np.asarray(all_sub_windows[index])
        # print("index: {}".format(index))
        # print("sub window: \n{}".format(sub_window))
        return sub_window

    def fill_tree(window, my_depth):
        """
        recurse down the tree and increment branches on the way
        """
        sub_window = compute_sub_window(window)
        frame[sub_window[0, 0]: sub_window[0, 1], sub_window[1, 0]: sub_window[1, 1]] += 1
        if my_depth < max_depth:
            fill_tree(sub_window, my_depth+1)

    def no_fill_tree(window, my_depth):
        """
        recurse down the tree without incrementing branches along the way
        """
        sub_window = compute_sub_window(window)
        if my_depth < max_depth:
            no_fill_tree(sub_window, my_depth+1)
        else:
            frame[sub_window[0, 0]: sub_window[0, 1], sub_window[1, 0]: sub_window[1, 1]] += 1


    # we only have to decide once if we want to fill branches or not
    if fill_branches:
        tree = fill_tree
    else:
        tree = no_fill_tree

    # start from the entire frame-- [[minx, maxx], [miny, maxy]
    window = np.array([
        [0, size[0]],
        [0, size[1]]])
    # print(window)

    while True:
        # descend the tree
        tree(window, 0)
        # print("ding")
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def triple_biased_tree(
        size=(400, 400),
        fill_branches=False,
        seed=10,
        max_depth=6,
        biases=(
            (
                (0.1, 0.7),
                (0.15, 0.05)
            ),
            (
                (0.4, 0.2),
                (0.1, 0.3)
            ),
            (
                (0.03, 0.2),
                (0.17, 0.6)
            ))):
    """
    render a square frame. divide it into quarters recursively. at every tick, descend the tree at
    random, with each node being given a certain bias to be selected. if fill_branches, add 1 to all
    cells inside the selected quadrant before selecting a child to descend through. at the
    specified depth, add 1 to all cells in the last quadrant.
    bias should be entered as you want it visually-- due to the row/ col fuckery we will transpose
    it here.
    """

    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    # seeding is optional
    if seed is not None:
        np.random.seed(seed)

    # transpose the bias matrix so that it visually corresponds to the row/ col standard
    # we will also flatten to facilitate indexing
    use_biases = []
    for bias in biases:
        bias = np.asarray(bias).T.flatten()
        # normalize so that it sums to 1
        bias /= bias.sum()
        # now do a cumulative sum to facilitate indexing
        bias = np.cumsum(bias)
        use_biases.append(bias)

    def compute_sub_window(window, my_bias):
        """
        given a window in the frame, pick a quadrant at random and return its min and max indices
        """
        # get a number from 0-1
        selection = np.random.uniform()
        # print(selection)
        # find where it slots in the bias matrix
        index = np.searchsorted(my_bias, selection)
        # print(index)
        # construct all possible sub-windows
        half_row = int(np.floor(window[0, 1] - window[0, 0]) * 0.5) + window[0, 0]
        half_col = int(np.floor(window[1, 1] - window[1, 0]) * 0.5) + window[1, 0]
        all_sub_windows = [
            [
                [window[0, 0], half_row],   # UL
                [window[1, 0], half_col]
            ],
            [
                [half_row, window[0, 1]],   # LL
                [window[1, 0], half_col]
            ],
            [
                [window[0, 0], half_row],   # UR
                [half_col, window[1, 1]]
            ],
            [
                [half_row, window[0, 1]],   # LR
                [half_col, window[1, 1]]

            ]]

        # now index into those sub-windows
        sub_window = np.asarray(all_sub_windows[index])
        # print("index: {}".format(index))
        # print("sub window: \n{}".format(sub_window))
        return sub_window

    def fill_tree(window, my_depth, my_bias, my_frame):
        """
        recurse down the tree and increment branches on the way
        """
        sub_window = compute_sub_window(window, my_bias)
        my_frame[sub_window[0, 0]: sub_window[0, 1], sub_window[1, 0]: sub_window[1, 1]] += 1
        if my_depth < max_depth:
            fill_tree(sub_window, my_depth+1, my_bias, my_frame)

    def no_fill_tree(window, my_depth, my_bias, my_frame):
        """
        recurse down the tree without incrementing branches along the way
        """
        sub_window = compute_sub_window(window, my_bias)
        if my_depth < max_depth:
            no_fill_tree(sub_window, my_depth+1, my_bias, my_frame)
        else:
            my_frame[sub_window[0, 0]: sub_window[0, 1], sub_window[1, 0]: sub_window[1, 1]] += 1


    # we only have to decide once if we want to fill branches or not
    if fill_branches:
        tree = fill_tree
    else:
        tree = no_fill_tree

    # start from the entire frame-- [[minx, maxx], [miny, maxy]
    window = np.array([
        [0, size[0]],
        [0, size[1]]])
    # print(window)

    while True:
        # descend the tree
        for stupid_index in range(3):
            tree(window, 0, use_biases[stupid_index], frame[:, :, stupid_index])
        # print("ding")
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # stacked_lines = partial(stack_animations, stick_generators=[marching_row, marching_column])
    # stick_render(marching_row)
    # stick_render(marching_column)
    # stick_render(stacked_lines)
    # LOTTA = lotta_rows_and_cols(10, 10)
    # stick_render(LOTTA)
    # biased_tree(fill_branches=True)
    # triple_biased_tree(fill_branches=True)
    # n_bias_tree_sample()
    n_bias_tree_rgb()
