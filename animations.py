# pylint: disable=E0401, E1101

"""
generators that animate some kind of basic scene. one frame per __next__ call.
these are pretty much just discrete event simulators.
"""

import sys
from collections import deque
from functools import partial

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



        --------------------------------------
        
        all of this part only has to be done once per star parameter specification. we can do it in another function called by the constructor method.
        we can also use the same function (method for random generator) for size and duration.

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


        --------------------------------------


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

