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
        self.star_generators = {
            "regular_stars" :  regular_stars,
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
                            self.star_generators[star_name](
                                self.random_generator,
                                **star_properties["args"]))


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
                    frame = draw_star(this_star_iterator.__next__(), frame)
                except StopIteration:
                    # if there's no more drawing to do, that's fine.
                    pass
                else:
                    # if there might be, put it back.
                    self.star_deque.appendleft(this_star_iterator)
            frame_number += 1
            # done with the frame
            yield frame



def draw_star(star, frame):
    """
    draw the output of a star generator on a frame
    """

    # what does the star generator output look like?

    return frame





def regular_stars(
        random_generator,
        color='red',
        size=5,
        duration=(30, 1000)):
    """
    using the supplied random number generator, draw a + at a random location. returns a dict:
    {
        "star" : 3d ndarray,
        "location" : ndarray star minimum corner location as proportion of frame edge. [0-1] float.
    }
    color can be specified as a string or 3 element ndarray (rgb, uint8)
    size and duration can be exact values (integer) or tuples of integers indicating a range for a
    uniform random number. size should be odd.
    """

    # if color was specified as a string, look up what it should be as an array
    if isinstance(color, str):
        color = DEFAULT_COLORS[color]

    # set a random location in the output
    output = {'location' : random_generator.uniform(size=2)}

    # and get a duration
    if isinstance(duration, tuple):
        duration = random_generator.random_integers(low=duration[0], high=duration[1])

    # and get a size

    # draw the original star
    original_star = make the star (using 1s and 0s only)
    we can make this function generic (higher-order) and pass it a star-generating function.
    or make it a method of the Void object so it will have access to the frame shape and random generator.
    
    for frame_number in range(duration):
        up until frame_number is half the duration, scale pixel color vector magnitude until it reaches max of color
        then after that decrease it until it hits 1 again.
        we can make the color scaling linear or exponential... etc

        output["star"] = original_star * color * scaling_factor

        yield output



