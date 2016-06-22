# pylint: disable=E0401, E1101

"""
creating novel visuals from a webcam. specifically, this is the I/O part.
"""

import sys

import cv2
import numpy as np


import effects
import primitives
import trees

# there's that dang ol void again
# make a filter that masks every pixel that is within a certain rgb distance from its last frame (block out everything not moving)
# make an object that has an iterator method. in that iterator it builds (at random intervals) new iterators that yield (for a certain 
# number of calls) a set of pixels drawing a shape like a star or upside down cross. these shapes grow brighter and then dim out.

# then in the render (which also compares with the last frame and blocks out unchanged areas) we call star_field.__next__() on the 
# while True loop. convolve some kind of filter with the star field and use the movement mask to combine it with the frame

# make the void fade in, in a region with smooth boundaries? somewhat difficult but will look cool


# pca projection: every 100 frames or something do a PCA on the frame and get the first principal component.
# project all the pixels in the subsequent frames along that component instead of using grayscale.




# monitor the framerate!
def framerate_wrapper(func, frame, time_per_pixel=0.0003, bar_height=30):
    """
    apply the given function to the given frame and record the execution time. 
    execution time is drawn as a white bar on the bottom of the frame, of length 
    execution_time / time_per_pixel
    """

    first_time = cv2.getTickCount()
    frame = func(frame)
    last_time = cv2.getTickCount()

    # this is the way openCV suggests to do it. probably more accurate than time.time
    execution_time = (last_time - first_time) / cv2.getTickFrequency()

    # the whole idea here is kinda shonky. too lazy to figure out how to draw text on image.
    bar_length = int(np.ceil(execution_time / time_per_pixel))
    frame[-bar_height:, :bar_length, :] = 255

    # print(execution_time)
    # print(bar_length)
    return frame

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
        _, frame = cap.read()
        roll = 150 * np.sin(frame_number * 0.01 * np.pi) # varies from 0-50 with 200 frame frequency
        cv2.imshow('frame', effects.halo(frame, roll=int(roll)))
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
            effects.halo_two_dee(
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
        _, frame = cap.read()
        # cv2.imshow('frame', func(frame))
        cv2.imshow('frame', framerate_wrapper(func, frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
 
#---------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # gray_window(int(sys.argv[1]))
        # render(primitives.draw_gray_tree, int(sys.argv[1]))
        render(trees.color_tree, int(sys.argv[1]))
        # special_halo_render(int(sys.argv[1]))
        # special_halo_render_two(int(sys.argv[1]))
        # difference_render(smooth_scale, int(sys.argv[1]))
    else:
        render(tree_thresh)

