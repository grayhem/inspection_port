# pylint: disable=E0401, E1101

"""
creating novel visuals from a webcam. specifically, this is the I/O part.
"""

import sys

from functools import partial

import cv2
import numpy as np


import effects
import primitives
import trees
import animations

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
    frame = np.atleast_3d(func(frame))
    last_time = cv2.getTickCount()

    # this is the way openCV suggests to do it. probably more accurate than time.time
    execution_time = (last_time - first_time) / cv2.getTickFrequency()

    # the whole idea here is kinda shonky. too lazy to figure out how to draw text on image.
    # bar_length = int(np.ceil(execution_time / time_per_pixel))
    # frame[-bar_height:, :bar_length, :] = 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(execution_time), (10, 50), font, 1, (255,255,255), 1, cv2.LINE_AA)

    # print(execution_time)
    # print(bar_length)
    return frame

def write_on_frame(frame, message):
    """
    write something on the frame. like, for instance, the time it took to render.
    """

    frame = np.atleast_3d(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(message), (10, 50), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

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
        cv2.imshow('frame', framerate_wrapper(func,frame))
        # cv2.imshow('frame', primitives.resize(framerate_wrapper(func, frame)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def camera_generator(device=0):
    """
    generator yielding frames from a webcam
    """

    cap = cv2.VideoCapture(device)
    while True:
        _, frame = cap.read()
        yield frame
    cap.release()
 
def render_with_animation(animation, mask_function, device=0):
    """
    use some masking function to substitute frames from an animation generator into a webcam feed
    """

    for frame, image in zip(camera_generator(device), animation()):
        mask = mask_function(frame)
        frame = primitives.mask_together(frame, image, mask)
        # cv2.imshow('frame', image)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def combine_gen_sources(source_a, source_b, mask):
    """
    given two source generators and a mask generator, combine the two sources using the mask
    """
    animation = zip(source_a(), source_b(), mask())

    first_time = cv2.getTickCount()
    for frame_a, frame_b, frame_mask in animation:
        frame = primitives.mask_together(frame_a, frame_b, frame_mask)
        last_time = cv2.getTickCount()
        execution_time = (last_time - first_time) / cv2.getTickFrequency()
        write_on_frame(frame, str(execution_time))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        first_time = cv2.getTickCount()

    cv2.destroyAllWindows()

#---------------------------------------------------------------------------------------------------

def render_generator(func, device=0):
    """
    render a cam feed with a generator yielding frames inside the while loop. just look at the code.
    """

    cap = cv2.VideoCapture(device)

    while True:
        _, frame = cap.read()
        jenny = func(frame)
        for this_frame in jenny:
            cv2.imshow('frame', this_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def render_rolling_corner(device=0):
    """
    move the upper left hand corner of the video input around the output. this is the prototype.
    """

    cap = cv2.VideoCapture(device)
    _, frame = cap.read()
    shape = np.asarray(frame.shape[:2]) / 2

    for position in rolling_position_generator(shape):
        _, frame = cap.read()
        # move the corner
        new_frame = frame.copy()
        piece = frame[:shape[0], :shape[1]]
        end = position + shape
        new_frame[position[0]: end[0], position[1]: end[1]] = piece
        # cv2.imshow('frame', primitives.sobel_triple(new_frame))
        cv2.imshow('frame', new_frame)
        # cv2.imshow('frame', framerate_wrapper(func, frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def rolling_position_generator(shape, start=(0, 0)):
    """
    yield a continuously rolling starting position for a sub-window.
    shape = half the size of the entire frame
    start = where the sub-window starts
    """
    position = np.array(start, dtype=np.int)

    while True:
        if position[1] < shape[1] and position[0] == 0:
            # move right until hits upper right corner
            position[1] += 1
        elif position[1] == shape[1] and position[0] < shape[0]:
            # move down until hits lower right corner
            position[0] += 1
        elif position[1] > 0 and position[0] == shape[0]:
            # move left until hits lower left corner
            position[1] -= 1
        elif position[1] == 0 and position[0] > 0:
            # move up until hits upper left corner
            position[0] -= 1

        yield position


def render_four_corners(device=0, func=None):
    """
    do 4 rolling corners. yes, they will overlap.
    """

    # get an example frame
    cap = cv2.VideoCapture(device)
    _, frame = cap.read()

    # dimensions of a sub-window
    shape = (np.asarray(frame.shape[:2])/2).astype(np.int)

    # upper left corner of each sub-window
    upper_lefts = np.array([
        [0, 0],
        [0, shape[1]],
        shape,
        [shape[0], 0]
    ], dtype=np.int)


    # lower right corner of each sub-window
    lower_rights = upper_lefts + shape

    # a position generator for each sub-window
    generators = [rolling_position_generator(shape, start=this_upper_left)\
        for this_upper_left in upper_lefts]

    for positions in zip(*generators):
        # get the original frame and the new frame
        _, frame = cap.read()
        new_frame = frame.copy()
        # drop each sub-window into its place in the frame. this is where the overlap happens
        for index, put_start in enumerate(positions):
            # this is where we put to
            put_end = put_start + shape

            # this is where we take from
            from_start = upper_lefts[index]
            from_end = lower_rights[index]

            # take
            sub_window = frame[from_start[0]: from_end[0], from_start[1]: from_end[1]]
            # put
            new_frame[put_start[0]: put_end[0], put_start[1]: put_end[1]] = sub_window

        if func is None:
            cv2.imshow('frame', new_frame)
        else:
            cv2.imshow('frame', framerate_wrapper(func, new_frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def render_four_backwards(device=0, func=None):
    """
    same as render_four_corners but instead of moving the corners of the original frame into new
    spots in the output frame, render moving sub-windows of the original frame into static corners
    of the output frame.
    """


    # get an example frame
    cap = cv2.VideoCapture(device)
    _, frame = cap.read()

    # dimensions of a sub-window
    shape = (np.asarray(frame.shape[:2])/2).astype(np.int)
    print(shape)

    # upper left corner of each sub-window
    upper_lefts = np.asarray([
        [0, 0],
        [0, shape[1]],
        shape,
        [shape[0], 0]
    ], dtype=np.int)

    # lower right corner of each sub-window
    lower_rights = upper_lefts + shape

    # a position generator for each sub-window
    generators = [rolling_position_generator(shape, start=this_upper_left)\
        for this_upper_left in upper_lefts]

    for positions in zip(*generators):
        # get the original frame and the new frame
        _, frame = cap.read()
        new_frame = frame.copy()
        # drop each sub-window into its place in the frame. this is where the overlap happens
        for index, from_start in enumerate(positions):
            # this is where we put to
            put_end = lower_rights[index]
            put_start = upper_lefts[index]

            # this is where we take from
            from_end = from_start + shape

            # take
            sub_window = frame[from_start[0]: from_end[0], from_start[1]: from_end[1]]

            # put
            new_frame[put_start[0]: put_end[0], put_start[1]: put_end[1]] = sub_window

        if func is None:
            cv2.imshow('frame', new_frame)
        else:
            cv2.imshow('frame', framerate_wrapper(func, new_frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# reflect and then rotate


#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # gray_window(int(sys.argv[1]))
        # render(primitives.draw_gray_tree, int(sys.argv[1]))
        # render(primitives.sobel_triple, int(sys.argv[1]))
        # render(primitives.dark_regions, int(sys.argv[1]))
        # render(effects.dark_sobel, int(sys.argv[1]))
        # render(effects.gray_snowflake, int(sys.argv[1]))
        # render(effects.special_snowflake, int(sys.argv[1]))
        # render(effects.sobel_glow, int(sys.argv[1]))

        # BIAS_TREE = partial(animations.n_bias_tree_rgb, size=(480, 640))
        # render (primitives.thresh_colors, int(sys.argv[1]))
        # render_with_animation(animations.rgb_bias_tree_generator, 
        # primitives.dark_region_mask, device=int(sys.argv[1]))
        STICKS = partial(animations.stick_frames, animations.lotta_rows_and_cols(10, 10))
        BIAS = partial(animations.rgb_bias_tree_generator, increment=10)
        # render_with_animation(STICKS, primitives.adaptive_threshold_mask, device=int(sys.argv[1]))
        CAMERA = partial(camera_generator, device=int(sys.argv[1]))
        COFFEE = partial(animations.coffee, dtype=np.bool, band_thickness=10)
        WALK = partial(animations.random_walk, dtype=np.bool)
        combine_gen_sources(BIAS, CAMERA, COFFEE)
        combine_gen_sources(BIAS, CAMERA, STICKS)
        combine_gen_sources(BIAS, CAMERA, WALK)

        # render_rolling_corner(int(sys.argv[1]))
        # render_four_corners(device=int(sys.argv[1]), func=primitives.resize)
        # render_four_backwards(device=int(sys.argv[1]), func=primitives.resize)
        # render(trees.color_tree, int(sys.argv[1]))
        # render(effects.sparkle_trees, int(sys.argv[1]))
        # render(effects.gray_skeleton, int(sys.argv[1]))
        # special_halo_render(int(sys.argv[1]))
        # special_halo_render_two(int(sys.argv[1]))
        # difference_render(smooth_scale, int(sys.argv[1]))
    else:
        render(tree_thresh)