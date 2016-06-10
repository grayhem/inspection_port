# pylint: disable=E0401, E1101

"""
capture and register imagery with a webcam. this is extremely crude and meant to be equally an 
artistic endeavor and learning experience.
"""

import sys

import cv2
import numpy as np

from skimage import feature
from skimage import morphology
from skimage import filters

import visual_fields

#---------------------------------------------------------------------------------------------------

def capture(num_frames, device=0):
    """
    capture (interactively) a number of frames and return them in a python list.
    press "c" to capture a frame or "q" to quit and return however many we already have.
    """
    camera = cv2.VideoCapture(device)

    frame_list = []
    while True:
        _, frame = camera.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            frame_list.append(frame)
            if len(frame_list) == num_frames:
                break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    return frame_list

#---------------------------------------------------------------------------------------------------

def register_translation(device=0):
    """
    capture 2 frames and use feature.register_translation to estimate a translation between them.

    TARGET moves to SOURCE.

    """

    # 2 framez
    frames = capture(2, device=device)

    # estimate parameters
    shifts, error, phasediff = feature.register_translation(
        visual_fields.grayscale(frames[0]),
        visual_fields.grayscale(frames[1]))

    # we'll just print them for now
    print("translation vector: {}".format(shifts))
    print("error: {}".format(error))
    print("phase difference (should be zero): {}".format(phasediff))

#---------------------------------------------------------------------------------------------------

class TranslationRegistrar(object):
    """
    initialize it with an image and register/ stitch other images to it. translation is the only
    supported transformation.
    """
    def __init__(self, source):
        """
        we'll register all subsequent images to the source
        """
        self.source = source
        self.image_dim = source.ndim
        # we'll use gray_source to register against
        if self.image_dim == 2:
            self.gray_source = self.source
        elif self.image_dim == 3:
            self.gray_source = visual_fields.grayscale(self.source)
        
        self.images = [self.source]
        self.translations = [np.zeros(2)]

    def add_target(self, *targets):
        """
        add another target (or more targets) to the list of targets
        """
        def check_shape(target):
            """
            check the shape of the array against our source
            """
            if not np.array_equal(self.source.shape, target.shape):
                raise ValueError("target shape doesn't match source")
        for this_target in targets:
            check_shape(this_target)
            # add our target
            self.images.append(this_target)
            # compute its offset to the source
            translation, _, _ = feature.register_translation(
                self.gray_source,
                visual_fields.grayscale(this_target))
            self.translations.append(translation)

    def stitch_images(self):
        """
        build a new image to contain the source and all targets.
        """

        # we'll set the lowest translations (which could be negative) to [0, 0] and move all others
        # relative to them
        translation_array = np.asarray(self.translations).astype(np.int)
        origin = translation_array.min(0)
        translation_array -= origin

        # compute the size of the output image
        image_size = self.source.shape[:2] + translation_array.max(0)
        if self.image_dim == 3:
            # i'm just gonna rush to the MVP here
            image_size = np.append(image_size, 3)
            print(image_size)
            blank_image = np.empty(image_size, dtype=np.float32)
            blank_image.fill(np.nan)
            output_images = []
            for this_image, this_translation in zip(self.images, translation_array):
                print(this_translation)
                new_image = blank_image.copy()
                stop = self.source.shape[:2] + this_translation
                new_image[this_translation[0] : stop[0], this_translation[1] : stop[1], :] = this_image
                output_images.append(new_image)
            output_stack = np.stack(output_images)
            print(output_stack.shape)
            mean_output = np.nanmean(output_stack, axis=0)
            std_output = np.nanstd(output_stack, axis=0)
            

            return kill_nans(mean_output), kill_nans(std_output)


def stitch(num_frames, device=0):
    """
    capture and stitch together a series of frames. translation only.
    """
    frames = capture(num_frames, device=device)

    registrar = TranslationRegistrar(frames[0])
    registrar.add_target(*frames[1:])

    mean, std = registrar.stitch_images()

    render_image(mean)
    render_image(std)

def kill_nans(frame):
    """
    replace nans with zeros and return a 0-255 image
    """

    output_nanmask = np.isnan(frame)
    np.putmask(frame, output_nanmask, 0)
    return frame.astype(np.uint8)



#---------------------------------------------------------------------------------------------------

def mean_stack(num_frames, device=0):
    """
    capture a number of frames, take their mean and render it.
    """

    frames = np.stack(capture(num_frames, device=device), axis=0)

    mean_frame = frames.mean(0).astype(np.uint8)
    render_image(mean_frame)

    std_frame = frames.std(0)
    render_image(std_frame)


def render_image(image):
    """
    render a single image
    """

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) == 2:
        DEVICE = int(sys.argv[1])
    else:
        DEVICE = 0
    stitch(6, device=DEVICE)
    # mean_stack(5, device=DEVICE)
    # register_translation(device=DEVICE)
    # FRAMES = capture(2, device=DEVICE)
    # print(len(FRAMES))
