#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
import PIL

from absl import app, flags

# CUSTOM CODE
# just copy experiment id here, output will be saved in out/experiment_id/run/experiment_id.csv
# EXPERIMENT_ID = "unetpp_patches_test_run_2023-06-23_18-39-58"
#
# FLAGS = flags.FLAGS
#
# flags.DEFINE_string(
#     "submission_filename", f"../out/{EXPERIMENT_ID}/run/{EXPERIMENT_ID}.csv", "The output csv for the submission.")
# flags.DEFINE_string(
#     "base_dir", f"../out/{EXPERIMENT_ID}/run", "The directory with the predicted masks.")

DEFAULT_FOREGROUND_THRESHOLD = 0.25  # percentage of pixels of val 255 required to assign a foreground label to a patch


# assign a label to a patch
def patch_to_label(patch, foreground_threshold):
    patch = patch.astype(np.float64) / 255
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename, mask_dir, foreground_threshold):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", os.path.basename(image_filename)).group(0))
    im = PIL.Image.open(image_filename)
    im_arr = np.asarray(im)
    if len(im_arr.shape) > 2:
        # Convert to grayscale.
        im = im.convert("L")
        im_arr = np.asarray(im)

    patch_size = 16
    mask = np.zeros_like(im_arr)
    for j in range(0, im_arr.shape[1], patch_size):
        for i in range(0, im_arr.shape[0], patch_size):
            patch = im_arr[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, foreground_threshold)
            mask[i:i + patch_size, j:j + patch_size] = int(label * 255)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

    if mask_dir:
        save_mask_as_img(mask, os.path.join(mask_dir, "mask_" + os.path.basename(image_filename)))


def save_mask_as_img(img_arr, mask_filename):
    img = PIL.Image.fromarray(img_arr)
    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    img.save(mask_filename)


def masks_to_submission(submission_filename, mask_dir, foreground_threshold, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, mask_dir=mask_dir, foreground_threshold=foreground_threshold))


# Shouldn't be run as a script, but kept it for reference
# def main(_):
#     image_filenames = [os.path.join(FLAGS.base_dir, name) for name in os.listdir(FLAGS.base_dir)]
#     masks_to_submission(FLAGS.submission_filename, "", *image_filenames)
#
#
# if __name__ == '__main__':
#     app.run(main)
