# https://github.com/hahnec/color-matcher
# https://github.com/victorca25/traiNNer (color_transfer.py)

import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
from color_matcher import ColorMatcher, __version__
from color_matcher.io_handler import load_img_file
from color_matcher.normalizer import Normalizer

cm = ColorMatcher()

"""
Script to apply color transfer from a source reference image to a target input image
    
Theory:
    https://www.scss.tcd.ie/Rozenn.Dahyot/pdf/pitie08bookchapter.pdf
    https://www.cse.cuhk.edu.hk/leojia/all_final_papers/color_cvpr05.PDF
    http://www.inf.ed.ac.uk/teaching/courses/vis/lecture_notes/lecture6.pdf

"""


def read_image(image):
    if isinstance(image, str):
        # read images as BGR
        return cv2.imread(image, cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        # use np image
        return image
    # elif pil .Image...:
    else:
        raise ValueError(
            "Unexpected image type. Either a path or a np.ndarray are supported"
        )


def scale_img(source=None, target=None):
    """
    Scale a source image to the same size as a target image
    """
    # raise ValueError("source and target shapes must be equal")
    # expand source to target size
    width = int(target.shape[1])
    height = int(target.shape[0])
    dim = (width, height)
    return cv2.resize(source, dim, interpolation=cv2.INTER_AREA)


def expand_img(image=None):
    # expand dimensions if grayscale
    if len(image.shape) < 3:
        return image[:, :, np.newaxis]
    else:
        return image


def _imstats(image, calc="direct"):
    """
    Calculate mean and standard deviation of an image along each channel.
    Using individual channels there's a very small difference with array forms,
    doesn't change the results

        Parameters:
        -------
            image: NumPy array OpenCV image
        calc: how to perform the canculation (differences are minimal,
            only included for completion)

    Returns:
        -------
        Mean (mu) and standard deviations (sigma)
    """
    if calc == "reshape":
        # reshape image from (H x W x 3) to (3 x HW) for vectorized operations
        image = image.astype("float32").reshape(-1, 3).T
        # calculate mean
        mu = np.mean(image, axis=1, keepdims=False)
        # calculate standard deviation
        sigma = np.std(image, axis=1, keepdims=False)
    elif calc == "direct":
        # calculate mean
        mu = np.mean(image, axis=(0, 1), keepdims=True)
        # calculate standard deviation
        sigma = np.std(image, axis=(0, 1), keepdims=True)
    elif calc == "split":
        # compute the mean and standard deviation of each channel independently
        (l, a, b) = cv2.split(image)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())
        mu = [lMean, aMean, bMean]
        sigma = [lStd, aStd, bStd]

    # return the color statistics
    return (mu, sigma)


def _scale_array(arr, clip=True, new_range=(0, 255)):
    """
    Trim NumPy array values to be in [0, 255] range with option of
    clipping or scaling.
    Parameters:
    -------
    arr: array to be trimmed to new_range (default: [0, 255] range)
    clip: if True, array will be limited with np.clip.
        if False then input array will be min-max scaled to
        range [max([arr.min(), 0]), min([arr.max(), 255])]
        by default
    new_range: range to be used for scaling
    Returns:
    -------
    NumPy array that has been scaled to be in [0, 255] range
    """

    if clip:
        # scaled = arr.copy()
        # scaled[scaled < 0] = 0
        # scaled[scaled > 255] = 255
        scaled = np.clip(arr, new_range[0], new_range[1])
        # scaled = np.clip(arr, 0, 255)
    else:
        # scaled_range = (max([arr.min(), new_range[0]]), min([arr.max(), new_range[1]]))
        scaled = _min_max_scale(arr, new_range=new_range)

    return scaled


def _min_max_scale(arr, new_range=(0, 255)):
    """
    Perform min-max scaling to a NumPy array
    Parameters:
    -------
    arr: NumPy array to be scaled to [new_min, new_max] range
    new_range: tuple of form (min, max) specifying range of
        transformed array
    Returns:
    -------
    NumPy array that has been scaled to be in
    [new_range[0], new_range[1]] range
    """
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()

    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        scaled = arr

    return scaled


def im2double(im):
    if im.dtype == "uint8":
        out = im.astype("float") / 255
    elif im.dtype == "uint16":
        out = im.astype("float") / 65535
    elif im.dtype == "float":
        out = im
    else:
        assert False
    out = np.clip(out, 0, 1)
    return out


def bgr2ycbcr(img, only_y=True):
    """bgr version of matlab rgb2ycbcr
    Python opencv library (cv2) cv2.COLOR_BGR2YCrCb has
    different parameters with MATLAB color convertion.
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img_ = img.astype(np.float32)
    if in_img_type != np.uint8:
        img_ *= 255.0
    # convert
    if only_y:
        # mat = [24.966, 128.553, 65.481])
        # rlt = np.dot(img_ , mat)/ 255.0 + 16.0
        rlt = np.dot(img_, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        # mat = np.array([[24.966, 128.553, 65.481],[112, -74.203, -37.797], [-18.214, -93.786, 112.0]])
        # mat = mat.T/255.0
        # offset = np.array([[[16, 128, 128]]])
        # rlt = np.dot(img_, mat) + offset
        # rlt = np.clip(rlt, 0, 255)
        ## rlt = np.rint(rlt).astype('uint8')
        rlt = np.matmul(
            img_,
            [
                [24.966, 112.0, -18.214],
                [128.553, -74.203, -93.786],
                [65.481, -37.797, 112.0],
            ],
        ) / 255.0 + [16, 128, 128]

        # to make ycrcb like cv2
        # rlt = rlt[:, :, (0, 2, 1)]

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def ycbcr2rgb_(img):
    """same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img_ = img.astype(np.float32)
    if in_img_type != np.uint8:
        img_ *= 255.0
    # convert
    rlt = np.matmul(
        img_,
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0, -0.00153632, 0.00791071],
            [0.00625893, -0.00318811, 0],
        ],
    ) * 255.0 + [-222.921, 135.576, -276.836]

    # xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    # img_[:, :, [1, 2]] -= 128
    # rlt = img_.dot(xform.T)
    np.putmask(rlt, rlt > 255, 255)
    np.putmask(rlt, rlt < 0, 0)

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def ycbcr2rgb(img, only_y=True):
    """
    bgr version of matlab ycbcr2rgb
    Python opencv library (cv2) cv2.COLOR_YCrCb2BGR has
    different parameters with MATLAB color convertion.

    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img_ = img.astype(np.float32)
    if in_img_type != np.uint8:
        img_ *= 255.0

    # to make ycrcb like cv2
    # rlt = rlt[:, :, (0, 2, 1)]

    # convert
    mat = np.array(
        [[24.966, 128.553, 65.481], [112, -74.203, -37.797], [-18.214, -93.786, 112.0]]
    )
    mat = np.linalg.inv(mat.T) * 255
    offset = np.array([[[16, 128, 128]]])

    rlt = np.dot((img_ - offset), mat)
    rlt = np.clip(rlt, 0, 255)
    ## rlt = np.rint(rlt).astype('uint8')

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def replace_channels(source=None, target=None, ycbcr=True, hsv=False, transfersv=False):
    """
    Extracts channels from source img and replaces the same channels
    from target, then returns the converted image.
    Args:
        target: bgr numpy array of input image.
        source: bgr numpy array of reference image.
        ycbcr: replace the color channels (Cb and Cr)
        hsv: replace the hue channel
        transfersv: if using hsv option, can also transfer the
            mean/std of the S and V channels
    Returns:
        target: transfered bgr numpy array of input image.
    """
    target = read_image(target)
    source = read_image(source)

    if source.shape != target.shape:
        source = scale_img(source, target)

    if ycbcr:
        # ycbcr_in = bgr2ycbcr(target, only_y=False)
        ycbcr_in = cv2.cvtColor(target, cv2.COLOR_BGR2YCR_CB)
        # if keep_y:
        y_in, _, _ = cv2.split(ycbcr_in)

        # ycbcr_ref = bgr2ycbcr(source, only_y=False)
        ycbcr_ref = cv2.cvtColor(source, cv2.COLOR_BGR2YCR_CB)

        # if histo_match:
        #     ycbcr_ref = histogram_matching(reference=ycbcr_ref, image=ycbcr_in)

        # ycbcr_out = stats_transfer(target=ycbcr_in, source=ycbcr_ref)

        # if keep_y:
        _, cb_out, cr_out = cv2.split(ycbcr_ref)
        ycbcr_out = cv2.merge([y_in, cb_out, cr_out])

        # target = ycbcr2rgb(ycbcr_out)
        target = cv2.cvtColor(ycbcr_out, cv2.COLOR_YCR_CB2BGR)

    if hsv:
        hsv_in = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        _, s_in, v_in = cv2.split(hsv_in)
        # h_in, s_in, v_in = cv2.split(hsv_in)

        hsv_ref = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
        h_out, _, _ = cv2.split(hsv_ref)

        if transfersv:
            hsv_out = stats_transfer(target=hsv_in, source=hsv_ref)
            _, s_out, v_out = cv2.split(hsv_out)
            hsv_out = cv2.merge([h_out, s_out, v_out])
        else:
            hsv_out = cv2.merge([h_out, s_in, v_in])

        target = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)

    return target.astype("uint8")


def hue_transfer(source=None, target=None):
    """Extracts hue from source img and applies mean and
    std transfer from target, then returns image with converted y.
    Args:
        target: bgr numpy array of input image.
        source: bgr numpy array of reference image.
    Returns:
        img_arr_out: transfered bgr numpy array of input image.
    """

    target = read_image(target)
    source = read_image(source)

    hsv_in = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    _, s_in, v_in = cv2.split(hsv_in)
    # h_in, s_in, v_in = cv2.split(hsv_in)

    hsv_ref = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)

    hsv_out = stats_transfer(target=hsv_in, source=hsv_ref)
    h_out, _, _ = cv2.split(hsv_out)
    # h_out, s_out, v_out = cv2.split(hsv_out)

    hsv_out = cv2.merge([h_out, s_in, v_in])
    # hsv_out = cv2.merge([h_in, s_out, v_out])

    img_arr_out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)

    return img_arr_out.astype("uint8")


def luminance_transfer(source=None, target=None):
    """Extracts luminance from source img and applies mean and
    std transfer from target, then returns image with converted y.
    Args:
        target: bgr numpy array of input image.
        source: bgr numpy array of reference image.
    Returns:
        img_arr_out: transfered bgr numpy array of input image.
    """

    target = read_image(target)
    source = read_image(source)

    # ycbcr_in = bgr2ycbcr(target, only_y=False)
    ycbcr_in = cv2.cvtColor(target, cv2.COLOR_BGR2YCR_CB)
    _, cb_in, cr_in = cv2.split(ycbcr_in)

    # ycbcr_ref = bgr2ycbcr(source, only_y=False)
    ycbcr_ref = cv2.cvtColor(source, cv2.COLOR_BGR2YCR_CB)

    ycbcr_out = stats_transfer(target=ycbcr_in, source=ycbcr_ref)
    y_out, _, _ = cv2.split(ycbcr_out)

    ycbcr_out = cv2.merge([y_out, cb_in, cr_in])

    # img_arr_out = ycbcr2rgb(ycbcr_out)
    img_arr_out = cv2.cvtColor(ycbcr_out, cv2.COLOR_YCR_CB2BGR)

    return img_arr_out.astype("uint8")


def ycbcr_transfer(source=None, target=None, keep_y=True, histo_match=False):
    """Convert img from rgb space to ycbcr space, apply mean and
    std transfer, then convert back.
    Args:
        target: bgr numpy array of input image.
        source: bgr numpy array of reference image.
        keep_y: option to keep the original target y channel unchanged.
        histo_match: option to do histogram matching before transfering the
            image statistics (if combined with keep_y, only color channels
            are modified).
    Returns:
        img_arr_out: transfered bgr numpy array of input image.
    """

    target = read_image(target)
    source = read_image(source)

    # ycbcr_in = bgr2ycbcr(target, only_y=False)
    ycbcr_in = cv2.cvtColor(target, cv2.COLOR_BGR2YCR_CB)
    if keep_y:
        y_in, _, _ = cv2.split(ycbcr_in)

    # ycbcr_ref = bgr2ycbcr(source, only_y=False)
    ycbcr_ref = cv2.cvtColor(source, cv2.COLOR_BGR2YCR_CB)

    if histo_match:
        ycbcr_ref = histogram_matching(reference=ycbcr_ref, image=ycbcr_in)

    ycbcr_out = stats_transfer(target=ycbcr_in, source=ycbcr_ref)

    if keep_y:
        _, cb_out, cr_out = cv2.split(ycbcr_out)
        ycbcr_out = cv2.merge([y_in, cb_out, cr_out])

    # img_arr_out = ycbcr2rgb(ycbcr_out)
    img_arr_out = cv2.cvtColor(ycbcr_out, cv2.COLOR_YCR_CB2BGR)

    return img_arr_out.astype("uint8")


def lab_transfer(source=None, target=None):
    """Convert img from rgb space to lab space, apply mean and
    std transfer, then convert back.
    Args:
        target: bgr numpy array of input image.
        source: bgr numpy array of reference image.
    Returns:
        img_arr_out: transfered bgr numpy array of input image.
    """

    target = read_image(target)
    source = read_image(source)

    lab_in = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    lab_ref = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)

    lab_out = stats_transfer(target=lab_in, source=lab_ref)
    img_arr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    return img_arr_out.astype("uint8")


def stats_transfer(source=None, target=None):
    """Adapt target's (mean, std) to source's (mean, std).
    img_o = (img_i - mean(img_i)) / std(img_i) * std(img_r) + mean(img_r).
    Args:
        target: bgr numpy array of input image.
        source: bgr numpy array of reference image.
    Returns:
        img_arr_out: transfered bgr numpy array of input image.
    """

    target = read_image(target)
    source = read_image(source)

    mean_in, std_in = _imstats(target)
    mean_ref, std_ref = _imstats(source)

    img_arr_out = (target - mean_in) / std_in * std_ref + mean_ref

    # clip
    img_arr_out = _scale_array(img_arr_out)
    return img_arr_out.astype("uint8")


def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(
        source.ravel(), return_inverse=True, return_counts=True
    )
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    # use linear interpolation of cdf to find new pixel values = interp(image, bins, cdf)
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

    # reshape to original image shape and return
    return interp_a_values[src_unique_indices].reshape(source.shape)


def histogram_matching(reference=None, image=None, clip=None):
    """
    Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.
    (https://en.wikipedia.org/wiki/Histogram_matching)

    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    Returns
    -------
    matched : ndarray
        Transformed input image.
    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.
    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    .. [2] https://github.com/scikit-image/scikit-image/blob/master/skimage/exposure/histogram_matching.py
    """

    image = read_image(image)  # target
    reference = read_image(reference)  # ref

    # expand dimensions if grayscale
    image = expand_img(image)
    reference = expand_img(reference)

    if image.ndim != reference.ndim:
        raise ValueError(
            "Image and reference must have the same number " "of channels."
        )

    if image.shape[-1] != reference.shape[-1]:
        raise ValueError(
            "Number of channels in the input image and " "reference image must match!"
        )

    matched = np.empty(image.shape, dtype=image.dtype)
    for channel in range(image.shape[-1]):
        matched_channel = _match_cumulative_cdf(
            image[..., channel], reference[..., channel]
        )
        matched[..., channel] = matched_channel

    if clip:
        matched = _scale_array(matched, clip=clip)

    return matched.astype("uint8")


def SOTransfer(
    source, target, steps=10, batch_size=5, reg_sigmaXY=16.0, reg_sigmaV=5.0, clip=False
):
    """
    Color Transform via Sliced Optimal Transfer, ported by @iperov
    https://dcoeurjo.github.io/OTColorTransfer

    source      - any float range any channel image
    target      - any float range any channel image, same shape as src
    steps       - number of solver steps
    batch_size  - solver batch size
    reg_sigmaXY - apply regularization and sigmaXY of filter, otherwise set to 0.0
    reg_sigmaV  - sigmaV of filter

    return value
    """

    source = read_image(source).astype("float32")
    target = read_image(target).astype("float32")

    if not np.issubdtype(source.dtype, np.floating):
        raise ValueError("source value must be float")
    if not np.issubdtype(target.dtype, np.floating):
        raise ValueError("target value must be float")

    # expand dimensions if grayscale
    target = expand_img(image=target)
    source = expand_img(image=source)

    # expand source to target size if smaller
    if source.shape != target.shape:
        source = scale_img(source, target)

    target_dtype = target.dtype
    h, w, c = target.shape
    new_target = target.copy()

    for step in range(steps):
        advect = np.zeros((h * w, c), dtype=target_dtype)
        for batch in range(batch_size):
            dir = np.random.normal(size=c).astype(target_dtype)
            dir /= np.linalg.norm(dir)

            projsource = np.sum(new_target * dir, axis=-1).reshape((h * w))
            projtarget = np.sum(source * dir, axis=-1).reshape((h * w))

            idSource = np.argsort(projsource)
            idTarget = np.argsort(projtarget)

            a = projtarget[idTarget] - projsource[idSource]
            for i_c in range(c):
                advect[idSource, i_c] += a * dir[i_c]
        new_target += advect.reshape((h, w, c)) / batch_size
        new_target = _scale_array(new_target, clip=clip)

    if reg_sigmaXY != 0.0:
        target_diff = new_target - target
        new_target = target + cv2.bilateralFilter(
            target_diff, 0, reg_sigmaV, reg_sigmaXY
        )

    # new_target = _scale_array(new_target, clip=clip)
    return new_target.astype("uint8")


class Regrain:
    def __init__(self, smoothness=1):
        """
        Regraining post-process to match color of resulting image and
        gradient of the source image.

        Automated colour grading using colour distribution transfer.
        F. Pitie , A. Kokaram and R. Dahyot (2007) Computer Vision and Image
        Understanding.

        https://github.com/frcs/colour-transfer/blob/master/regrain.m

        Parameters:
        smoothness (default=1, smoothness>=0): sets the fidelity of the
            original gradient field. e.g. smoothness = 0 implies resulting
            image = graded image.
        """
        self.nbits = [4, 16, 32, 64, 64, 64]
        self.smoothness = smoothness
        self.level = 0
        # self.eps = 2.2204e-16

    def regrain(self, source=None, target=None):
        """
        Keep gradient of target and color of source.
        https://github.com/frcs/colour-transfer/blob/master/regrain.m

        Resulting image = regrain(I_original, I_graded, [self.smoothness])
        """

        source = read_image(source)  # ref
        target = read_image(target)  # target

        # expand source to target size if smaller
        if source.shape != target.shape:
            source = scale_img(source, target)

        target = target / 255.0
        source = source / 255.0
        img_arr_out = np.copy(target)
        img_arr_out = self.regrain_rec(
            img_arr_out, target, source, self.nbits, self.level
        )

        # clip
        img_arr_out = _scale_array(img_arr_out, new_range=(0, 1))
        img_arr_out = (255.0 * img_arr_out).astype("uint8")
        return img_arr_out

    def regrain_rec(self, img_arr_out, target, source, nbits, level):
        """
        Direct translation of matlab code.
        https://github.com/frcs/colour-transfer/blob/master/regrain.m
        """

        [h, w, _] = target.shape
        h2 = (h + 1) // 2
        w2 = (w + 1) // 2
        if len(nbits) > 1 and h2 > 20 and w2 > 20:
            # Note: could use matlab-like bilinear imresize instead, cv2 has no antialias
            resize_arr_in = cv2.resize(target, (w2, h2), interpolation=cv2.INTER_LINEAR)
            resize_arr_col = cv2.resize(
                source, (w2, h2), interpolation=cv2.INTER_LINEAR
            )
            resize_arr_out = cv2.resize(
                img_arr_out, (w2, h2), interpolation=cv2.INTER_LINEAR
            )
            resize_arr_out = self.regrain_rec(
                resize_arr_out, resize_arr_in, resize_arr_col, nbits[1:], level + 1
            )
            img_arr_out = cv2.resize(
                resize_arr_out, (w, h), interpolation=cv2.INTER_LINEAR
            )
        img_arr_out = self.solve(img_arr_out, target, source, nbits[0], level)
        return img_arr_out

    def solve(self, img_arr_out, target, source, nbit, level, eps=1e-6):
        """
        Direct translation of matlab code.
        https://github.com/frcs/colour-transfer/blob/master/regrain.m
        """

        [width, height, c] = target.shape
        first_pad_0 = lambda arr: np.concatenate((arr[:1, :], arr[:-1, :]), axis=0)
        first_pad_1 = lambda arr: np.concatenate((arr[:, :1], arr[:, :-1]), axis=1)
        last_pad_0 = lambda arr: np.concatenate((arr[1:, :], arr[-1:, :]), axis=0)
        last_pad_1 = lambda arr: np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)

        delta_x = last_pad_1(target) - first_pad_1(target)
        delta_y = last_pad_0(target) - first_pad_0(target)
        delta = np.sqrt((delta_x**2 + delta_y**2).sum(axis=2, keepdims=True))

        psi = 256 * delta / 5
        psi[psi > 1] = 1
        phi = 30.0 * 2 ** (-level) / (1 + 10 * delta / self.smoothness)

        phi1 = (last_pad_1(phi) + phi) / 2
        phi2 = (last_pad_0(phi) + phi) / 2
        phi3 = (first_pad_1(phi) + phi) / 2
        phi4 = (first_pad_0(phi) + phi) / 2

        rho = 1 / 5.0
        for i in range(nbit):
            den = psi + phi1 + phi2 + phi3 + phi4
            num = (
                np.tile(psi, [1, 1, c]) * source
                + np.tile(phi1, [1, 1, c])
                * (last_pad_1(img_arr_out) - last_pad_1(target) + target)
                + np.tile(phi2, [1, 1, c])
                * (last_pad_0(img_arr_out) - last_pad_0(target) + target)
                + np.tile(phi3, [1, 1, c])
                * (first_pad_1(img_arr_out) - first_pad_1(target) + target)
                + np.tile(phi4, [1, 1, c])
                * (first_pad_0(img_arr_out) - first_pad_0(target) + target)
            )
            img_arr_out = (
                num / np.tile(den + eps, [1, 1, c]) * (1 - rho) + rho * img_arr_out
            )
        return img_arr_out


class PDFTransfer:
    def __init__(self, n=300, eps=1e-6, m=6, c=3):
        """Hyper parameters.

        Attributes:
            c: dim of rotation matrix, 3 for ordinary imgage.
            n: discretization num of distribution of image's pixels.
            m: num of random orthogonal rotation matrices.
            eps: epsilon prevents from dividing by zero.
        """
        self.n = n
        self.eps = eps
        if c == 3:
            self.rotation_matrices = Rotations.optimal_rotations()
        else:
            self.rotation_matrices = Rotations.random_rotations(m, c=c)

    def pdf_tranfer(self, source=None, target=None):
        """Apply probability density function transfer.
        img_o = t(img_i) so that f_{t(img_i)}(r, g, b) = f_{img_r}(r, g, b),
        where f_{img}(r, g, b) is the probability density function of img's rgb values.

        O = t(I), where t: R^3-> R^3 is a continous mapping so that
        f{t(I)}(r, g, b) = f{R}(r, g, b).

        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.7694
        https://github.com/pengbo-learn/python-color-transfer

        Args:
            target: bgr numpy array of input image.
            source: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """

        target = read_image(target)  # target
        source = read_image(source)  # ref

        # reshape (h, w, c) to (c, h*w)
        [h, w, c] = target.shape
        reshape_arr_in = target.reshape(-1, c).transpose() / 255.0
        reshape_arr_ref = source.reshape(-1, c).transpose() / 255.0
        # pdf transfer
        reshape_arr_out = self.pdf_transfer_nd(
            arr_in=reshape_arr_in, arr_ref=reshape_arr_ref
        )
        # reshape (c, h*w) to (h, w, c)
        reshape_arr_out = _scale_array(reshape_arr_out, new_range=(0, 1))
        reshape_arr_out = (255.0 * reshape_arr_out).astype("uint8")
        img_arr_out = reshape_arr_out.transpose().reshape(h, w, c)
        return img_arr_out

    def pdf_transfer_nd(self, arr_in=None, arr_ref=None, step_size=1):
        """Apply n-dim probability density function transfer.
        Args:
            arr_in: shape=(n, x).
            arr_ref: shape=(n, x).
            step_size: arr = arr + step_size * delta_arr.
        Returns:
            arr_out: shape=(n, x).
        """
        # n times of 1d-pdf-transfer
        arr_out = np.array(arr_in)
        for rotation_matrix in self.rotation_matrices:
            rot_arr_in = np.matmul(rotation_matrix, arr_out)
            rot_arr_ref = np.matmul(rotation_matrix, arr_ref)
            rot_arr_out = np.zeros(rot_arr_in.shape)
            for i in range(rot_arr_out.shape[0]):
                rot_arr_out[i] = self._pdf_transfer_1d(rot_arr_in[i], rot_arr_ref[i])
            # func = lambda x, n : self._pdf_transfer_1d(x[:n], x[n:])
            # rot_arr = np.concatenate((rot_arr_in, rot_arr_ref), axis=1)
            # rot_arr_out = np.apply_along_axis(func, 1, rot_arr, rot_arr_in.shape[1])
            rot_delta_arr = rot_arr_out - rot_arr_in
            delta_arr = np.matmul(
                rotation_matrix.transpose(), rot_delta_arr
            )  # np.linalg.solve(rotation_matrix, rot_delta_arr)
            arr_out = step_size * delta_arr + arr_out
        return arr_out

    def _pdf_transfer_1d(self, arr_in=None, arr_ref=None):
        """Apply 1-dim probability density function transfer.
        Args:
            arr_in: 1d numpy input array.
            arr_ref: 1d numpy reference array.
        Returns:
            arr_out: transfered input array.
        """

        arr = np.concatenate((arr_in, arr_ref))
        # discretization as histogram
        min_v = arr.min() - self.eps
        max_v = arr.max() + self.eps
        xs = np.array([min_v + (max_v - min_v) * i / self.n for i in range(self.n + 1)])
        hist_in, _ = np.histogram(arr_in, xs)
        hist_ref, _ = np.histogram(arr_ref, xs)
        xs = xs[:-1]
        # compute probability distribution
        cum_in = np.cumsum(hist_in)
        cum_ref = np.cumsum(hist_ref)
        d_in = cum_in / cum_in[-1]
        d_ref = cum_ref / cum_ref[-1]
        # transfer
        t_d_in = np.interp(d_in, d_ref, xs)
        t_d_in[d_in <= d_ref[0]] = min_v
        t_d_in[d_in >= d_ref[-1]] = max_v
        arr_out = np.interp(arr_in, xs, t_d_in)
        return arr_out


class Rotations:
    """generate orthogonal matrices for pdf transfer."""

    @classmethod
    def random_rotations(cls, m, c=3):
        """Random rotation."""

        assert m > 0
        rotation_matrices = [np.eye(c)]
        rotation_matrices.extend(
            [np.matmul(rotation_matrices[0], self.rvs(dim=c)) for _ in range(m - 1)]
        )
        return rotation_matrices

    @classmethod
    def optimal_rotations(cls):
        """Optimal rotation.

        Automated colour grading using colour distribution transfer.
        F. Pitié , A. Kokaram and R. Dahyot (2007) Journal of Computer Vision and Image Understanding.
        """

        rotation_matrices = [
            [
                [1.000000, 0.000000, 0.000000],
                [0.000000, 1.000000, 0.000000],
                [0.000000, 0.000000, 1.000000],
            ],
            [
                [0.333333, 0.666667, 0.666667],
                [0.666667, 0.333333, -0.666667],
                [-0.666667, 0.666667, -0.333333],
            ],
            [
                [0.577350, 0.211297, 0.788682],
                [-0.577350, 0.788668, 0.211352],
                [0.577350, 0.577370, -0.577330],
            ],
            [
                [0.577350, 0.408273, 0.707092],
                [-0.577350, -0.408224, 0.707121],
                [0.577350, -0.816497, 0.000029],
            ],
            [
                [0.332572, 0.910758, 0.244778],
                [-0.910887, 0.242977, 0.333536],
                [-0.244295, 0.333890, -0.910405],
            ],
            [
                [0.243799, 0.910726, 0.333376],
                [0.910699, -0.333174, 0.244177],
                [-0.333450, -0.244075, 0.910625],
            ],
            # [[-0.109199, 0.810241, 0.575834], [0.645399, 0.498377, -0.578862], [0.756000, -0.308432, 0.577351]],
            # [[0.759262, 0.649435, -0.041906], [0.143443, -0.104197, 0.984158], [0.634780, -0.753245, -0.172269]],
            # [[0.862298, 0.503331, -0.055679], [-0.490221, 0.802113, -0.341026], [-0.126988, 0.321361, 0.938404]],
            # [[0.982488, 0.149181, 0.111631], [0.186103, -0.756525, -0.626926], [-0.009074, 0.636722, -0.771040]],
            # [[0.687077, -0.577557, -0.440855], [0.592440, 0.796586, -0.120272], [-0.420643, 0.178544, -0.889484]],
            # [[0.463791, 0.822404, 0.329470], [0.030607, -0.386537, 0.921766], [-0.885416, 0.417422, 0.204444]],
        ]
        rotation_matrices = [np.array(x) for x in rotation_matrices]
        # for x in rotation_matrices:
        #    print(np.matmul(x.transpose(), x))
        #    import pdb
        #    pdb.set_trace()
        return rotation_matrices

    @classmethod
    def rvs(self, dim=3):
        """generate orthogonal matrices with dimension=dim.

        This is the rvs method pulled from the https://github.com/scipy/scipy/pull/5622/files,
        with minimal change - just enough to run as a stand alone numpy function.
        """
        random_state = np.random
        H = np.eye(dim)
        D = np.ones((dim,))
        for n in range(1, dim):
            x = random_state.normal(size=(dim - n + 1,))
            D[n - 1] = np.sign(x[0])
            x[0] -= D[n - 1] * np.sqrt((x * x).sum())
            # Householder transformation
            Hx = np.eye(dim - n + 1) - 2.0 * np.outer(x, x) / (x * x).sum()
            mat = np.eye(dim)
            mat[n - 1 :, n - 1 :] = Hx
            H = np.dot(H, mat)
            # Fix the last sign such that the determinant is 1
        D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
        # Equivalent to np.dot(np.diag(D), H) but faster, apparently
        H = (D * H.T).T
        return H


# Alternative CT calculation test to use with BlendingAlt. Still produces the lines in the images
def CT_alt(im=None, window_size=3):
    """
    Take a gray scale image and for each pixel around the center of the window generate a bit value of length
    window_size * 2 - 1. window_size of 3 produces bit length of 8, and 5 produces 24.

    The image gets border of zero padded pixels half the window size.

    Bits are set to one if pixel under consideration is greater than the center, otherwise zero.

    :param image: numpy.ndarray(shape=(MxN), dtype=numpy.uint8)
    :param window_size: int odd-valued
    :return: numpy.ndarray(shape=(MxN), , dtype=numpy.uint8)
    >>> image = np.array([ [50, 70, 80], [90, 100, 110], [60, 120, 150] ])
    >>> np.binary_repr(transform(image)[0, 0])
    '1011'
    >>> image = np.array([ [60, 75, 85], [115, 110, 105], [70, 130, 170] ])
    >>> np.binary_repr(transform(image)[0, 0])
    '10011'
    """

    half_window_size = window_size // 2
    image = cv2.copyMakeBorder(
        im,
        top=half_window_size,
        left=half_window_size,
        right=half_window_size,
        bottom=half_window_size,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    # Get the source image dims
    # w, h = im.size
    # h, w = im.shape
    rows, cols = image.shape

    # Initialize output array
    # Census = np.zeros((h-2, w-2), dtype='uint8')
    Census = np.zeros(
        (rows - half_window_size * 2, cols - half_window_size * 2), dtype=np.uint8
    )

    # centre pixels, which are offset by (1, 1)
    # cp = im[1:h-1, 1:w-1]
    center_pixels = image[
        half_window_size : rows - half_window_size,
        half_window_size : cols - half_window_size,
    ]

    # offsets of non-central pixels
    # offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]
    offsets = [
        (row, col)
        for row in range(half_window_size)
        for col in range(half_window_size)
        if not row == half_window_size + 1 == col
    ]

    # Do the pixel comparisons
    # for u, v in offsets:
    #     Census = (Census << 1) | (im[v:v+h-2, u:u+w-2] >= cp)
    for row, col in offsets:
        Census = (Census << 1) | (
            image[
                row : row + rows - half_window_size * 2,
                col : col + cols - half_window_size * 2,
            ]
            >= center_pixels
        )

    # print(Census.shape)
    return Census


def BlendingAlt(LR, HR):
    # TODO: Note: expects a single channel Y
    # H, W, _ = LR.shape
    H, W = LR.shape
    # H1, W1, _ = HR.shape
    H1, W1 = HR.shape
    assert H1 == H and W1 == W
    Census = CT_alt(LR)
    blending0 = Census * HR + (1 - Census) * LR
    return blending0


# Original CT calculation, to use with Blending1 and Blending2
def CT_descriptor(im):
    # TODO: Note: expects a single channel Y
    # H, W, _ = im.shape
    H, W = im.shape
    windowSize = 3
    Census = np.zeros((H, W))
    CT = np.zeros((H, W, windowSize, windowSize))
    C = np.int((windowSize - 1) / 2)
    for i in range(C, H - C):
        for j in range(C, W - C):
            cen = 0
            for a in range(-C, C + 1):
                for b in range(-C, C + 1):
                    if not (a == 0 and b == 0):
                        # TODO: Note: expects a single channel Y
                        if im[i + a, j + b] < im[i, j]:
                            cen += 1
                            CT[i, j, a + C, b + C] = 1
            Census[i, j] = cen
    Census = Census / 8
    # print(Census.shape, CT.shape)
    return Census, CT


def Blending1(LR, HR):
    # TODO: Note: expects a single channel Y
    # H, W, _ = LR.shape
    H, W = LR.shape
    # H1, W1, _ = HR.shape
    H1, W1 = HR.shape
    assert H1 == H and W1 == W
    Census, CT = CT_descriptor(LR)
    blending1 = Census * HR + (1 - Census) * LR
    # blending1 = cv2.addWeighted(HR, Census, LR, 1-Census, 0)
    return blending1


def Blending2(LR, HR):
    # TODO: Note: expects a single channel Y
    # H, W, _ = LR.shape
    H, W = LR.shape
    # H1, W1, _ = HR.shape
    H1, W1 = HR.shape
    assert H1 == H and W1 == W
    Census1, CT1 = CT_descriptor(LR)
    Census2, CT2 = CT_descriptor(HR)
    # print("1: ", Census1.min(), Census1.max(), CT1.min(), CT1.max())
    # print("2: ", Census2.min(), Census2.max(), CT2.min(), CT2.max())
    weight = np.zeros((H, W))
    x = np.zeros((3, 3))
    for i in range(H):
        for j in range(W):
            x = np.absolute(CT1[i, j] - CT2[i, j])
            weight[i, j] = x.sum()

    weight = weight / weight.max()
    blending2 = weight * LR + (1 - weight) * HR
    # blending2 = cv2.addWeighted(LR, weight, HR, 1-weight, 0)
    return blending2


def gaussian_blur(image=None, ksize=(3, 3), sigma=0.85):
    return cv2.GaussianBlur(image, ksize, sigma)


def _get_paths(path):
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            img_path = os.path.join(dirpath, fname)
            images.append(img_path)
    return images


def paired_walk_dir(
    s_path=None,
    t_path=None,
    o=None,
    algo=None,
    regrain=None,
    histo=None,
    blending=None,
    rep=None,
    threads=8,
):
    source_paths = _get_paths(s_path)
    target_paths = _get_paths(t_path)

    from multiprocessing import Pool
    from itertools import repeat

    with Pool(threads) as p:
        r = list(
            tqdm(
                p.starmap(
                    apply_transfer,
                    zip(
                        source_paths,
                        target_paths,
                        repeat(o),
                        repeat(algo),
                        repeat(regrain),
                        repeat(histo),
                        repeat(blending),
                        repeat(rep),
                    ),
                ),
                total=len(source_paths),
            )
        )


def walk_dir(
    s=None,
    t_path=None,
    o=None,
    algo=None,
    regrain=None,
    histo=None,
    blending=None,
    rep=None,
):
    for dpath, dnames, fnames in os.walk(t_path):
        for f in fnames:
            # os.chdir(t_path)
            full_path = os.path.join(t_path, f)
            # print(full_path)
            apply_transfer(
                s=s,
                t=full_path,
                o=o,
                algo=algo,
                regrain=regrain,
                histo=histo,
                blending=blending,
                rep=rep,
            )


def apply_transfer(
    s=None, t=None, o=None, algo=None, regrain=None, histo=None, blending=None, rep=None
):
    img = read_image(image=t)
    # img_name = t.split("/")[-1].split(".")[0]
    img_name = os.path.splitext(os.path.basename(t))[0]
    nam = ""

    # Pre-processing
    if rep:
        # replace image channels
        # Note: the target image with replaced channels could be used as the new target
        # or source for the other algorithms. If using as source, images have to be aligned
        replace = "source"  #'target'
        if replace == "target":
            img = replace_channels(
                source=s, target=img, ycbcr=True, hsv=False, transfersv=False
            )
        elif replace == "source":
            s = replace_channels(
                source=s, target=img, ycbcr=True, hsv=True, transfersv=True
            )
        nam = "{}_{}".format(nam, "rep")

    if algo:
        algos = algo.split(",")
        if isinstance(algos, str):
            algos = [algos]

        for alg in algos:
            if algo == "rgb" or algo == "bgr":
                # mean transfer
                img = stats_transfer(source=s, target=img)
            elif algo == "lab":
                # lab transfer
                img = lab_transfer(source=s, target=img)
            elif algo == "ycbcr":
                # ycbcr transfer
                img = ycbcr_transfer(source=s, target=img)
            elif algo == "lum":
                # luminance transfer
                img = luminance_transfer(source=s, target=img)
            elif algo == "hue":
                # hue transfer
                img = hue_transfer(source=s, target=img)
            elif algo == "pdf":
                # pdf transfer
                img = PDFTransfer(n=300).pdf_tranfer(source=s, target=img)
            elif algo == "sot":
                # sliced OT
                img = SOTransfer(source=s, target=img, steps=10, clip=False)
            elif algo == "histo":
                # histogram matching
                img = histogram_matching(reference=s, image=img)
            elif algo in ("hm", "reinhard", "mvgd", "mkl", "hm-mvgd-hm", "hm-mkl-hm"):
                s = read_image(image=s)
                # vic seems to apply changes the other way around, swapping
                source, target = img, s
                if algo == "hm":
                    img = cm.transfer(src=source, ref=target, method="hm")
                elif algo == "reinhard":
                    img = cm.transfer(src=source, ref=target, method="reinhard")
                elif algo == "mvgd":
                    img = cm.transfer(src=source, ref=target, method="mvgd")
                elif algo == "mkl":
                    # Monge-Kantorovich Linearization
                    img = cm.transfer(src=source, ref=target, method="mkl")
                elif algo == "hm-mvgd-hm":
                    img = cm.transfer(src=source, ref=target, method="hm-mvgd-hm")
                elif algo == "hm-mkl-hm":
                    img = cm.transfer(src=source, ref=target, method="hm-mkl-hm")

            # cv2.imwrite('{}/{}_{}.png'.format(o, img_name, algo),img)

            nam = "{}_{}".format(nam, alg)

    # Post-processing
    if histo:
        # histogram matching
        img = histogram_matching(reference=s, image=img)
        nam = "{}_{}".format(nam, "histo")

    if blending:
        blend_src = "target"  #'source'
        if blend_src == "target":
            orimg = read_image(image=t)
        if blend_src == "target_blur":
            orimg = read_image(image=t)
            orimg = gaussian_blur(image=orimg, ksize=(5, 5), sigma=0)  # ksize=(15,15)
        elif blend_src == "source":
            orimg = read_image(image=s)
            # expand source to target size if smaller
            if orimg.shape != img.shape:
                orimg = scale_img(orimg, img)
        elif blend_src == "regrain":
            orimg = Regrain().regrain(source=img, target=t)

        ycbcr_orimg = cv2.cvtColor(orimg, cv2.COLOR_BGR2YCR_CB)
        # ycbcr_orimg = im2double(ycbcr_orimg).astype('float64')
        y_orig, _, _ = cv2.split(ycbcr_orimg)

        ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        # ycbcr_img = im2double(ycbcr_img).astype('float64')
        y_img, cr_img, cb_img = cv2.split(ycbcr_img)
        CT_alt(y_img)

        # blend_y = Blending1(y_orig, y_img)
        blend_y = Blending2(y_orig, y_img)
        # blend_y = BlendingAlt(y_orig, y_img)

        blend_y = _scale_array(blend_y, clip=False).astype("uint8")
        # blend_y = _scale_array(255*blend_y, clip=False).astype("uint8")
        # cr_img = _scale_array(255*cr_img, clip=False).astype("uint8")
        # cb_img = _scale_array(255*cb_img, clip=False).astype("uint8")

        # print(blend_y.shape, blend_y.dtype)
        # print(cr_img.shape, cr_img.dtype)
        # print(cb_img.shape, cb_img.dtype)

        img = cv2.merge([blend_y, cr_img, cb_img])
        img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
        nam = "{}_{}".format(nam, "blending")

    if regrain:
        img = Regrain().regrain(source=img, target=t)
        nam = "{}_{}".format(nam, "regrain")

    # algo = algo.replace(',', '_')
    cv2.imwrite("{}/{}{}.png".format(o, img_name, nam), img)
    # print(img_name)
    # print('{}/{}{}.png'.format(o, img_name, nam))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str, help="You take colors from this image.")
    parser.add_argument("-t", type=str, help="You apply colors to this image.")
    parser.add_argument("-o", type=str, help="Output directory for resulting images.")
    parser.add_argument(
        "-algo",
        type=str,
        required=False,
        help="""Select which algorithm to use. Options: rgb, lab, ycbcr, lum, pdf, sot, hm, reinhard, mvgd, mkl, 
        hm-mvgd-hm, hm-mkl-hm.""",
    )
    parser.add_argument(
        "-regrain",
        dest="regrain",
        default=False,
        action="store_true",
        help="If added will use regrain for post-process.",
    )
    # parser.add_argument('-no-regrain', dest='regrain', action='store_false')
    parser.add_argument(
        "-histo",
        dest="histo",
        default=False,
        action="store_true",
        help="If added will use histogram matching after color transfer.",
    )
    parser.add_argument(
        "-blending",
        dest="blending",
        default=False,
        action="store_true",
        help="If added will blend original image and resulting color transfer image.",
    )
    parser.add_argument(
        "-rep",
        dest="rep",
        default=False,
        action="store_true",
        help="If added will replace channels of the target image from the source image.",
    )
    parser.add_argument(
        "-threads",
        type=int,
        default=8,
        required=False,
        help="Amount of threads for folder processing.",
    )

    args = parser.parse_args()
    s = args.s if args.s else "./source.png"
    t = args.t if args.t else "./target.png"
    o = args.o if args.o else "./"
    algo = args.algo
    regrain = args.regrain
    # regrain = False #True #
    histo = args.histo
    blending = args.blending
    rep = args.rep
    threads = args.threads

    # checks if both paths are a single file
    if os.path.isfile(s) and os.path.isfile(t):
        apply_transfer(
            s=s,
            t=t,
            o=o,
            algo=algo,
            regrain=regrain,
            histo=histo,
            blending=blending,
            rep=rep,
        )
    # checks if source is a single image and target path is a directory
    elif os.path.isdir(t) and os.path.isfile(s):
        walk_dir(
            s=s,
            t_path=t,
            o=o,
            algo=algo,
            regrain=regrain,
            histo=histo,
            blending=blending,
            rep=rep,
        )
    # check if both paths are a directory, must be paired images
    elif os.path.isdir(t) and os.path.isdir(s):
        paired_walk_dir(
            s_path=s,
            t_path=t,
            o=o,
            algo=algo,
            regrain=regrain,
            histo=histo,
            blending=blending,
            rep=rep,
            threads=threads,
        )
    else:
        raise ValueError(
            "Unsuported combination of source and target type: {} and {}."
        ).format(s, t)

    print("Done")
