# -*- coding: utf-8 -*-
from collections import Sequence

import numpy as np


class WarpTransform:
    """
    Interface class that represents a warp transformation as a combined rotation,
    scale, skew and translation 3 x 3 matrix. The transformation is called for each
    sample of a batch.
    """
    def __call__(self, index, in_shape, out_shape, rng):
        """
        __call__ needs to be implemented in subclasses

        Args:
            index (int): the index of the sample in the dataset
            in_shape (tuple of ints): the shape of the input sample
            out_shape (tuple of ints): the shape of the output sample
            rng (numpy.random.RandomState): a random number generator seeded by the dataloader

        Returns:
            out (tuple of numerics): a flatten, row-major 3 x 3 warp matrix returned in
                a tuple of numerics.
        """
        return NotImplementedError('__call__ needs to be implemented in subclasses')


class OOBTransform:
    """
    Interface class that represents an out of bounds transformation. The
    transformation is called for each sample of a batch.
    """
    def __call__(self, index, in_shape, out_shape, rng):
        """
        __call__ needs to be implemented in subclasses

        Args:
            index (int): the index of the sample in the dataset
            in_shape (tuple of ints): the shape of the input sample
            out_shape (tuple of ints): the shape of the output sample
            rng (numpy.random.RandomState): a random number generator seeded by the dataloader

        Returns:
            out (tuple of numerics): a tuple in RGB order containing the RGB color to
                use when no data is available. It should be in RGB order.
        """
        return NotImplementedError('__call__ needs to be implemented in subclasses')


class ColorTransform:
    """
    Interface class that represents a color transformation from YCbCr to RGB as
    defined in Benzina's kernel. The transformation is called for each sample of
    a batch.

    =====  =================================
    Index  Description
    =====  =================================
    0      ITU-R BT.601-6-625 recommentation

           * Kr = 0.299
           * Kg = 0.587
           * Kb = 0.114

           but full scale

           * Y,Cb,Cr in [0, 255]

    1      ITU-R BT.601-6-625 recommentation

           * Kr = 0.299
           * Kg = 0.587
           * Kb = 0.114

           with head/footroom

           * Y       in [16,235]
           * Cb,Cr   in [16,240]

    2      ITU-R BT.709 recommentation

           * Kr = 0.2126
           * Kg = 0.7152
           * Kb = 0.0722

           with head/footroom

           * Y       in [16,235]
           * Cb,Cr   in [16,240]

    3      ITU-R BT.2020 recommentation

           * Kr = 0.2627
           * Kg = 0.6780
           * Kb = 0.0593

           with head/footroom

           * Y       in [16,235]
           * Cb,Cr   in [16,240]
    =====  =================================
    """
    def __call__(self, index, in_shape, out_shape, rng):
        """
        __call__ needs to be implemented in subclasses

        Args:
            index (int): the index of the sample in the dataset
            in_shape (tuple of ints): the shape of the input sample
            out_shape (tuple of ints): the shape of the output sample
            rng (numpy.random.RandomState): a random number generator seeded by the dataloader

        Returns:
            out (tuple of numerics): a tuple containing a single int indicating which
                method to use when converting a sample's YCbCr value to RGB.
        """
        return NotImplementedError('__call__ needs to be implemented in subclasses')


class NormTransform:
    """
    Interface class that represents a normalization transformation. The transformation
    is called for each sample of a batch.
    """
    def __call__(self, index, in_shape, out_shape, rng):
        """
        __call__ needs to be implemented in subclasses

        Args:
            index (int): the index of the sample in the dataset
            in_shape (tuple of ints): the shape of the input sample
            out_shape (tuple of ints): the shape of the output sample
            rng (numpy.random.RandomState): a random number generator seeded by the dataloader

        Returns:
            out (tuple of numerics): a tuple in RGB order containing the normalization
                constant of a sample's RGB channels. Components will be multiplied to the
                respective channels of a sample.
        """
        return NotImplementedError('__call__ needs to be implemented in subclasses')


class BiasTransform:
    """
    Interface class that represents a bias transformation. The transformation
    is called for each sample of a batch.
    """
    def __call__(self, index, in_shape, out_shape, rng):
        """
        __call__ needs to be implemented in subclasses

        Args:
            index (int): the index of the sample in the dataset
            in_shape (tuple of ints): the shape of the input sample
            out_shape (tuple of ints): the shape of the output sample
            rng (numpy.random.RandomState): a random number generator seeded by the dataloader

        Returns:
            out (tuple of numerics): a tuple in RGB order containing the bias of a
                sample's RGB channels. Components will be substracted to the respective
                channels of a sample.
        """
        return NotImplementedError('__call__ needs to be implemented in subclasses')


class ConstantWarpTransform (WarpTransform):
    """
    Represents a constant warp transformation to be applied on each sample of a
    batch independently of its index.

    Args:
        warp (iterable of numerics, optional): a flatten, row-major 3 x 3 warp matrix
            (default: flatten identity matrix).
    """
    def __init__(self, warp=(1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0)):
        if warp is None:
            warp = (1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0)
        self.warp = tuple(warp)

    def __call__(self, index, in_shape, out_shape, rng):
        return self.warp


class ConstantOOBTransform  (OOBTransform):
    """
    Represents a constant out of bounds transformation to be applied on each
    sample of a batch independently of its index.

    Args:
        oob (numeric or iterable of numerics, optional): an iterable in RGB order
            containing the RGB color to use when no data is available (default:
            (0.0, 0.0, 0.0)).
    """
    def __init__(self, oob=(0.0, 0.0, 0.0)):
        if   oob is None:
            oob = (0.0, 0.0, 0.0)
        elif isinstance(oob, (int, float)):
            oob = (float(oob),)*3
        self.oob = tuple(oob)

    def __call__(self, index, in_shape, out_shape, rng):
        return self.oob


class ConstantColorTransform(ColorTransform):
    """
    Represents a constant color transformation to be applied on each sample of a
    batch independently of its index.

    Args:
        index (int, optional): the index of the method to use when converting
            a sample's YCbCr value to RGB (default: 0).
    """
    def __init__(self, index=0):
        if   index is None:
            index = (0,)
        elif isinstance(index, (int)):
            index = (int(index),)
        self.index = tuple(index)

    def __call__(self, index, in_shape, out_shape, rng):
        return self.index


class ConstantNormTransform (NormTransform):
    """
    Represents a constant norm transformation to be applied on each sample of a
    batch independently of its index.

    Args:
        norm (numeric or iterable of numerics, optional): an iterable in RGB order
            containing the normalization constant of a sample's RGB channels. Components
            will be multiplied to the respective channels of a sample
            (default: (1.0, 1.0, 1.0)).
    """
    def __init__(self, norm=(1.0, 1.0, 1.0)):
        if   norm is None:
            norm = (1.0, 1.0, 1.0)
        elif isinstance(norm, (int, float)):
            norm = (float(norm),)*3
        self.norm = tuple(norm)

    def __call__(self, index, in_shape, out_shape, rng):
        return self.norm


class ConstantBiasTransform (BiasTransform):
    """
    Represents a constant bias transformation to be applied on each sample of a
    batch independently of its index.

    Args:
        bias (numeric or iterable of numerics, optional): an iterable in RGB order
            containing the bias of a sample's RGB channels. Components will be
            substracted to the respective channels of a sample (default: (0.0, 0.0, 0.0)).
    """
    def __init__(self, bias=(0.0, 0.0, 0.0)):
        if   bias is None:
            bias = (0.0, 0.0, 0.0)
        elif isinstance(bias, (int, float)):
            bias = (float(bias),)*3
        self.bias = tuple(bias)

    def __call__(self, index, in_shape, out_shape, rng):
        return self.bias


class SimilarityTransform   (WarpTransform):
    """
    Similarity warp transformation of the image keeping center invariant.

    A crop of random size, aspect ratio and location is made. This crop can
    then be flipped and/or rotated to finally be resized to output size.

    Args:
        scale (Sequence or float or int, optional): crop area scaling factor
            interval, e.g (a, b), then scale is randomly sampled from the range
            a <= scale <= b. If scale is a number instead of sequence, the
            range of scale will be (scale^-1, scale).
            (default: ``(+1.0, +1.0)``)
        ratio (Sequence or float or int, optional): range of crop aspect ratio.
            If ratio is a number instead of sequence like (min, max), the range
            of aspect ratio will be (ratio^-1, ratio). Will keep original
            aspect ratio by default.
        degrees (Sequence or float or int, optional): range of degrees to
            select from. If degrees is a number instead of sequence like
            (min, max), the range of degrees will be (-degrees, +degrees).
            (default: ``(-0.0, +0.0)``)
        translate (Sequence or float or int, optional): sequence of maximum
            absolute fraction for horizontal and vertical translations. For
            example translate=(a, b), then horizontal shift is randomly sampled
            in the range -output_width * a < dx < output_width * a and vertical
            shift is randomly sampled in the range
            -output_height * b < dy < output_height * b. If translate is a
            number instead of sequence, translate will be
            (translate, translate). These translations are applied
            independently from :attr:`random_crop`. (default: ``(0.0, 0.0)``)
        flip_h (bool, optional): probability of the image being flipped
            horizontally. (default: ``+0.0``)
        flip_v (bool, optional): probability of the image being flipped
            vertically. (default: ``+0.0``)
        resize (bool, optional): resize the cropped image to fit the output
            size. It is forced to ``True`` if :attr:`scale` or :attr:`ratio`
            are specified. (default: ``False``)
        keep_ratio (bool, optional): match the smaller edge to the
            corresponding output edge size, keeping the aspect ratio after
            resize. Has no effect if :attr:`resize` is ``False``.
            (default: ``False``)
        random_crop (bool, optional): randomly crop the image instead of
            a center crop. (default: ``False``)
    """
    def __init__(self,
                 scale=(+1.0, +1.0),
                 ratio=None,
                 degrees=(-0.0, +0.0),
                 translate=(+0.0, +0.0),
                 flip_h=+0.0,
                 flip_v=+0.0,
                 resize=False,
                 keep_ratio=False,
                 random_crop=False):
        if not isinstance(scale, Sequence):
            scale = min(scale, 1/scale)
            scale = (scale, 1/scale)
        assert len(scale) == 2, \
            "scale should be a number or a sequence of length 2."
        for s in scale:
            if s <= 0:
                raise ValueError("scale values should be positive")

        if ratio is not None:
            if not isinstance(ratio, Sequence):
                ratio = (ratio, 1/ratio)
            assert len(ratio) == 2, \
                "ratio should be a number or a sequence of length 2."
            for ar in ratio:
                if ar <= 0:
                    raise ValueError("ratio values should be positive")
            ratio = (min(ratio), max(ratio))

        if not isinstance(degrees, Sequence):
            if degrees < 0:
                raise ValueError("If radians is a single number, it must be "
                                 "positive.")
            degrees = (-degrees, degrees)
        else:
            assert len(degrees) == 2, \
                "degrees should be a number or a sequence of length 2."
            degrees = degrees

        assert isinstance(translate, Sequence) and len(translate) == 2, \
            "translate should be a sequence and it must be of length 2."
        for t in translate:
            if not (0.0 <= t <= 1.0):
                raise ValueError("translation values should be between 0 and "
                                 "1")

        self.s = scale
        self.ar = ratio
        self.r = degrees
        self.t = translate
        self.fh = float(flip_h)
        self.fv = float(flip_v)
        self.resize = resize or scale != (1.0, 1.0) or ratio is not None
        self.keep_ratio = keep_ratio and self.resize
        self.random_crop = random_crop

    def __call__(self, index, in_shape, out_shape, rng):
        """Return a random similarity transformation."""
        s = np.exp(rng.uniform(low=np.log(self.s[0]), high=np.log(self.s[1])))
        if self.ar is not None:
            for _ in range(10):
                ar = np.exp(rng.uniform(low=np.log(self.ar[0]),
                                        high=np.log(self.ar[1])))
                crop_area = s * in_shape[0] * in_shape[1]
                crop_w = np.sqrt(crop_area * ar)
                crop_h = np.sqrt(crop_area / ar)

                if 0 < crop_w <= in_shape[0] and 0 < crop_h <= in_shape[1]:
                    break
            else:
                # Fallback to central crop
                in_ar = float(in_shape[0]) / float(in_shape[1])
                if in_ar < self.ar[0]:
                    crop_w = in_shape[0]
                    crop_h = int(round(crop_w / self.ar[0]))
                elif in_ar > self.ar[1]:
                    crop_h = in_shape[1]
                    crop_w = int(round(crop_h * self.ar[1]))
                else:  # whole image
                    crop_w, crop_h = in_shape
                self.random_crop = False
        elif self.resize:
            sqrt_s = np.sqrt(s)
            crop_w = in_shape[0] * sqrt_s
            crop_h = in_shape[1] * sqrt_s
        else:
            crop_w, crop_h = out_shape
        if self.random_crop:
            random_crop_tx = max(0, (in_shape[0] - crop_w) / 2)
            random_crop_ty = max(0, (in_shape[1] - crop_h) / 2)
            crop_x = rng.uniform(low=-random_crop_tx, high=random_crop_tx)
            crop_y = rng.uniform(low=-random_crop_ty, high=random_crop_ty)
        else:
            crop_x = 0
            crop_y = 0
        r = rng.uniform(low=self.r[0], high=self.r[1])
        max_tx = self.t[0] * out_shape[0]
        max_ty = self.t[1] * out_shape[1]
        tx = rng.uniform(low=-max_tx, high=max_tx)
        ty = rng.uniform(low=-max_ty, high=max_ty)
        fh = rng.uniform() < self.fh
        fv = rng.uniform() < self.fv

        H = compute_affine_matrix(in_shape, out_shape,
                                  (crop_x, crop_y, crop_w, crop_h),
                                  r, (tx, ty), fh, fv, self.resize,
                                  self.keep_ratio)

        return tuple(H.flatten().tolist())


class RandomResizedCrop     (SimilarityTransform):
    """
    Crop to random size, aspect ratio and location.

    A crop of random size, aspect ratio and location is made. This crop is
    finally resized to output size.

    This is popularly used to train the Inception networks.

    Args:
        scale (Sequence or float or int, optional): crop area scaling factor
            interval, e.g (a, b), then scale is randomly sampled from the range
            a <= scale <= b. If scale is a number instead of sequence, the
            range of scale will be (scale^-1, scale).
            (default: ``(+0.08, +1.0)``)
        ratio (Sequence or float or int, optional): range of crop aspect ratio.
            If ratio is a number instead of sequence like (min, max), the range
            of aspect ratio will be (ratio^-1, ratio). Will keep original
            aspect ratio by default. (default: ``(3./4., 4./3.)``)
    """
    def __init__(self,
                 scale=(+0.08, +1.0),
                 ratio=(3./4., 4./3.)):
        SimilarityTransform.__init__(self, scale=scale, ratio=ratio,
                                     resize=True, random_crop=True)


class CenterResizedCrop     (SimilarityTransform):
    """
    Crops at the center and resize.

    A crop at the center is made then resized to the output size.

    Args:
        scale (float or int, optional): edges scaling factor.
            (default: ``+1.0``)
        keep_ratio (bool, optional): match the smaller edge to the
            corresponding output edge size, keeping the aspect ratio after
            resize. (default: ``False``)
    """
    def __init__(self,
                 scale=+1.0,
                 keep_ratio=True):
        SimilarityTransform.__init__(self,
                                     scale=(pow(scale, 2), pow(scale, 2)),
                                     resize=True, keep_ratio=keep_ratio)


def compute_affine_matrix(in_shape,
                          out_shape,
                          crop=None,
                          degrees=0.0,
                          translate=(0.0, 0.0),
                          flip_h=False,
                          flip_v=False,
                          resize=False,
                          keep_ratio=False):
    """
    Similarity warp transformation of the image keeping center invariant.

    Args:
        in_shape (Sequence): the shape of the input image
        out_shape (Sequence): the shape of the output image
        crop (Sequence, optional): crop center location, width and height. The
            center location is relative to the center of the image. If
            :attr:`resize` is not ``True``, crop is simply a translation in the
            :attr:`in_shape` space.
        degrees (float or int, optional): degrees to rotate the crop.
            (default: ``(0.0)``)
        translate (Sequence, optional): horizontal and vertical translations.
            (default: ``(0.0, 0.0)``)
        flip_h (bool, optional): flip the image horizontally.
            (default: ``False``)
        flip_v (bool, optional): flip the image vertically.
            (default: ``False``)
        resize (bool, optional): resize the cropped image to fit the output's
            size. (default: ``False``)
        keep_ratio (bool, optional): match the smaller edge to the
            corresponding output edge size, keeping the aspect ratio after
            resize. Has no effect if :attr:`resize` is ``False``.
            (default: ``False``)
    """
    if crop is not None:
        T_crop_x, T_crop_y, crop_w, crop_h = crop
    else:
        T_crop_x, T_crop_y = 0, 0
        crop_w, crop_h = in_shape
    r = np.deg2rad(degrees)
    tx, ty = translate
    fh = 1 - 2 * float(flip_h)
    fv = 1 - 2 * float(flip_v)

    #
    # H = T_inshape*T_crop*R*S_resize*T_outshapeT
    #
    T_i_x = (in_shape[0] - 1) / 2
    T_i_y = (in_shape[1] - 1) / 2
    T_inshape = np.asarray([[fh, 0, T_i_x],
                            [0, fv, T_i_y],
                            [0, 0, 1]])
    T_crop = np.asarray([[1, 0, T_crop_x],
                         [0, 1, T_crop_y],
                         [0, 0, 1]])
    R = np.asarray([[+np.cos(r), -np.sin(r), 0],
                    [+np.sin(r), +np.cos(r), 0],
                    [0, 0, 1]])
    S_r_x = 1
    S_r_y = 1
    if resize:
        top_left, bot_right = R.dot([[-crop_w / 2, crop_w / 2],
                                     [-crop_h / 2, crop_h / 2],
                                     [1, 1]]).transpose()[:, 0:2]
        crop_w, crop_h = np.absolute(bot_right - top_left)
        S_r_x = crop_w / out_shape[0]
        S_r_y = crop_h / out_shape[1]
        if keep_ratio:
            scale_ratio = min(S_r_x, S_r_y)
            S_r_x = scale_ratio
            S_r_y = scale_ratio
    S_resize = np.asarray([[S_r_x, 0, 0],
                           [0, S_r_y, 0],
                           [0, 0, 1]])
    T_o_x = tx - (out_shape[0] - 1) / 2
    T_o_y = ty - (out_shape[1] - 1) / 2
    T_outshapeT = np.asarray([[1, 0, T_o_x],
                              [0, 1, T_o_y],
                              [0, 0, 1]])
    return T_inshape.dot(T_crop).dot(R).dot(S_resize).dot(T_outshapeT)
