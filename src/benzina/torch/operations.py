# -*- coding: utf-8 -*-
import benzina.native
import numpy as np


class WarpTransform:
    """
    Interface class that represents a warp transformation as a combined rotation,
    scale, skew and translation 3 x 3 matrix. The transformation is called for each
    sample of a batch.
    """
    def __init__(self):
        pass
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
    def __call__(self, index, in_shape, out_shape, rng):
        return NotImplementedError('__call__ needs to be implemented in subclasses')
class OOBTransform:
    """
    Interface class that represents an out of bounds transformation. The
    transformation is called for each sample of a batch.
    """
    def __init__(self):
        pass
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
    def __call__(self, index, in_shape, out_shape, rng):
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
    def __init__(self):
        pass
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
    def __call__(self, index, in_shape, out_shape, rng):
        return NotImplementedError('__call__ needs to be implemented in subclasses')
class NormTransform:
    """
    Interface class that represents a normalization transformation. The transformation
    is called for each sample of a batch.
    """
    def __init__(self):
        pass
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
    def __call__(self, index, in_shape, out_shape, rng):
        return NotImplementedError('__call__ needs to be implemented in subclasses')
class BiasTransform:
    """
    Interface class that represents a bias transformation. The transformation
    is called for each sample of a batch.
    """
    def __init__(self):
        pass
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
    def __call__(self, index, in_shape, out_shape, rng):
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


class ConstantNormTransform(NormTransform):
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
    Represents a random similarity warp transformation to be applied on each
    sample of a batch.

    Args:
        scale (numeric or iterable of numerics, optional): the scale range to draw a
            random value from. If a single numeric, the value and it's inverse will
            be used to define the range (default: (+1.0,+1.0)).
        rotation (iterable of numerics, optional): the rotation range in radian to
            draw a random value from. If a single numeric, the value and it's inverse
            will be used to define the range (default: (-0.0,+0.0)).
        translation_x (iterable of numerics, optional): the translation on the x axis
            range to draw a random value from. If a single numeric, the value and it's
            inverse will be used to define the range (default: (-0,+0)).
        translation_y (iterable of numerics, optional): the translation on the y axis
            range to draw a random value from. If a single numeric, the value and it's
            inverse will be used to define the range (default: (-0,+0)).
        flip_h (iterable of numerics, optional): the horizontal flip probability range.
            Valid values are between 0 and 1 (default: (0.0)).
        flip_v (iterable of numerics, optional): the vertical flip probability range.
            Valid values are between 0 and 1 (default: (0.0)).
        autoscale (bool, optional): If ``True``, the sample will be automatically scaled
            to the output shape before applying the other transformations (default:
            False).
    """
    def __init__(self,
                 scale         = (+1.0,+1.0),
                 rotation      = (-0.0,+0.0),
                 translation_x = (-0,+0),
                 translation_y = (-0,+0),
                 flip_h        = 0.0,
                 flip_v        = 0.0,
                 autoscale     = False):
        if isinstance(scale, (int, float)):
            scale = float(scale)
            scale = min(scale, 1/scale)
            scale = (scale, 1/scale)
        
        if isinstance(rotation, (int, float)):
            rotation = float(rotation)
            rotation = max(rotation, -rotation)
            rotation = (-rotation, rotation)
        
        if isinstance(translation_x, (int, float)):
            translation_x = float(translation_x)
            translation_x = max(translation_x, -translation_x)
            translation_x = (-translation_x, translation_x)
        
        if isinstance(translation_y, (int, float)):
            translation_y = float(translation_y)
            translation_y = max(translation_y, -translation_y)
            translation_y = (-translation_y, translation_y)
        
        assert(scale[0] > 0 and scale[1] > 0)
        
        self.s         = scale
        self.r         = rotation
        self.tx        = translation_x
        self.ty        = translation_y
        self.fh        = float(flip_h)
        self.fv        = float(flip_v)
        self.autoscale = autoscale
    
    def __call__(self, index, in_shape, out_shape, rng):
        """Return a random similarity transformation."""
        return benzina.native.similarity(out_shape[0], out_shape[1],
                                         in_shape[0],  in_shape[1],
                                         self.s[0],    self.s[1],
                                         self.r[0],    self.r[1],
                                         self.tx[0],   self.tx[1],
                                         self.ty[0],   self.ty[1],
                                         self.fh,      self.fv,
                                         int(rng.randint(2**64, dtype=np.uint64)),
                                         self.autoscale)
