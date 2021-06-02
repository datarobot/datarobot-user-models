# -*- coding: utf-8 -*-
"""

Copyright 2017 xuannianz github user

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

NOTICE: THIS FILE HAS BEEN MODIFIED BY DataRobot UNDER COMPLIANCE WITH THE APACHE 2.0 LICENCE FROM THE ORIGINAL WORK
OF THE OWNER github user xuannianz. THE FOLLOWING IS THE COPYRIGHT OF THE DOCUMENTS IN:
https://github.com/xuannianz/EfficientDet/tree/master/utils
"""

import functools
import numpy as np
from tensorflow import keras


_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get("backend", _KERAS_BACKEND)
    layers = kwargs.get("layers", _KERAS_LAYERS)
    models = kwargs.get("models", _KERAS_MODELS)
    utils = kwargs.get("utils", _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ["backend", "layers", "models", "utils"]:
            raise TypeError("Invalid keyword argument: %s", key)
    return backend, layers, models, utils


def inject_keras_modules(func):
    import keras

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs["backend"] = keras.backend
        kwargs["layers"] = keras.layers
        kwargs["models"] = keras.models
        kwargs["utils"] = keras.utils
        return func(*args, **kwargs)

    return wrapper


def inject_tfkeras_modules(func):
    import tensorflow.keras as tfkeras

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs["backend"] = tfkeras.backend
        kwargs["layers"] = tfkeras.layers
        kwargs["models"] = tfkeras.models
        kwargs["utils"] = tfkeras.utils
        return func(*args, **kwargs)

    return wrapper


def init_keras_custom_objects():
    import keras
    import efficientnet as model

    custom_objects = {
        "swish": inject_keras_modules(model.get_swish)(),
        "FixedDropout": inject_keras_modules(model.get_dropout)(),
    }

    keras.utils.generic_utils.get_custom_objects().update(custom_objects)


def init_tfkeras_custom_objects():
    import tensorflow.keras as tfkeras
    import efficientnet as model

    custom_objects = {
        "swish": inject_tfkeras_modules(model.get_swish)(),
        "FixedDropout": inject_tfkeras_modules(model.get_dropout)(),
    }

    tfkeras.utils.get_custom_objects().update(custom_objects)


def preprocess_image(image, image_size):
    # image, RGB
    image_width, image_height = image.size

    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = image.resize((resized_width, resized_height))
    image = np.asarray(image, dtype="float32")
    image /= 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode="constant")

    return image, scale


def reorder_vertexes(vertexes):
    """
    reorder vertexes as the paper shows, (top, right, bottom, left)
    Args:
        vertexes: np.array (4, 2), should be in clockwise
    Returns:
    """
    assert vertexes.shape == (4, 2)
    xmin, ymin = np.min(vertexes, axis=0)
    xmax, ymax = np.max(vertexes, axis=0)

    # determine the first point with the smallest y,
    # if two vertexes has same y, choose that with smaller x,
    ordered_idxes = np.argsort(vertexes, axis=0)
    ymin1_idx = ordered_idxes[0, 1]
    ymin2_idx = ordered_idxes[1, 1]
    if vertexes[ymin1_idx, 1] == vertexes[ymin2_idx, 1]:
        if vertexes[ymin1_idx, 0] <= vertexes[ymin2_idx, 0]:
            first_vertex_idx = ymin1_idx
        else:
            first_vertex_idx = ymin2_idx
    else:
        first_vertex_idx = ymin1_idx
    ordered_idxes = [(first_vertex_idx + i) % 4 for i in range(4)]
    ordered_vertexes = vertexes[ordered_idxes]
    # drag the point to the corresponding edge
    ordered_vertexes[0, 1] = ymin
    ordered_vertexes[1, 0] = xmax
    ordered_vertexes[2, 1] = ymax
    ordered_vertexes[3, 0] = xmin
    return ordered_vertexes


def postprocess_boxes(boxes, scale, height, width):
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes


class AnchorParameters:
    """
    The parameters that define how anchors are generated.
    Args
        sizes : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios : List of ratios to use per location in a feature map.
        scales : List of scales to use per location in a feature map.
    """

    def __init__(
        self,
        sizes=(32, 64, 128, 256, 512),
        strides=(8, 16, 32, 64, 128),
        ratios=(1, 0.5, 2),
        scales=(2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)),
    ):
        self.sizes = sizes
        self.strides = strides
        self.ratios = np.array(ratios, dtype=keras.backend.floatx())
        self.scales = np.array(scales, dtype=keras.backend.floatx())

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    # ratio=h/w
    ratios=np.array([1, 0.5, 2], keras.backend.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def layer_shapes(image_shape, model):
    """
    Compute layer shapes given input image shape and the model.
    Args
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is transformed in the pyramid.
    Returns
        A dictionary mapping layer names to image shapes.
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            input_shapes = [shape[inbound_layer.name] for inbound_layer in node.inbound_layers]
            if not input_shapes:
                continue
            shape[layer.name] = layer.compute_output_shape(
                input_shapes[0] if len(input_shapes) == 1 else input_shapes
            )

    return shape


def make_shapes_callback(model):
    """
    Make a function for getting the shape of the pyramid levels.
    """

    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """
    Guess shapes based on pyramid levels.
    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.
    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape, pyramid_levels=None, anchor_params=None, shapes_callback=None,
):
    """
    Generators anchors for a given shape.
    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.
    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    feature_map_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4), dtype=np.float32)
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales,
        )
        shifted_anchors = shift(feature_map_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors.astype(np.float32)


def shift(feature_map_shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size.
    Args
        feature_map_shape : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, feature_map_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, feature_map_shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
    ).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X scales w.r.t. a reference window.
    Args:
        base_size:
        ratios:
        scales:
    Returns:
    """
    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(np.repeat(scales, len(ratios))[None], (2, 1)).T

    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.tile(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.tile(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, scale_factors=None):
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    cxa = anchors[:, 0] + wa / 2.0
    cya = anchors[:, 1] + ha / 2.0

    w = gt_boxes[:, 2] - gt_boxes[:, 0]
    h = gt_boxes[:, 3] - gt_boxes[:, 1]
    cx = gt_boxes[:, 0] + w / 2.0
    cy = gt_boxes[:, 1] + h / 2.0
    # Avoid NaN in division and log below.
    ha += 1e-7
    wa += 1e-7
    h += 1e-7
    w += 1e-7
    tx = (cx - cxa) / wa
    ty = (cy - cya) / ha
    tw = np.log(w / wa)
    th = np.log(h / ha)
    if scale_factors:
        ty /= scale_factors[0]
        tx /= scale_factors[1]
        th /= scale_factors[2]
        tw /= scale_factors[3]
    targets = np.stack([ty, tx, th, tw], axis=1)
    return targets
