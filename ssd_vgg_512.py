# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Definition of 512 VGG-based SSD network.

This model was initially introduced in:
SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325

Two variants of the model are defined: the 300x300 and 512x512 models, the
latter obtaining a slightly better accuracy on Pascal VOC.

Usage:
    with slim.arg_scope(ssd_vgg.ssd_vgg()):
        outputs, end_points = ssd_vgg.ssd_vgg(inputs)
@@ssd_vgg
"""
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common
from nets import ssd_vgg_300

slim = tf.contrib.slim


# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    """Implementation of the SSD VGG-based 512 network.

    The default features layers with 512x512 image input are:
      conv4 ==> 64 x 64
      conv7 ==> 32 x 32
      conv8 ==> 16 x 16
      conv9 ==> 8 x 8
      conv10 ==> 4 x 4
      conv11 ==> 2 x 2
      conv12 ==> 1 x 1
    The default image size used to train this network is 512x512.
    """
    default_params = SSDParams(
        img_shape=(512, 512),
        num_classes=7,
        no_annotation_label=7,
        feat_layers=['Fblock3','Fblock4', 'Fblock5', 'Fblock6', 'block7'],
        feat_shapes=[(64, 64),(32, 32),(16, 16),(8, 8),(4, 4)],
        anchor_size_bounds=[0.10, 0.90],
        anchor_sizes=[(16,28),
                      (28, 48),
                      (48, 81),
                      (81, 156),
                      (156, 256)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3]],
        anchor_steps=[8,16,32, 64, 128],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1,-1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_512_vgg'):
        """Network definition.
        """
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #
    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        # if clipping_bbox is not None:
        #     rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def layer_shape(layer):
    """Returns the dimensions of a 4D layer tensor.
    Args:
      layer: A 4-D Tensor of shape `[height, width, channels]`.
    Returns:
      Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if layer.get_shape().is_fully_defined():
        return layer.get_shape().as_list()
    else:
        static_shape = layer.get_shape().with_rank(4).as_list()
        dynamic_shape = tf.unstack(tf.shape(layer), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(512, 512)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (512 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * 0.04, img_size * 0.1]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        shape = l.get_shape().as_list()[1:4]
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

# data_format='NCHW'
# funsion_axis=1
# training=True
data_format='NHWC'
funsion_axis=-1
training=False
def unit_neck(input,fillter,scope,shortchut,stride=1):
    with tf.variable_scope("mob_unit" + scope):
            output = slim.conv2d(input, fillter[0], [1, 1],activation_fn=tf.nn.selu,data_format=data_format)
            print('-----',scope, output.shape)
            output = slim.separable_conv2d(output, fillter[1], [3, 3], depth_multiplier=1,
                                           stride=stride,activation_fn=tf.nn.selu,data_format=data_format)
            print('-----',scope, output.shape)
            output = slim.conv2d(output, fillter[2], [1, 1], activation_fn=None,data_format=data_format)
            print('-----',scope, output.shape)
            output = slim.batch_norm(output, activation_fn=None, is_training=training,data_format=data_format)
            if shortchut==True:
                output=output+input
            return output

def block(net,num,count,scope,stride=2):
    net = unit_neck(net, num[0], scope+'-1', False, stride=stride)
    for i in range(count-1):
        net = unit_neck(net, num[1], scope+'-'+str(i+2), True)
    return net

def expand(net,shape):
    mark=0
    if mark==0:
        net = tf.transpose(net, [0, 2, 3, 1])
        net = tf.image.resize_nearest_neighbor(net, size=shape)
        net = tf.transpose(net, [0, 3, 1, 2])
    else:
        net = tf.image.resize_nearest_neighbor(net, size=shape)
    return net

# =========================================================================== #
# Functional definition of VGG-based SSD 512.
# =========================================================================== #
def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_512_vgg'):
    """SSD net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_512_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        print(inputs.shape)
        #inputs = tf.transpose(inputs, [0,2,3,1])
        with tf.variable_scope("RseNet50"):
            with tf.variable_scope("RseNet50"):
                net = slim.conv2d(inputs, 64, [3, 3], activation_fn=tf.nn.selu, stride=2,data_format=data_format)  # 256
                print(net.shape)
                end_points['block0'] = net
                print('block0')
                net = unit_neck(net, [32, 32, 32], 'block1', False)
                end_points['block1'] = net
                print('block1')
                net = block(net, [[96, 96, 48], [144, 144, 48]], 2, 'block1')  # 128
                end_points['block2'] = net
                print('block2')
                net = block(net, [[144, 144, 64], [192, 192, 64]], 3, 'block2')  # 64
                end_points['block3'] = net
                print('block3')
                net = block(net, [[144, 144, 64], [192, 192, 64]], 3, 'block3')  # 32
                end_points['block4'] = net
                print('block4')
                net = block(net, [[192, 192, 128], [384, 384, 128]], 4, 'block4')  # 16
                end_points['block5'] = net
                print('block5')
                net = block(net, [[384, 384, 192], [576, 576, 192]], 3, 'block5', stride=1)
                net = block(net, [[576, 576, 320], [960, 960, 320]], 3, 'block6')  # 8
                end_points['block6'] = net
                print('block6')
                net = block(net, [[576, 576, 320], [960, 960, 320]], 3, 'block7')  # 4
                end_points['block7'] = net
            end_point = 'Fblock6'
            with tf.variable_scope(end_point):
                p=slim.conv2d_transpose(net,256,[3,3],stride=2,activation_fn=tf.nn.selu,data_format=data_format)
                c=slim.conv2d(end_points['block6'],256,[3,3],activation_fn=tf.nn.selu,data_format=data_format)
                print(p.shape, c.shape)
                f=tf.concat([p,c],axis=funsion_axis)
                net = slim.conv2d(f, 256, [1, 1], scope='F6conv1x1', activation_fn=tf.nn.selu,data_format=data_format)
                print(net.shape)
            end_points[end_point] =  net
            end_point = 'Fblock5'
            with tf.variable_scope(end_point):
                p = slim.conv2d_transpose(net, 256, [3, 3], stride=2, activation_fn=tf.nn.selu,data_format=data_format)
                c = slim.conv2d(end_points['block5'], 256, [3, 3], activation_fn=tf.nn.selu,data_format=data_format)
                print(p.shape, c.shape)
                f = tf.concat([p, c], axis=funsion_axis)
                net = slim.conv2d(f, 256, [1, 1], scope='F5conv1x1', activation_fn=tf.nn.selu,data_format=data_format)
                print(net.shape)
            end_points[end_point] =  net
            end_point = 'Fblock4'
            with tf.variable_scope(end_point):
                p = slim.conv2d_transpose(net, 256, [3, 3], stride=2, activation_fn=tf.nn.selu,data_format=data_format)
                c = slim.conv2d(end_points['block4'], 256, [3, 3], activation_fn=tf.nn.selu,data_format=data_format)
                print(p.shape, c.shape)
                f = tf.concat([p, c], axis=funsion_axis)
                net = slim.conv2d(f, 256, [1, 1], scope='F4conv1x1', activation_fn=tf.nn.selu,data_format=data_format)
                print(net.shape)
            end_points[end_point] =  net
            end_point = 'Fblock3'
            with tf.variable_scope(end_point):
                p = slim.conv2d_transpose(net, 256, [3, 3], stride=2, activation_fn=tf.nn.selu,data_format=data_format)
                c = slim.conv2d(end_points['block3'], 256, [3, 3], activation_fn=tf.nn.selu,data_format=data_format)
                print(p.shape, c.shape)
                f = tf.concat([p, c], axis=funsion_axis)
                net = slim.conv2d(f, 256, [1, 1], scope='F3conv1x1', activation_fn=tf.nn.selu,data_format=data_format)
                print(net.shape)
            end_points[end_point] =  net
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_vgg_300.ssd_multibox_layer(end_points[layer],
                                                      num_classes,
                                                      anchor_sizes[i],
                                                      anchor_ratios[i],
                                                      normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)

        return predictions, localisations, logits, end_points
ssd_net.default_image_size = 512


def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
    """Caffe scope definition.

    Args:
      caffe_scope: Caffe scope object with loaded weights.

    Returns:
      An arg_scope.
    """
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc


# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    with tf.name_scope(scope, 'ssd_losses'):
        l_cross_pos = []
        l_cross_neg = []
        l_loc = []
        for i in range(len(logits)):
            dtype = logits[i].dtype
            with tf.name_scope('block_%i' % i):
                # Determine weights Tensor.
                pmask = gscores[i] > match_threshold
                fpmask = tf.cast(pmask, dtype)
                n_positives = tf.reduce_sum(fpmask)

                # Select some random negative entries.
                # n_entries = np.prod(gclasses[i].get_shape().as_list())
                # r_positive = n_positives / n_entries
                # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                # Negative mask.
                no_classes = tf.cast(pmask, tf.int32)
                predictions = slim.softmax(logits[i])
                nmask = tf.logical_and(tf.logical_not(pmask),
                                       gscores[i] > -0.5)
                fnmask = tf.cast(nmask, dtype)
                nvalues = tf.where(nmask,
                                   predictions[:, :, :, :, 0],
                                   1. - fnmask)
                nvalues_flat = tf.reshape(nvalues, [-1])
                # Number of negative entries to select.
                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
                n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
                max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                n_neg = tf.minimum(n_neg, max_neg_entries)

                val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                minval = val[-1]
                # Final negative mask.
                nmask = tf.logical_and(nmask, -nvalues > minval)
                fnmask = tf.cast(nmask, dtype)

                # Add cross-entropy loss.
                with tf.name_scope('cross_entropy_pos'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],labels=gclasses[i])
                    # gclass = tf.one_hot(gclasses[i], 34, 1.0, 0.0)
                    # logit = tf.multiply(tf.nn.softmax(logits[i]),gclass)
                    # weight= tf.multiply(tf.square(tf.subtract(1.0, logit)), 0.75)
                    # loss=tf.reduce_max(tf.nn.weighted_cross_entropy_with_logits(logits=logits[i], targets=gclass, pos_weight=weight),-1)
                    loss = tf.losses.compute_weighted_loss(loss, fpmask)
                    l_cross_pos.append(loss)

                with tf.name_scope('cross_entropy_neg'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],labels=no_classes)
                    # no_class = tf.one_hot(no_classes, 34, 1.0, 0.0)
                    # logit = tf.multiply(tf.nn.softmax(logits[i]),no_class)
                    # weight= tf.reduce_max(tf.multiply(tf.square(tf.subtract(1.0, logit)), 0.25),-1,keep_dims=True)
                    # loss = tf.reduce_max(tf.nn.weighted_cross_entropy_with_logits(logits=logits[i], targets=no_class, pos_weight=weight),-1)
                    loss = tf.losses.compute_weighted_loss(loss, fnmask)
                    l_cross_neg.append(loss)

                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * fpmask, axis=-1)
                    loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                    loss = tf.losses.compute_weighted_loss(loss, weights)
                    l_loc.append(loss)

        # Additional total losses...
        with tf.name_scope('total'):
            total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
            total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
            total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
            total_loc = tf.add_n(l_loc, 'localization')

            # Add to EXTRA LOSSES TF.collection
            tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
            tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
            tf.add_to_collection('EXTRA_LOSSES', total_cross)
            tf.add_to_collection('EXTRA_LOSSES', total_loc)
''' eval
print(inputs.shape)
        inputs=tf.transpose(inputs,[0,3,1,2])
        print(inputs.shape)
        with tf.variable_scope("RseNet50"):
            net = slim.conv2d(inputs, 64, [3, 3], activation_fn=tf.nn.selu, stride=2,data_format='NCHW')  # 256
            tnet=tf.transpose(net,[0,2,3,1])
            end_points['block0'] = tnet
            print('block0')
            net = unit_neck(net, [32, 32, 16], 'block1', False)
            tnet = tf.transpose(net, [0, 2, 3, 1])
            end_points['block1'] = tnet
            print('block1')
            net = block(net, [[96, 96, 24], [144, 144, 24]], 2, 'block1')#128
            tnet = tf.transpose(net, [0, 2, 3, 1])
            end_points['block2'] = tnet
            print('block2')
            net = block(net, [[144, 144, 32], [192, 192, 32]], 3, 'block2')  # 64
            tnet = tf.transpose(net, [0, 2, 3, 1])
            end_points['block3'] = tnet
            print('block3')
            net = block(net, [[144, 144, 32], [192, 192, 32]], 3, 'block3')  # 32
            tnet = tf.transpose(net, [0, 2, 3, 1])
            end_points['block4'] = tnet
            print('block4')
            net = block(net, [[192, 192, 64], [384, 384, 64]], 4, 'block4')  # 16
            tnet = tf.transpose(net, [0, 2, 3, 1])
            end_points['block5'] = tnet
            print('block5')
            net = block(net, [[384, 384, 96], [576, 576, 96]], 3, 'block5', stride=1)
            net = block(net, [[576, 576, 160], [960, 960, 160]], 3, 'block6')  # 8
            tnet = tf.transpose(net, [0, 2, 3, 1])
            end_points['block6'] = tnet
            print('block6')
            net = block(net, [[576, 576, 160], [960, 960, 160]], 3, 'block7') #4
            tnet = tf.transpose(net, [0, 2, 3, 1])
            end_points['block7'] = tnet
'''