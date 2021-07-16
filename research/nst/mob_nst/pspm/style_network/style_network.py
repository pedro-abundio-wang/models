"""Perceptual Losses for Real-Time Style Transfer and Super-Resolution.

Related papers
- https://arxiv.org/abs/1603.08155

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging

import numpy as np
from PIL import Image

import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import applications
from tensorflow.keras import preprocessing


class ReflectionPadding2D(layers.Layer):
    """
      2D Reflection Padding
      Attributes:
        - padding: (padding_width, padding_height) tuple
    """
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor,
                      [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]],
                      'REFLECT')


def load_image(path_to_image):
    max_dim = 512
    image = Image.open(path_to_image)
    long = max(image.size)
    scale = max_dim/long
    image = image.resize((round(image.size[0]*scale), round(image.size[1]*scale)), Image.ANTIALIAS)
    image = preprocessing.image.img_to_array(image)
    # We need to broadcast the image array such that it has a batch dimension
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_image(image):
    image = applications.vgg19.preprocess_input(image)
    return tf.convert_to_tensor(image)


def load_and_preprocess_image(path_to_image):
    # Util function to open, resize and format pictures into appropriate tensors
    image = load_image(path_to_image)
    image = applications.vgg19.preprocess_input(image)
    return tf.convert_to_tensor(image)


def deprocess_image(processed_img):
    # processed_img: numpy array
    image = processed_img.copy()
    if len(image.shape) == 4:
        image = np.squeeze(image, 0)
    assert len(image.shape) == 3, ("Input to deprocess image must be an image of "
                                   "dimension [1, height, width, channel] or [height, width, channel]")
    if len(image.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # Perform the inverse of the preprocessing step
    # Remove zero-center by mean pixel
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255).astype("uint8")
    return image


def residual_block(input_tensor,
                   filters):
    """
    Our residual blocks each contain two 3x3 convolutional layers with the same
    number of filters on both layer. We use the residual block design of Gross and
    Wilber [2] (shown in Figure 1), which differs from that of He et al [3] in that the
    ReLU nonlinearity following the addition is removed; this modified design was
    found in [2] to perform slightly better for image classification.
    """

    if backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=bn_axis)(x)

    x = layers.add([x, input_tensor])

    return x


def transformation_network():
    """
    Our image transformation networks roughly follow the architectural guidelines
    set forth by Radford et al [42]. We do not use any pooling layers, instead using
    strided and fractionally strided convolutions for in-network downsampling and
    upsampling. Our network body consists of five residual blocks [43] using the ar-
    chitecture of [44]. All non-residual convolutional layers are followed by spatial
    batch normalization [45] and ReLU nonlinearities with the exception of the out-
    put layer, which instead uses a scaled tanh to ensure that the output image has
    pixels in the range [0; 255]. Other than the first and last layers which use 9x9
    kernels, all convolutional layers use 3x3 kernels. The exact architectures of all
    our networks can be found in the supplementary material.

    Architecture:
    Layer                          Activation size
    ----------------------------------------------
    Input                          3 x 256 x 256
    Reflection Padding (40 x 40)   3 x 336 x 336
    32 x 9 x 9 conv, stride 1      32 x 336 x 336
    64 x 3 x 3 conv, stride 2      64 x 168 x 168
    128 x 3 x 3 conv, stride 2     128 x 84 x 84
    Residual block, 128 filters    128 x 80 x 80
    Residual block, 128 filters    128 x 76 x 76
    Residual block, 128 filters    128 x 72 x 72
    Residual block, 128 filters    128 x 68 x 68
    Residual block, 128 filters    128 x 64 x 64
    64 x 3 x 3 conv, stride 1/2    64 x 128 x 128
    32 x 3 x 3 conv, stride 1/2    32 x 256 x 256
    3 x 9 x 9 conv, stride 1       3 x 256 x 256
    """

    input_shape = (256, 256, 3)
    img_input = layers.Input(shape=input_shape)
    x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        channel_axis = 1
    else:  # channels_last
        channel_axis = -1

    x = ReflectionPadding2D(padding=(40, 40))(x)
    x = layers.Conv2D(
        filters=32,
        kernel_size=9,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)

    x = layers.Conv2DTranspose(
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(
        filters=32,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=3,
        kernel_size=9,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = tf.nn.tanh(x) * 150 + 255. / 2

    # Create model.
    model = models.Model(img_input, x, name='image_transformation_network')

    model.summary()

    return model


def loss_network():
    # load pre-trained vgg model
    model = applications.vgg16.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3))

    model.trainable = False

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # model (as a dict).
    feature_extractor = models.Model(inputs=model.inputs, outputs=outputs_dict)

    return feature_extractor


def style_network():

    input_shape = (256, 256, 3)
    img_input = layers.Input(shape=input_shape)
    x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)

    transformation_model = transformation_network()
    loss_model = loss_network()

    gen_img = transformation_model(x)
    features = loss_model(gen_img)

    model = models.Model(inputs=img_input, outputs=features, name='style_network')

    return model


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: Tensor of shape (1, height, width, channel) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (height * width * channel)

    Returns:
    - gram: Tensor of shape (channel, channel) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    _, height, width, channel = features.shape
    feature_maps = tf.reshape(features, (-1, channel))
    gram = tf.matmul(tf.transpose(feature_maps), feature_maps)
    if normalize:
        gram = tf.divide(gram, tf.cast(height * width * channel, gram.dtype))

    return gram


def style_loss(style, combination):
    style_gram = gram_matrix(style, normalize=True)
    combination_gram = gram_matrix(combination, normalize=True)
    return tf.square(tf.norm(style_gram - combination_gram))


def content_loss(content, combination):
    assert content.shape == combination.shape
    _, height, width, channel = content.shape
    return tf.square(tf.norm(combination - content)) / (height * width * channel)


def total_variation_loss(image):
    """
    Compute total variation loss.

    Inputs:
    - img: Tensor of shape (1, height, width, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    image = tf.squeeze(image)
    height, width, channel = image.shape

    img_col_start = tf.slice(image, [0, 0, 0], [height, width - 1, channel])
    img_col_end = tf.slice(image, [0, 1, 0], [height, width - 1, channel])
    img_row_start = tf.slice(image, [0, 0, 0], [height - 1, width, channel])
    img_row_end = tf.slice(image, [1, 0, 0], [height - 1, width, channel])
    return tf.square(tf.norm(img_col_end - img_col_start)) + tf.square(tf.norm(img_row_end - img_row_start))


def compute_loss(feature_extractor,
                 combination_image,
                 content_image,
                 style_image,
                 content_layer_name,
                 content_weight,
                 style_layer_names,
                 style_weights,
                 total_variation_weight):

    input_tensor = tf.concat(
        [content_image, style_image, combination_image], axis=0
    )

    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # content loss
    layer_features = features[content_layer_name]
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(content_image_features, combination_features)

    # style loss
    if style_layer_names is not None:
        for i, layer_name in enumerate(style_layer_names):
            layer_features = features[layer_name]
            style_weight = style_weights[i]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            loss += style_weight * style_loss(style_features, combination_features)

    # total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss


def compute_loss_and_grads(feature_extractor,
                           combination_image,
                           content_image,
                           style_image,
                           content_layer_name,
                           content_weight,
                           style_layer_names,
                           style_weights,
                           total_variation_weight):
    """
    ## Add a tf.function decorator to loss & gradient computation
    To compile it, and thus make it fast.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(feature_extractor,
                            combination_image,
                            content_image,
                            style_image,
                            content_layer_name,
                            content_weight,
                            style_layer_names,
                            style_weights,
                            total_variation_weight)
    grads = tape.gradient(loss, combination_image)
    return loss, grads


def run(style_image_path,
        content_layer_name,
        content_weight,
        style_layer_names,
        style_weights,
        total_variation_weight,
        result_prefix):

    """
    Training Details. Our style transfer networks are trained on the Microsoft
    COCO dataset [50]. We resize each of the 80k training images to 256x256 and
    train our networks with a batch size of 4 for 40,000 iterations, giving roughly
    two epochs over the training data. We use Adam [51] with a learning rate of
    1e-3. The output images are regularized with total variation regularization
    with a strength of between 1e-6 and 1e-4, chosen via cross-validation
    per style target. We do not use weight decay or dropout, as the model does
    not overfit within two epochs. For all style transfer experiments we compute
    feature reconstruction loss at layer relu2_2 and style reconstruction loss at
    layers relu1_2, relu2_2, relu3_3, and relu4_3 of the VGG-16 loss network.
    """

    content_image_path = 'elephant.png'
    style_image_path = 'starry_night.jpg'

    content_image = preprocess_image(content_image_path)
    style_image = preprocess_image(style_image_path)
    combination_image = tf.Variable(preprocess_image(content_image_path))

    feature_extractor = style_network()

    optimizer = optimizers.Adam(
        optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=100, decay_rate=0.96
        )
    )

    # most cases optimization converges to satisfactory results within 500 iterations.
    iterations = 40000

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    for i in range(1, iterations + 1):
        loss, grads = compute_loss_and_grads(
            feature_extractor,
            combination_image,
            content_image,
            style_image,
            content_layer_name,
            content_weight,
            style_layer_names,
            style_weights,
            total_variation_weight)
        optimizer.apply_gradients(zip(grads, feature_extractor.trainable_weights))
        # clipping the image y to the range [0, 255] at each iteration
        clipped = tf.clip_by_value(combination_image, min_vals, max_vals)
        combination_image.assign(clipped)

        if i % 100 == 0:
            logging.info("Iteration %d: loss=%.2f" % (i, loss))


def main(_):

    params = {
        'style_image_path': 'starry_night.jpg',
        'content_layer_name': 'block2_conv2',
        'content_weight': 0,
        'style_layer_names': [
            "block1_conv2",
            "block2_conv2",
            "block3_conv3",
            "block4_conv3"
        ],
        'style_weights': [0.25, 0.25, 0.25, 0.25],
        'total_variation_weight': 1e-5,
        'result_prefix': 'style_reconstructions_block5_conv1'
    }

    run(**params)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)

