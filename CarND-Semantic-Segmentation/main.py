import os.path
import tensorflow as tf
import math
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import tqdm


#--------------------------
# Meta
#--------------------------

NUMBER_CLASSES = 2
IMAGE_SHAPE = (160, 576)

EPOCHS = 100
BATCH_SIZE = 4
TRAINING_IMAGES = 289

LEARNING_RATE = 0.0001
DROPOUT = 0.5


DATA_DIR = './data'
RUNS_DIR = './runs'
TRAINING_DATA_DIR ='./data/data_road/training'
VGG_PATH = './data/vgg'


#--------------------------
# Assert
#--------------------------

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


#--------------------------
# Functions
#--------------------------


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'


    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)


    return w1, keep, l3, l4, l7

def conv_1x1(layer, layer_name, num_classes):
    """ Return the output of a 1x1 convolution of a layer """
    return tf.layers.conv2d(inputs = layer,
                          filters =  num_classes,
                          kernel_size = (1, 1),
                          strides = (1, 1),
                          name = layer_name,
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

def upsample(layer, k, s, layer_name, num_classes):
    """ Return the output of transpose convolution given kernel_size k and strides s """
    return tf.layers.conv2d_transpose(inputs = layer,
                                    filters = num_classes,
                                    kernel_size = (k, k),
                                    strides = (s, s),
                                    padding = 'same',
                                    name = layer_name,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # Apply a 1x1 convolution to encoder layers
    vgg_layer3_out_inv = conv_1x1(layer = vgg_layer3_out, layer_name = "layer3conv1x1", num_classes = num_classes)
    vgg_layer4_out_inv = conv_1x1(layer = vgg_layer4_out, layer_name = "layer4conv1x1", num_classes = num_classes)
    vgg_layer7_out_inv = conv_1x1(layer = vgg_layer7_out, layer_name = "layer7conv1x1", num_classes = num_classes)

    decoderlayer1 = upsample(layer = vgg_layer7_out_inv, k = 4, s = 2, layer_name = "decoderlayer1", num_classes = num_classes)
    decoderlayer2 = tf.add(decoderlayer1, vgg_layer4_out_inv, name = "decoderlayer2")

    decoderlayer3 = upsample(layer = decoderlayer2, k = 4, s = 2, layer_name = "decoderlayer3", num_classes = num_classes)
    decoderlayer4 = tf.add(decoderlayer3, vgg_layer3_out_inv, name = "decoderlayer4")

    decoderlayer_output = upsample(layer = decoderlayer4, k = 16, s = 8, layer_name = "decoderlayer_output", num_classes = num_classes)

    return decoderlayer_output

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    class_labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = class_labels))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    for epoch in range(1, epochs + 1):
        print("Epoch: " + str(epoch) + "/" + str(epochs))

        total_loss = []
        batch = get_batches_fn(batch_size)
        size = math.ceil(TRAINING_IMAGES / batch_size)

        for i, images_labels in tqdm(enumerate(batch), desc="Batch", total=size):
            images, labels = images_labels

            feed = {input_image: images,
                    correct_label: labels,
                    keep_prob: DROPOUT,
                    learning_rate: LEARNING_RATE}

            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed)
            total_loss.append(loss)

        mean_loss = sum(total_loss) / size

        print("Loss:  " + str(mean_loss) + "\n")

    return

def run_tests():
    # gathering the test
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_for_kitti_dataset(DATA_DIR)
    tests.test_train_nn(train_nn)


def run():
    correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUMBER_CLASSES])
    learning_rate = tf.placeholder(tf.float32)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # A function to get batches
    get_batches_fn = helper.gen_batch_function(TRAINING_DATA_DIR, IMAGE_SHAPE)

    with tf.Session() as session:
          
        # Returns the three layers, keep probability and input layer from the vgg architecture
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, VGG_PATH)

        # The resulting network architecture from adding a decoder on top of the given vgg model
        model_output = layers(layer3, layer4, layer7, NUMBER_CLASSES)

        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, NUMBER_CLASSES)

        # Initialize all variables
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        # Train the neural network
        train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn, 
               train_op, cross_entropy_loss, image_input,
               correct_label, keep_prob, learning_rate)

        # Run the model with the test images and save each painted output image (roads painted green)
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, session, IMAGE_SHAPE, logits, keep_prob, image_input)

if __name__ == "__main__":
    run_tests()
    run()