import io
import time
import cv2
import numpy as np
import tensorflow as tf
import moviepy.editor as mpy

def make_gif(images, fname, fps=5):
    duration = len(images)/fps
    def make_frame(t):
        return images[int(len(images)/duration*t)]
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=fps)

# Look at the filters
class network01():
    def __init__(self, myScope):
        self.input = tf.placeholder(shape=[None,136,136,3],
                                    dtype=tf.float32)
        self.conv1_weights = tf.get_variable(name=myScope+"_conv1_weights",
                                             shape=[6,6,3,12])
        self.conv1 = tf.nn.conv2d(input=self.input,
                                  filter=self.conv1_weights,
                                  strides=[1,1,1,1],      # strides are [batch, height, width, channels]
                                  padding="SAME",
                                  name=myScope+"_conv1")

        # split the filters into 4 different sets, a number divisible by 3, a set of images for each channel (RGB)
        # allows splitting [1, 136, 136, 12] into 4x[1, 136, 136, 3]
        self.sets = tf.split(value=self.conv1,
                             num_or_size_splits=int(12/3),
                             axis=3)
        self.output = tf.reduce_mean(self.sets, axis=0)


def implement_network_01():
    raw_image = cv2.imread(filename='./RGB01.png')
    raw_image = np.expand_dims(raw_image, axis=0)
    print('The shape of the input image is {0:}.'.format(raw_image.shape))
    tf.reset_default_graph()
    model = network01(myScope='test')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        new_image, output, filters = sess.run(fetches=[model.conv1, model.output, model.conv1_weights],
                                     feed_dict={model.input:raw_image})
    print('The shape of the conv1 layer is {0:}.'.format(new_image[0].shape))
    new_image_01 = np.reshape(a=new_image[0],
                              newshape=(new_image[0].shape[1], new_image[0].shape[0] * new_image[0].shape[2]),
                              order='F')
    print('The shape of the conv1 layer is {0:}.'.format(new_image_01.shape))
    cv2.imwrite(filename='./RGB01_filter.png',
                img=new_image_01)

    # add the raw_input image next to the output image
    output_processed = np.hstack((raw_image[0], output[0]))
    cv2.imwrite(filename="./RGB01_output.png",
                img=output[0])
    cv2.imwrite(filename="./RGB01_output_combined.png",
                img=output_processed)

    print(filters[0])

implement_network_01()

