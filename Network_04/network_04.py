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


# Add an additional layer to the network
class network_04():
    def __init__(self, myScope, filters=12, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.input = tf.placeholder(shape=[None,136,136,3],
                                    dtype=tf.float32)
        self.conv1_weights = tf.get_variable(name=myScope+"_conv1_weights",
                                             shape=[6,6,3,filters])
        self.conv1_raw = tf.nn.conv2d(input=self.input,
                                      filter=self.conv1_weights,
                                      strides=[1,1,1,1],      # strides are [batch, height, width, channels]
                                      padding="SAME",
                                      name=myScope+"_conv1")
        self.conv1_bias = tf.constant(0.1, shape=[filters])
        self.conv1 = tf.nn.relu(tf.nn.bias_add(value=self.conv1_raw,
                                    bias=self.conv1_bias))

        self.conv2_weights = tf.get_variable(name=myScope+"_conv2_weights",
                                             shape=[6,6,12,filters])
        self.conv2_raw = tf.nn.conv2d(input=self.conv1,
                                      filter=self.conv2_weights,
                                      strides=[1,1,1,1],      # strides are [batch, height, width, channels]
                                      padding="SAME",
                                      name=myScope+"_conv2")
        self.conv2_bias = tf.constant(0.1, shape=[filters])
        self.conv2 = tf.nn.relu(tf.nn.bias_add(value=self.conv2_raw,
                                    bias=self.conv2_bias))

        # split the filters into 4 different sets, a number divisible by 3, a set of images for each channel (RGB)
        # allows splitting [1, 136, 136, 12] into 4x[1, 136, 136, 3]
        self.sets = tf.split(value=self.conv2,
                             num_or_size_splits=int(filters/3),
                             axis=3)
        self.output = tf.reduce_mean(self.sets, axis=0)



        self.target = tf.placeholder(shape=[None,136,136,3],
                                     dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(labels=self.target,
                                                 predictions=self.output)

        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

# Add an additional layer to the network
class network_04_b():
    def __init__(self, myScope, filters=12, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.input = tf.placeholder(shape=[None,136,136,3],
                                    dtype=tf.float32)
        self.conv1_weights = tf.get_variable(name=myScope+"_conv1_weights",
                                             shape=[6,6,3,filters])
        self.conv1_raw = tf.nn.conv2d(input=self.input,
                                      filter=self.conv1_weights,
                                      strides=[1,1,1,1],      # strides are [batch, height, width, channels]
                                      padding="SAME",
                                      name=myScope+"_conv1")
        self.conv1_bias = tf.constant(0.1, shape=[filters])
        self.conv1 = tf.nn.relu(tf.nn.bias_add(value=self.conv1_raw,
                                    bias=self.conv1_bias))


        self.conv2_weights = tf.get_variable(name=myScope+"_conv2_weights",
                                             shape=[6,6,12,filters])
        self.conv2_raw = tf.nn.conv2d(input=self.conv1,
                                      filter=self.conv2_weights,
                                      strides=[1,1,1,1],      # strides are [batch, height, width, channels]
                                      padding="SAME",
                                      name=myScope+"_conv2")
        self.conv2_bias = tf.constant(0.1, shape=[filters])
        self.conv2 = tf.nn.relu(tf.nn.bias_add(value=self.conv2_raw,
                                    bias=self.conv2_bias))


        self.conv3_weights = tf.get_variable(name=myScope+"_conv3_weights",
                                             shape=[6,6,12,filters])
        self.conv3_raw = tf.nn.conv2d(input=self.conv2,
                                      filter=self.conv3_weights,
                                      strides=[1,1,1,1],      # strides are [batch, height, width, channels]
                                      padding="SAME",
                                      name=myScope+"_conv3")
        self.conv3_bias = tf.constant(0.1, shape=[filters])
        self.conv3 = tf.nn.relu(tf.nn.bias_add(value=self.conv3_raw,
                                    bias=self.conv3_bias))

        # split the filters into 4 different sets, a number divisible by 3, a set of images for each channel (RGB)
        # allows splitting [1, 136, 136, 12] into 4x[1, 136, 136, 3]
        self.sets = tf.split(value=self.conv3,
                             num_or_size_splits=int(filters/3),
                             axis=3)
        self.output = tf.reduce_mean(self.sets, axis=0)



        self.target = tf.placeholder(shape=[None,136,136,3],
                                     dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(labels=self.target,
                                                 predictions=self.output)

        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

# Add an additional layer to the network
class network_04_c():
    def __init__(self, myScope, filters=12, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.input = tf.placeholder(shape=[None,136,136,3],
                                    dtype=tf.float32)
        self.conv1_weights = tf.get_variable(name=myScope+"_conv1_weights",
                                             shape=[6,6,3,filters])
        self.conv1_raw = tf.nn.conv2d(input=self.input,
                                      filter=self.conv1_weights,
                                      strides=[1,1,1,1],      # strides are [batch, height, width, channels]
                                      padding="SAME",
                                      name=myScope+"_conv1")
        self.conv1_bias = tf.constant(0.1, shape=[filters])
        self.conv1 = tf.nn.relu(tf.nn.bias_add(value=self.conv1_raw,
                                    bias=self.conv1_bias))

        self.conv2_weights = tf.get_variable(name=myScope+"_conv2_weights",
                                             shape=[6,6,12,filters])
        self.conv2_raw = tf.nn.conv2d(input=self.conv1,
                                      filter=self.conv2_weights,
                                      strides=[1,1,1,1],      # strides are [batch, height, width, channels]
                                      padding="SAME",
                                      name=myScope+"_conv2")
        self.conv2_bias = tf.constant(0.1, shape=[filters])
        self.conv2 = tf.nn.relu(tf.nn.bias_add(value=self.conv2_raw,
                                    bias=self.conv2_bias))

        self.conv3_weights = tf.get_variable(name=myScope+"_conv3_weights",
                                             shape=[6,6,12,filters])
        self.conv3_raw = tf.nn.conv2d(input=self.conv2,
                                      filter=self.conv3_weights,
                                      strides=[1,1,1,1],      # strides are [batch, height, width, channels]
                                      padding="SAME",
                                      name=myScope+"_conv3")
        self.conv3_bias = tf.constant(0.1, shape=[filters])
        self.conv3 = tf.nn.relu(tf.nn.bias_add(value=self.conv3_raw,
                                    bias=self.conv3_bias))

        self.conv4_weights = tf.get_variable(name=myScope+"_conv4_weights",
                                             shape=[6,6,filters,filters])
        self.conv4_raw = tf.nn.conv2d(input=self.conv3,
                                      filter=self.conv4_weights,
                                      strides=[1,1,1,1],      # strides are [batch, height, width, channels]
                                      padding="SAME",
                                      name=myScope+"_conv4")
        self.conv4_bias = tf.constant(0.1, shape=[filters])
        self.conv4 = tf.nn.relu(tf.nn.bias_add(value=self.conv4_raw,
                                    bias=self.conv4_bias))

        # split the filters into 4 different sets, a number divisible by 3, a set of images for each channel (RGB)
        # allows splitting [1, 136, 136, 12] into 4x[1, 136, 136, 3]
        self.sets = tf.split(value=self.conv4,
                             num_or_size_splits=int(filters/3),
                             axis=3)
        self.output = tf.reduce_mean(self.sets, axis=0)



        self.target = tf.placeholder(shape=[None,136,136,3],
                                     dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(labels=self.target,
                                                 predictions=self.output)

        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


def implement_network_04(input_image='./RGB01.png', target_image='./RGB01.png', output_implemetation='04a'):
    learning_rate = 1e-4
    raw_image = cv2.cvtColor(cv2.imread(filename=input_image), cv2.COLOR_BGR2RGB)
    raw_image = np.expand_dims(raw_image, axis=0)
    raw_target = cv2.cvtColor(cv2.imread(filename=target_image), cv2.COLOR_BGR2RGB)
    raw_target = np.expand_dims(raw_target, axis=0)
    tf.reset_default_graph()


    model = network_04(myScope='Nothing_knew', filters=12, learning_rate=learning_rate)


    init = tf.global_variables_initializer()
    conv1_images = []
    output_images = []

    with tf.Session() as sess:
        sess.run(init)

        total_steps = 10000
        for i in range(total_steps+1):
            _, loss, conv1, conv2, output = sess.run(fetches=[model.trainer,
                                                                                   model.loss,
                                                                                   model.conv1,
                                                                                   model.conv2,
                                                                                   model.output],
                                              feed_dict={model.input:raw_image,
                                                         model.target:raw_target})
            print('Step {0:4>d} - loss = {1:f}'.format(i, loss))

            # create a mural of the filtered images
            conv1_processed = np.reshape(a=conv1[0],
                                         newshape=(conv1[0].shape[1], conv1[0].shape[0] * conv1[0].shape[2]),
                                         order='F')
            conv2_processed = np.reshape(a=conv2[0],
                                         newshape=(conv2[0].shape[1], conv2[0].shape[0] * conv2[0].shape[2]),
                                         order='F')

            conv_processed = np.vstack((conv1_processed, conv2_processed))

            skip = 100
            # process filter images for gif
            if i%skip == 0:
                f = 'frame {0: 7,d}         loss {1: 10,.1f}'.format(i, loss)
                _, buffer = cv2.imencode(".png", conv_processed)
                io_buf = io.BytesIO(buffer)
                decoded_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), 1)
                _, w, c = decoded_img.shape
                header = np.ones(shape=(20, w, c), dtype=np.uint8)
                cv2.putText(header, f, (2,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                decoded_img = np.concatenate((header, decoded_img))
                conv1_images.append(decoded_img)
                if i+1 == total_steps:
                    for x in range(10):
                        conv1_images.append(decoded_img)

                # process output image for gif
                # add the raw_input image next to the output image
                if input_image == target_image:
                    output_processed = np.hstack((raw_image[0], output[0]))
                else:
                    output_processed = np.hstack((raw_image[0], output[0], raw_target[0]))

                _, w, c = output_processed.shape
                header = np.ones(shape=(20, w, c), dtype=np.uint8)
                cv2.putText(header, f, (2,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                output_processed = np.concatenate((header, output_processed))

                footer_text_1 = "input"
                footer_text_2 = "output"
                footer_text_3 = "target"
                footer = np.ones(shape=(24, w, c), dtype=np.uint8)
                footer.fill(255)
                cv2.putText(footer, footer_text_1, (30,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
                cv2.putText(footer, footer_text_2, (168,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
                cv2.putText(footer, footer_text_3, (306,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)

                _, buffer = cv2.imencode(".png", output_processed)
                io_buf = io.BytesIO(buffer)
                decoded_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), -1)
                output_images.append(decoded_img)
                if i+1 == total_steps:
                    for x in range(10):
                        output_images.append(decoded_img)

        make_gif(conv1_images, './output/network_' + output_implemetation + '_conv1_' + str(learning_rate) + '.gif')
        make_gif(output_images, './output/network_' + output_implemetation + '_output_' + str(learning_rate) + '.gif')

        raw_image = cv2.cvtColor(cv2.imread(filename='./RGB01.png'), cv2.COLOR_BGR2RGB)
        raw_image = np.expand_dims(raw_image, axis=0)
        output = sess.run(fetches=[model.output], feed_dict={model.input:raw_image})
        cv2.imwrite(filename='./output1.png', img=output[0][0])


def implement_network_04b(input_image='./RGB01.png', target_image='./RGB01.png', output_implemetation='04a'):
    learning_rate = 1e-4
    raw_image = cv2.cvtColor(cv2.imread(filename=input_image), cv2.COLOR_BGR2RGB)
    raw_image = np.expand_dims(raw_image, axis=0)
    raw_target = cv2.cvtColor(cv2.imread(filename=target_image), cv2.COLOR_BGR2RGB)
    raw_target = np.expand_dims(raw_target, axis=0)
    tf.reset_default_graph()


    model = network_04_b(myScope='Nothing_knew', filters=12, learning_rate=learning_rate)


    init = tf.global_variables_initializer()
    conv1_images = []
    output_images = []

    with tf.Session() as sess:
        sess.run(init)

        total_steps = 10000
        for i in range(total_steps+1):
            _, loss, conv1, conv2, conv3, output = sess.run(fetches=[model.trainer,
                                                              model.loss,
                                                              model.conv1,
                                                              model.conv2,
                                                              model.conv3,
                                                              model.output],
                                                     feed_dict={model.input:raw_image,
                                                                model.target:raw_target})
            print('Step {0:4>d} - loss = {1:f}'.format(i, loss))

            # create a mural of the filtered images
            conv1_processed = np.reshape(a=conv1[0],
                                         newshape=(conv1[0].shape[1], conv1[0].shape[0] * conv1[0].shape[2]),
                                         order='F')
            conv2_processed = np.reshape(a=conv2[0],
                                         newshape=(conv2[0].shape[1], conv2[0].shape[0] * conv2[0].shape[2]),
                                         order='F')
            conv3_processed = np.reshape(a=conv3[0],
                                         newshape=(conv3[0].shape[1], conv3[0].shape[0] * conv3[0].shape[2]),
                                         order='F')

            conv_processed = np.vstack((conv1_processed, conv2_processed, conv3_processed))

            skip = 100
            # process filter images for gif
            if i%skip == 0:
                f = 'frame {0: 7,d}         loss {1: 10,.1f}'.format(i, loss)
                _, buffer = cv2.imencode(".png", conv_processed)
                io_buf = io.BytesIO(buffer)
                decoded_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), 1)
                _, w, c = decoded_img.shape
                header = np.ones(shape=(20, w, c), dtype=np.uint8)
                cv2.putText(header, f, (2,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                decoded_img = np.concatenate((header, decoded_img))
                conv1_images.append(decoded_img)
                if i+1 == total_steps:
                    for x in range(10):
                        conv1_images.append(decoded_img)

                # process output image for gif
                # add the raw_input image next to the output image
                if input_image == target_image:
                    output_processed = np.hstack((raw_image[0], output[0]))
                else:
                    output_processed = np.hstack((raw_image[0], output[0], raw_target[0]))

                _, w, c = output_processed.shape
                header = np.ones(shape=(20, w, c), dtype=np.uint8)
                cv2.putText(header, f, (2,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                output_processed = np.concatenate((header, output_processed))

                footer_text_1 = "input"
                footer_text_2 = "output"
                footer_text_3 = "target"
                footer = np.ones(shape=(24, w, c), dtype=np.uint8)
                footer.fill(255)
                cv2.putText(footer, footer_text_1, (30,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
                cv2.putText(footer, footer_text_2, (168,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
                cv2.putText(footer, footer_text_3, (306,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)

                _, buffer = cv2.imencode(".png", output_processed)
                io_buf = io.BytesIO(buffer)
                decoded_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), -1)
                output_images.append(decoded_img)
                if i+1 == total_steps:
                    for x in range(10):
                        output_images.append(decoded_img)

        make_gif(conv1_images, './output/network_' + output_implemetation + '_conv1_' + str(learning_rate) + '.gif')
        make_gif(output_images, './output/network_' + output_implemetation + '_output_' + str(learning_rate) + '.gif')

        raw_image = cv2.cvtColor(cv2.imread(filename='./RGB01.png'), cv2.COLOR_BGR2RGB)
        raw_image = np.expand_dims(raw_image, axis=0)
        output = sess.run(fetches=[model.output], feed_dict={model.input:raw_image})
        cv2.imwrite(filename='./output1.png', img=output[0][0])


def implement_network_04c(input_image='./RGB01.png', target_image='./RGB01.png', output_implemetation='04a'):
    learning_rate = 1e-4
    raw_image = cv2.cvtColor(cv2.imread(filename=input_image), cv2.COLOR_BGR2RGB)
    raw_image = np.expand_dims(raw_image, axis=0)
    raw_target = cv2.cvtColor(cv2.imread(filename=target_image), cv2.COLOR_BGR2RGB)
    raw_target = np.expand_dims(raw_target, axis=0)
    tf.reset_default_graph()


    model = network_04_c(myScope='Nothing_knew', filters=12, learning_rate=learning_rate)


    init = tf.global_variables_initializer()
    conv1_images = []
    output_images = []

    with tf.Session() as sess:
        sess.run(init)

        total_steps = 10000
        for i in range(total_steps+1):
            _, loss, conv1, conv2, conv3, conv4, output = sess.run(fetches=[model.trainer,
                                                                     model.loss,
                                                                     model.conv1,
                                                                     model.conv2,
                                                                     model.conv3,
                                                                     model.conv4,
                                                                     model.output],
                                                            feed_dict={model.input:raw_image,
                                                                       model.target:raw_target})
            print('Step {0:4>d} - loss = {1:f}'.format(i, loss))

            # create a mural of the filtered images
            conv1_processed = np.reshape(a=conv1[0],
                                         newshape=(conv1[0].shape[1], conv1[0].shape[0] * conv1[0].shape[2]),
                                         order='F')
            conv2_processed = np.reshape(a=conv2[0],
                                         newshape=(conv2[0].shape[1], conv2[0].shape[0] * conv2[0].shape[2]),
                                         order='F')
            conv3_processed = np.reshape(a=conv3[0],
                                         newshape=(conv3[0].shape[1], conv3[0].shape[0] * conv3[0].shape[2]),
                                         order='F')
            conv4_processed = np.reshape(a=conv4[0],
                                         newshape=(conv4[0].shape[1], conv4[0].shape[0] * conv4[0].shape[2]),
                                         order='F')

            conv_processed = np.vstack((conv1_processed, conv2_processed, conv3_processed, conv4_processed))

            skip = 100
            # process filter images for gif
            if i%skip == 0:
                f = 'frame {0: 7,d}         loss {1: 10,.1f}'.format(i, loss)
                _, buffer = cv2.imencode(".png", conv_processed)
                io_buf = io.BytesIO(buffer)
                decoded_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), 1)
                _, w, c = decoded_img.shape
                header = np.ones(shape=(20, w, c), dtype=np.uint8)
                cv2.putText(header, f, (2,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                decoded_img = np.concatenate((header, decoded_img))
                conv1_images.append(decoded_img)
                if i+1 == total_steps:
                    for x in range(10):
                        conv1_images.append(decoded_img)

                # process output image for gif
                # add the raw_input image next to the output image
                if input_image == target_image:
                    output_processed = np.hstack((raw_image[0], output[0]))
                else:
                    output_processed = np.hstack((raw_image[0], output[0], raw_target[0]))

                _, w, c = output_processed.shape
                header = np.ones(shape=(20, w, c), dtype=np.uint8)
                cv2.putText(header, f, (2,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                output_processed = np.concatenate((header, output_processed))

                footer_text_1 = "input"
                footer_text_2 = "output"
                footer_text_3 = "target"
                footer = np.ones(shape=(24, w, c), dtype=np.uint8)
                footer.fill(255)
                cv2.putText(footer, footer_text_1, (30,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
                cv2.putText(footer, footer_text_2, (168,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
                cv2.putText(footer, footer_text_3, (306,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)

                _, buffer = cv2.imencode(".png", output_processed)
                io_buf = io.BytesIO(buffer)
                decoded_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), -1)
                output_images.append(decoded_img)
                if i+1 == total_steps:
                    for x in range(10):
                        output_images.append(decoded_img)

        make_gif(conv1_images, './output/network_' + output_implemetation + '_conv1_' + str(learning_rate) + '.gif')
        make_gif(output_images, './output/network_' + output_implemetation + '_output_' + str(learning_rate) + '.gif')

        raw_image = cv2.cvtColor(cv2.imread(filename='./RGB01.png'), cv2.COLOR_BGR2RGB)
        raw_image = np.expand_dims(raw_image, axis=0)
        output = sess.run(fetches=[model.output], feed_dict={model.input:raw_image})
        cv2.imwrite(filename='./output1.png', img=output[0][0])

#implement_network_04c(input_image='./RGB01.png', target_image='./RGB03.png', output_implemetation='04_c_0.0001')

