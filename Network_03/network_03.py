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


# Add a target and look at the evolution of the filters
class network03():
    def __init__(self, myScope, filters=12, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.input = tf.placeholder(shape=[None,136,136,3],
                                    dtype=tf.float32)
        self.conv1_weights = tf.get_variable(name=myScope+"_conv1_weights",
                                             shape=[6,6,3,filters])
        self.conv1 = tf.nn.conv2d(input=self.input,
                                  filter=self.conv1_weights,
                                  strides=[1,1,1,1],      # strides are [batch, height, width, channels]
                                  padding="SAME",
                                  name=myScope+"_conv1")

        # split the filters into 4 different sets, a number divisible by 3, a set of images for each channel (RGB)
        # allows splitting [1, 136, 136, 12] into 4x[1, 136, 136, 3]
        self.sets = tf.split(value=self.conv1,
                             num_or_size_splits=int(filters/3),
                             axis=3)
        self.output = tf.reduce_mean(self.sets, axis=0)



        self.target = tf.placeholder(shape=[None,136,136,3],
                                     dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(labels=self.target,
                                                 predictions=self.output)

        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)




def implement_network_03(input_image='./RGB01.png', target_image='./RGB01.png', output_implemetation='03a'):
    learning_rate = 1e-4
    raw_image = cv2.cvtColor(cv2.imread(filename=input_image), cv2.COLOR_BGR2RGB)
    raw_image = np.expand_dims(raw_image, axis=0)
    raw_target = cv2.cvtColor(cv2.imread(filename=target_image), cv2.COLOR_BGR2RGB)
    raw_target = np.expand_dims(raw_target, axis=0)
    tf.reset_default_graph()
    model = network03(myScope='test', filters=12, learning_rate=learning_rate)
    init = tf.global_variables_initializer()
    conv1_images = []
    output_images = []

    with tf.Session() as sess:
        sess.run(init)

        total_steps = 1000
        for i in range(total_steps+1):
            _, loss, conv1, output = sess.run(fetches=[model.trainer, model.loss, model.conv1, model.output],
                                                           feed_dict={model.input:raw_image,
                                                                      model.target:raw_target})
            #print('Step {0:4>d} - loss = {1:f}'.format(i, loss))
            print()
            # create a mural of the filtered images
            conv1_processed = np.reshape(a=conv1[0],
                                         newshape=(conv1[0].shape[1], conv1[0].shape[0] * conv1[0].shape[2]),
                                         order='F')

            for each in conv1[0][30][29]:
                print(each, end='\t')
            #print(conv1[0][42][41][0], conv1[0][42][41][1], conv1[0][42][41][0], conv1[0][42][41][0], )


            skip = 10
            # process filter images for gif
            if i%skip == 0:
                f = 'frame {0: 7,d}         loss {1: 10,.1f}'.format(i, loss)
                _, buffer = cv2.imencode(".png", conv1_processed)
                io_buf = io.BytesIO(buffer)
                decoded_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), 1)
                _, w, c = decoded_img.shape
                header = np.ones(shape=(20, w, c), dtype=np.uint8)
                header.fill(255)
                cv2.putText(header, f, (2,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
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
                header.fill(255)
                cv2.putText(header, f, (2,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

                footer_text_1 = "input"
                footer_text_2 = "output"
                footer_text_3 = "target"
                footer = np.ones(shape=(24, w, c), dtype=np.uint8)
                footer.fill(255)
                cv2.putText(footer, footer_text_1, (30,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
                cv2.putText(footer, footer_text_2, (168,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
                cv2.putText(footer, footer_text_3, (306,18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)

                output_processed = np.concatenate((header, output_processed, footer))

                _, buffer = cv2.imencode(".png", output_processed)
                io_buf = io.BytesIO(buffer)
                decoded_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), 1)
                output_images.append(decoded_img)
                if i+1 == total_steps:
                    for x in range(10):
                        output_images.append(decoded_img)
        make_gif(conv1_images, './output/network_' + output_implemetation + '_conv1_' + str(learning_rate) + '.gif')
        make_gif(output_images, './output/network_' + output_implemetation + '_output_' + str(learning_rate) + '.gif')


#implement_network_03(input_image='./RGB01.png', target_image='./RGB01.png', output_implemetation='03keep_1.0')
implement_network_03(input_image='./RGB01.png', target_image='./RGB03.png', output_implemetation='03m2keep_0.0001')

