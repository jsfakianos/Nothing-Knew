
## Network 5
**pooling convolution layers**

So that the old professor doesn't get upset, let's take off our cummerbund and bowtie. The network is stepped back to 3 convolution layers.

We will add a pooling layer, which either takes the average or the max value from a group of neighboring pixels. The layer would normally have more than one intended function. 

First, to decrease the size of the convolved images by pooling pixels. This lessens the computation load by decreasing the number of parameters and is thus a positive feature of the convolution layer compared to the fully connected neural network layer. Though, consistent with our desire to see the effects on our image, we pool the layers, but use a stride that leaves layer at the same size. This essentially negates the first intented function of pooling. 

Secondly, the pooling layer provides a means to recognize shapes in different positions of the image. This function may help in recognizing edges that were not initially trained.

Here are the results of network 5. They might be compared to the [3 convolution layer network](./page3.md) that did not employ pooling (the output images are provided side-by-side). 

![Conv image 1](/Network_05/conv5_01.gif "3 convolution layers with max_pooling")

![Output image 1](/Network_05/output5_01.gif "output image with pooling")
![Output image 2](/Network_05/output4b.gif "output image without pooling")

The images in the networks with pooling layers exhibit blurred convolutions, but the final output is quite crisp. 

The images also compare favorably to the ["Tuxedo milking the cow"](./page5.md) images. Thus, revealing that we have reduced out computations considerably, moving from 22,032 trainable parameters per step to 11,664 trainable parameters per step, yet still improved the final output of the network.




