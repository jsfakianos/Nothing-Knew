# Nothing Knew
I love watching code learn to perform tasks. From deep reinforcement learning to simple neural networks, the reward or the loss is an important key to understanding whether your hyperparameters are optimally set and your model is actually learning. Though, this can quickly lose it's appeal as a fun way to watch your model and it really offers little insight into how the model actually learns. A much more fun way to watch learning is to view the progress along all of the layers. This is **nothing new**. People have been doing this a very long time. Yet, here are some of my insights from walking back to the basics and watching a model evolve.

## background
Before Tensorflow was released, I wrote and assembled cells in Java to perform some neat operations. While models can crunch numbers for weeks on end, there is no guarantee that the layers are converging toward a useful inference tool. Plus, coding backpropagation is very tedious. To give myself some assurance that my code was working, I always used a dashboard with a graphical component to each cell. While I used the dashboard less after Tensorflow made everything much more accessible, I felt like revisiting some of the aspects I have missed in the post-Tensorflow convenience. Upon starting, I decided that I could easily make this a simple introduction into convolutional neural networks for those interested in learning machine learning but know nothing, like when I knew nothing and people helped me.

Disclaimer: *This convolutional neural network is not one that you would use for real work. I over simplified it by removing the big data aspect of machine learning (i use one image here), loading some simple shapes into an image, not pooling or reducing the layers, and asking whether I could get some insight into how a convolutional neural network learns. I also made a big assumption that the model learns one image that same way that it learns a million.*


## Network 1
**convolve without training**

The Network_01 folder demonstrates a simple convolutional layer. A placeholder was created for the input image, 12 (6x6x3) filters were initialized with a distribution around 0, and a 2D convolution layer was made to hold the resulting tensor. 12 filters were chosen because I wanted the number to be divisible by 3, for the number of channels (red, green, and blue). The width and height of the filter were chosen on intuition to be large enough to learn a feature, but small enough to be computationally efficient. The last parameter in the filter is the number of channels for the rgb image, 3.

![Input image](/Network_01/RGB01.png "Input image")

The Input image above was made to be saturated in the three red, green, and blue channels with simple shapes. When the image is passed to the model, the random filters are passed over the image in the designated strides, which were (suboptimally and ill-advisedly) used to preserve the shape on the original image. Also, note that pooling was not used, nor was bias in this first network. The convolution layer looked like this after the filters were passed over the input image. 

![Conv1 image](/Network_01/RGB01_filter.png "convolutional layer mural")

The convolution layer was then reassembled into a single rgb image by dividing the filters into 4 reds, 4 greens, and 4 blues, and then stacking the averages. The result is an image that vaguely looks like the input. Not much can be read into the output at this point. The filters were randomly initialized, and the resulted values of the filters were small (between ~0.1 and -0.1). So, if one of the filter values was 0.01 and it was applied to a saturated red pixel, the resulting pixel value would be 255 * 0.01 = 2 (int(2.55) = 2), which is close to black. Note that RGB images are additive, so the white background of the image is (255, 255, 255).

![Output image](/Network_01/RGB01_output_combined.png "Output image")


## Network 2
**let's start learning**

We can start training the filters, also called 'weights' or 'kernels', by adding a target image, a measure of the differences between the output and the target, and an optimizer that would try to minimize the differences. Since the output we saw above is so unlike the input image, I again kept things simple and asked if we could get the original image back out. The model is thus Dr. Frankenstein, and the output is his monster. During the training, we can clearly see the filters evolving over 1,000 consecutive steps. 

![Conv3 image](/Network_03/conv1e4.gif "convolutional layer mural evolution")


...and the output image ends up looking quite similar to the input image. 
![output 3 image](/Network_03/output1e4.gif "convolutional layer mural evolution")

What we would expect from the filters is that the resulting background pixels should all average very closely to (255, 255, 255). Then, the pixels in the red circle should be (255, 0, 0)... and so forth. In fact, if we look at the pixel values at the very center of the red circle, we see the values from filters 1, 4, 7, and 10, which become the red channel in the output, converge to an average of 250. The pixel at that spot actually has the value (250, 8, 6).
![chart image](/Network_03/chart1.png "training pixels on simple image")

**the learning rate**

For dramatic effect the learning rate of the network was set very low, 0.0001 or 1e-4. This hyperparameter value allowed the loss to be reduced slowly and training to likely proceed in a well behaved manner without prior knowledge of the difficulty. The value could likely be improved to converge closer to the perfect value and even faster, or it could be changed to a number that allows us to gain better insight of gradient descent through visualization. 
![chart image 2](/Network_03/chart2.png "learning rates")

In the 2 charts above, the learning rates were increased 100-fold. The first learning rate increase reduced the number of steps for reaching (255, 0, 0)+-2 to only 312 steps. The second increase reached (255, 0, 0)+-2 after about 210 steps. Though, the learning rate equal to 1.0 clearly exhibited "exploding gradients" and early in the training decided that the pixel we are tracking should be (9175, 9200, 9199). I think that the simplicity of the input image allowed the model to get back on track, but this would likely not be the case for a more complicated network or image. 

**can the model learn to draw**

What would happen if we made the image a little more complicated. At least complicated in the sense that some different edges (straight) and corners are introduced, and we introduce them so that they overlay the original circles? The model basically learns to produce a similar output, at the same rate as when the simpler input was provided. The difference in the loss value is larger when fed the more complicated image, but this seems to due to the greater number of edges in the latter image. The output images are fuzzy at edges in pictures. More edges leads to more fuzziness, which leads to higher loss value. We used the same learning rate (learning rate = 0.0001) as shown in the similar animations above. 

![Conv3 image 3](/Network_03/conv1e4compl.gif "convolutional layer mural evolution")
![Ouput image 3](/Network_03/output1e4compl.gif "more complicated input")

To my eyes, it looks like the intermediate colors hang on a little longer. However, that's just my eyes (and yours?). When plotted on a graph, the convergence is nearly identical. Trust me and I'll save your bandwidth. 

Anyway, we shouldn't lose focus on learning rates. We want to see if we can gain insights on how the convolutional layers are learning. Thus far it's been pretty simple. We can step it up by asking the model to morph the colors. What happens if the input image is different from the output image? The model still seems to be able to shift primary colors around. The loss ends up with a much higher value, which does go away with a more optimal learning rate.

![Conv3 image 4](/Network_03/conv1e4m1.gif "convolutional layer evolution during morph")
![Ouput image 4](/Network_03/output1e4m1.gif "morph primary colors")


The bigger difference comes when the model is asked to not just change colors, but to add edges into the image that were not present in the input. When the input is the simple image and the target is the more complicated image. The model indicates that it has reached it's limit. It does not seem to have the capacity to morph between non-contiguous colors and edges.

![Conv3 image 5](/Network_03/conv1e4m2.gif "convolutional layer evolution during edge morph")
![Ouput image 5](/Network_03/output1e4m2.gif "morph edges")

The model will be given a little more capacity in the following pages. Though, the animated gifs have become to much for this single page. Follow this link to continue. If it is not here, it's because I haven't uploaded yet.







