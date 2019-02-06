
## Network 3
**finishing the convolution layer**

Neural Networks usually have an associated bias unit and an activation function for each layer. For the conv2d layer in Tensorflow, activation can be left at 'None' and the activation will be linear. This is okay for these experiments, but we'll employ a rectifier, or rectified linear unit (ReLU) to the layer. It is commonly used and appropriate for image models because it keeps everything above zero. A pixel value of (-10, -100, 10) is leveled to (0, 0, 10) anyway, so the ReLu is just there for completeness in case the model gets used for something else. 

Bias too... It isn't specifically needed here for this small network. Though, networks are mathematical tools, and mathmeticians have had a long-term love/hate relationship with zeros (ignore that black pixels are (0, 0, 0)). In neural networks, a zero value usually means 50/50, or I can't decide. So, consider the bias unit as the car dealer who asks you to decide between a red car and a blue car. You can't decide until the dealer says, "take the blue one", and you suddenly start thinking about why you don't want that color at all. 

So, the code in Network 4 has the bias and activation layers employed. And, as promised, it makes little difference in the output. The model still does not want to place the new shapes into the output image. 

In the images below, the training was carried out for 10,000 steps and the gif includes an image of every 100. The longer training was to show that the model really "wants" to put something into the new picture. The colors in the circles and background are less defined, the region where the new structure would be contains a little more variation, but the model just can't figure it out.

![Conv image 1](/Network_03/conv_01.gif "convolution layer mural evolution with bias and activation")
![Output image 1](/Network_03/output_01.gif "output image with bias and activation")


## Network 4
**stacking convolution layers**

Maybe the reason that the model cannot figure it out is that the capacity of its neural network is too small. The idea behind deep learning is that adding more layers to a neural network allows learning to be more complex and robust. So, let's deepen our model and see if that allows it to add the new edges. 

The first convolution layer will now feed into another convolution layer, and the output viewed with the deeper learning capacity. 

![Conv image 2](/Network_04/conv4a_01.gif "convolution layer mural evolution with bias and activation")
![Output image 2](/Network_04/output4a.gif "output image with bias and activation")

Now we are getting somewhere! The deep convolution model is almost able to draw in the new structure. Let's add more layers and see if the trend continues. 

Though, the gifs are getting bigger. So, let's show the results on independent pages. 

[Go here for 3 convolution layers.](./page3.md)

[Or here for 4 convolution layers.](./page4.md)







