
## Network 4
**stacking 4 convolution layers**


![Conv image 2](/Network_04/conv4c_01.gif "convolution layer mural evolution with bias and activation")
![Output image 2](/Network_04/output4c.gif "output image with bias and activation")

With 4 layers the deep learning is able to bridge the new structure across the red and green circles and reaches further outward from the blue circle. But it's just not there. 

Can you identify why the model is unable to accurately draw the newer image? 

A clue is in the way the model is drawing the rectangles. It uses concentric arcs to draw a line. So far, it has only learned to draw circles. The straight edges and corners are new to the model. When given a deeper capacity to improvise, it uses what it has learned creatively, literally. It stacks the arcs to give an approximation of a rectangle. It also adds another shrinking cricle in near the edge of the rectangle that overlaps with the original circles. 

Shall I continue... [add another layer.](./page5.md)
