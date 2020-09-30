### Introduction/Background 

It has been long known technique to convert colored images into grayscale, but it is a much more challenging task to hallucinate<sup>[3]</sup> colors from greyscale images. In this project, we aim to leverage the power of machine learning to extract meaningful information out of the semantics<sup>[1]</sup><sup>[2]</sup> of the image to regenerate a plausible colorized image. Some of its potential application includes colorization of historical images that were captured in greyscale. 

### Problem Definition 

We will be working with images in the CIELAB color space, which means that we will take images with only the Lightness channel as input and generate the 2 missing channels (A and B color channels) based on semantic information in the images. Because an object can take on multiple colors in nature (e.g. an apple can be green, red, or yellow)<sup>[4]</sup>, our goal is not to predict the original color pattern of the image. Instead, we want to generate a viable colorized image that looks realistic to the human eyes.  

### Methods 

##### Unsupervised 
We will use clustering algorithms to gain insight into our dataset. We also would use PCA to try to reduce dimensionality of the images and then feed it into our supervised learning method i.e. the Convolution Neural Network. 

##### Supervised
We will frame the problem as a multinomial classification with predefined AB color pairs for each pixel.  Our model of choice will be a Convolutional Neural Network<sup>[1]</sup><sup>[2]</sup><sup>[3]</sup><sup>[4]</sup> to extract semantic features and map it onto a per pixel probability distribution over all the AB pairs. We will then try a variety of techniques to choose the color per pixel, be it taking the color with the highest probability per pixel or taking a mode over neighboring pixels.  

### Potential results

As an outcome of this project, we aim to create realistic colorized pictures from gray scale images. To evaluate our results, we will use the metric of a Colorization Turing Test<sup>[3]</sup>, where we will ask human subjects to identify the artificially generated color image from the ground truth image. This is the best evaluation for our goal of generating realistic people that can “fool” people into thinking that it’s real. Other metrics that we can use includes the MAE of the predicted images and the ground truth images. 

### Discussion/Conclusions 
Due to the complex nature of the process of converting grayscale images to their full correct color image, our primary goal is to get some sort of semblance of the true color to the image, not necessarily the fully saturated version. One of the challenges we foresee is the imbalance<sup>[3]</sup> between low saturated pixels (majority) and the high saturated pixels (rare) in images. This might cause the output to be dominated by low saturation pixels and therefore cause the output image to look dull and grey-ish. 

### References 
1. Iizuka, S., Simo-Serra, E., & Ishikawa, H. (2016). Let there be color! ACM Transactions on Graphics, 35(4), 1-11. doi:10.1145/2897824.2925974 

2. Larsson, G., Maire, M., & Shakhnarovich, G. (2016). Learning Representations for Automatic Colorization. Computer Vision – ECCV 2016 Lecture Notes in Computer Science, 577-593. doi:10.1007/978-3-319-46493-0_35 

3. Zhang, R., Isola, P., & Efros, A. A. (2016). Colorful Image Colorization. Computer Vision – ECCV 2016 Lecture Notes in Computer Science, 649-666. doi:10.1007/978-3-319-46487-9_40 

4. Hwang, J. (2016). Image Colorization with Deep Convolutional Neural Networks. 
