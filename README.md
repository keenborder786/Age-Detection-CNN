# Age-Detection-CNN

<h1>Introduction</h1>
This is my personal project in which I have used the domain of transfer learning to make a convolutional neural network which 
detects the age group of the given image. Please note their are are four age categories:child,young,adult and elder. The CNN is design to
classify each image into one of these categories.

<h2>Architecture Used</h2>
<img src="https://user-images.githubusercontent.com/20038775/51536739-8f40bb80-1e72-11e9-8863-b79a347d80f4.png">
 <p>The Inception network was an important milestone in the development of CNN classifiers. Prior to its inception (pun intended), most popular CNNs just stacked convolution layers deeper and deeper, hoping to get better performance.</p>

<p>The Inception network, on the other hand, was complex (heavily engineered). It used a lot of tricks to push performance; both in terms of speed and accuracy. Its constant evolution led to the creation of several versions of the network.</p>
  
  <p>I combine Inception-ResNet-v2 with a additional Convulational and Pooling Layer. The Inception Layer was previously trained on ImageNet data set but I took its bottom layer and combine it with my own One-Layer simple CNN. Further, I did not freeze the Inception-ResNet Layer and allowed its training in correspondence to my dataset</p>


<h3>Training Dataset</h3>
I used the following dataset:<a href="http://chalearnlap.cvc.uab.es/dataset/26/description/">Data</a>

<h4>Files</h4>
<p>1-CNN-Main Notebook which shows the final implementation with detailed comments</p>
<p>2-DataGeneration.py-Python Scipt to convert Image data into inputtable format and also other preprocessings</p>

