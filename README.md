Main script: a_or_b.py

The original project was about distinguishing horses and humans. 
This one is generalized solution for binary neural network which distinguishes between any given two types of objects

What it does:
1. Opens pop-up window to choose folder with training data (and optionally validation data)
The folder must contain two sub-folders named by the data type. (e.g humans, horses)
2. Load training images from folder (shows some random examples of images)
3. Preprocess images using convolutions and pooling 
4. Feeds training images to train a neural network
5. After neural network is trained - continuously asked user to upload an image to recognize - until the user clicks "Cancel"
6. Gives the result of what is on picture


**Example of resources:**

Training pictures for horses:

 https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip
 
 Validation pictures for horses:
 
  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip
  
The archive has to be unpacked and chosen in the script