Hi everyone, welcome to my initial repository for my master's thesis: "Your Friend Tailor"

In today's age, online shopping will continue to increase. 
It has been shown that consumers are concerned with the lack of accurate sizing measurements from fashion vendors.
My work intends to implement a standard sizing metric and utilize State of the Art computer vision technology
to take one photo of a person and accurately gauge their garment measurements.

Starting off, the initial MiDaS py file was made to test the accuracy of MiDaS's depth estimation model + to see if our garment measurement calculations can be done on 2d photos.
I'm also using data collected from Depth In Humans, a medical dataset that measures depth via an xbox kinect.
I've also created my own dataset with my own photos and measurements that I have yet to test/implement.

Right now, I have two main files. 
I have a MiDaS py file that can take a deepfahsion2 image + annotations, create a depth map, segment the actor's shirt, and convert it from 2D to a 3d point cloud, and finally create a 3d mesh.
I have a Keypoint estimation py file that takes in deepfashion2 data and estimates keypoint placement. 

My current hiccups are:
- being able to properly scale the 2d annotations correctly onto the 3d mesh in the MiDaS file. I need the annotations to work in 3D so i can grab geodesic distance calculations of the side seams and waistline of the individual.
- I want to get it to work with the Keypoint RCNN + Resnet, but right now I just have Resnet 50 for the keypoint estimator file. 


I'm still testing and trying to make sure everything works accordingly. 
The Deepfashion2 data must be requested from their respective authors/creators in order for this file to work.

