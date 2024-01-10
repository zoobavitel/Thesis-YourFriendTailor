Hi everyone, welcome to my initial repository for my master's thesis: "Your Friend Tailor"

In today's age, online shopping will continue to increase. 
It has been shown that consumers are concerned with the lack of accurate sizing measurements given by fashion vendors.
My work intends to implement a standard sizing metric as well as utilizing State of the Art computer vision technology
to take one photo of a person and accurately gauge their garment measurements.

Starting off, the initial py file is testing the accuracy of MiDaS's depth estimation model in conjunction of
Json data from a dataset called Deepfashion2.

Right now, the python file takes the keypoint data from a json file, takes the corellated image and generates a depth map, then
takes the depth values and assigns them to each key point in a NumPy array. This is a starting point and will eventually allow me to 
take the data and turn it into point cloud data and model it in a 3D environment.

I'm still testing and trying to make sure everything works accordingly. 
The Deepfashion2 data must be requested from their respective authors/creators in order for this file to work.

Eventually, I want to make point cloud data from the images and use either PyMesh or Open3D to visualize this data to make sure it is accurate
before I start linking other ML models together. I'd like to combine the efforts of MiDaS and Lite-HRNet in order to visualize point cloud data from any image,
then do some geometry calculations to provide accurate garment measurements.
