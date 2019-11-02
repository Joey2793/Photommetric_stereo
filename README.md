# Photommetric_stereo

Photometric stereo is a technique to recover local surface orientations by using different images of an object
captured under different directional illumination and the same viewpoint.

Dataset: Apple, Pear, Elephant

<h2>Basic Algorithm:</h2>
<p>A. Calibration of illumination direction and intensity
In all test images sets, there will exist a metal sphere and a white matte Lamertian sphere with tested object for each image. From the metal sphere, the light direction can be estimated from the brightest point on the metal
sphere. </br>
L = 2(N · R)N − R, where N is the normal and R is the visual direction [0,0,1]</br>
With the Lambertain Model, The max pixel value of matte sphere is the light intensity.</br>
</p>
<p>
B. Estimation of normal and albedo
In the image set, each pixel intensity I = Li kd (N · L), where kd is surface albedo, L is the light direction, Li is light Intensity, N is the unit surface normal. 

In the equation, kd and N are unknown. To solve kd and N, at least three equations need to be formulated, which forms:
<img = "   "/>
