# Photommetric_stereo

Photometric stereo is a technique to recover local surface orientations by using different images of an object
captured under different directional illumination and the same viewpoint.

Dataset: Apple, Pear, Elephant

<h2>Basic Algorithm:</h2>
<p>A. Calibration of illumination direction and intensity</br>
In all test images sets, there will exist a metal sphere and a white matte Lamertian sphere with tested object for each image. From the metal sphere, the light direction can be estimated from the brightest point on the metal
sphere. </br>
L = 2(N · R)N − R, where N is the normal and R is the visual direction [0,0,1]</br>
With the Lambertain Model, The max pixel value of matte sphere is the light intensity.</br>
</p>
<p>
B. Estimation of normal and albedo</br>
In the image set, each pixel intensity I = Li kd (N · L), where kd is surface albedo, L is the light direction, Li is light Intensity, N is the unit surface normal. </br>

In the equation, kd and N are unknown. To solve kd and N, at least three equations need to be formulated, which forms:
![Img1](https://github.com/Joey2793/Photommetric_stereo/blob/master/img/img1.JPG)</br>
For each position of the image, we can get one kd and N, the result will get an albedo map and normal map.
</p>
<p>
C. Reconstruction depth based on normal</br>
The idea is that the each normal are perpendicular to tangent vectors on the surface. For each pixel (x,y), two equations can be formed for the bottom pixel (x, y + 1) and for the right pixel (x + 1,y): </br>
Vr = (x+1, y, zx+1,y) – (x,y,zx,y)      0 = N · Vr      Zx+1,y – Zx,y = -nx/nz</br>
Vb = (x, y+1, zx,y+1) – (x,y,zx,y)      0 = N · Vb      Zx,y+1 – Zx,y = -ny/nz
</p>

<h2>Result</h2>
<p>
![apple_albedo](https://github.com/Joey2793/Photommetric_stereo/blob/master/img/apple_albedo.png)
![apple_depth](https://github.com/Joey2793/Photommetric_stereo/blob/master/img/apple_depth.png)
![apple_normal](https://github.com/Joey2793/Photommetric_stereo/blob/master/img/apple_normal.png)
</p>
