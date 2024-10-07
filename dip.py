import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.exposure import adjust_gamma
from skimage.filters import gaussian

# Load the image and normalize it.
image = cv2.imread('./test13.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

Ir, Ig, Ib = image[:,:,0], image[:,:,1], image[:,:,2]

Ir_mean, Ig_mean, Ib_mean = Ir.mean(), Ig.mean(), Ib.mean()

# Color Compensation.
alpha = 1
Irc = Ir + alpha * (Ig_mean - Ir_mean)*(1-Ir)*Ig
alpha = 0.5  
Ibc = Ib + alpha * (Ig_mean - Ib_mean)*(1-Ib)*Ig

# White Balance (Using Gray World Assumption)
Iwb = np.stack([Irc, Ig, Ibc], axis=-1)

Igamma = adjust_gamma(Iwb, gamma=2)

# Image Sharpening using Unsharp Masking.
sigma = 20
N = 30
Igauss = Iwb
for _ in range(N):
    Igauss = gaussian(Igauss, sigma=sigma, channel_axis=-1)
    Igauss = np.minimum(Iwb, Igauss)

# Calculate normalized difference and apply histogram equalization.
Norm = (Iwb - Igauss)
for channel in range(3):
    Norm[:,:,channel] = cv2.equalizeHist((Norm[:,:,channel] * 255).astype(np.uint8)) / 255.0

Isharp = (Iwb + Norm) / 2

# Calculate Weight Maps
def saliency_detection(image):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map = saliency.computeSaliency((image*255).astype(np.uint8))
    return saliency_map

# Laplacian Contrast Weight Map
def laplacian_weight_map(image):
    return np.sqrt(cv2.Laplacian(rgb2gray(image), cv2.CV_64F)**2)

# Calculate weights for images.
WC1 = laplacian_weight_map(Isharp)
WS1 = saliency_detection(Isharp)
WSAT1 = np.sqrt(np.mean((Isharp - rgb2gray(Isharp).reshape(Isharp.shape[0], Isharp.shape[1], 1))**2, axis=2))

WC2 = laplacian_weight_map(Igamma)
WS2 = saliency_detection(Igamma)
WSAT2 = np.sqrt(np.mean((Igamma - rgb2gray(Igamma).reshape(Igamma.shape[0], Igamma.shape[1], 1))**2, axis=2))

W1 = (WC1 + WS1 + WSAT1 + 0.1) / (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2 + 0.2)
W2 = (WC2 + WS2 + WSAT2 + 0.1) / (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2 + 0.2)

# Multi-scale Fusion using Gaussian and Laplacian pyramids
def gaussian_pyramid(image, levels):
    gp = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        gp.append(image)
    return gp

def laplacian_pyramid(image, levels):
    gp = gaussian_pyramid(image, levels)
    lp = [gp[-1]]
    for i in range(levels-1, 0, -1):
        size = (gp[i-1].shape[1], gp[i-1].shape[0])
        laplacian = cv2.pyrUp(gp[i], dstsize=size)
        laplacian = cv2.subtract(gp[i-1], laplacian)
        lp.append(laplacian)
    return lp[::-1]

# Construct pyramids for inputs and weights.
level = 5
Weight1 = gaussian_pyramid(W1, level)
Weight2 = gaussian_pyramid(W2, level)

R1 = laplacian_pyramid(Isharp[:, :, 0], level)
G1 = laplacian_pyramid(Isharp[:, :, 1], level)
B1 = laplacian_pyramid(Isharp[:, :, 2], level)
R2 = laplacian_pyramid(Igamma[:, :, 0], level)
G2 = laplacian_pyramid(Igamma[:, :, 1], level)
B2 = laplacian_pyramid(Igamma[:, :, 2], level)

# Fusion with consistent resizing
Rr, Rg, Rb = [], [], []
reference_shape = (Weight1[0].shape[1], Weight1[0].shape[0]) 

for k in range(level):
    resized_R1 = cv2.resize(R1[k], reference_shape)
    resized_G1 = cv2.resize(G1[k], reference_shape)
    resized_B1 = cv2.resize(B1[k], reference_shape)
    
    resized_R2 = cv2.resize(R2[k], reference_shape)
    resized_G2 = cv2.resize(G2[k], reference_shape)
    resized_B2 = cv2.resize(B2[k], reference_shape)
    
    resized_Weight1 = cv2.resize(Weight1[k], reference_shape)
    resized_Weight2 = cv2.resize(Weight2[k], reference_shape)
    
    Rr.append(resized_Weight1 * resized_R1 + resized_Weight2 * resized_R2)
    Rg.append(resized_Weight1 * resized_G1 + resized_Weight2 * resized_G2)
    Rb.append(resized_Weight1 * resized_B1 + resized_Weight2 * resized_B2)
    
# Reconstruct the final image from the pyramid.
def pyramid_reconstruct(pyramid):
    image = pyramid[0]
    for i in range(1, len(pyramid)):
        expected_size = (pyramid[i].shape[1], pyramid[i].shape[0])
        
        image = cv2.pyrUp(image)
        image = cv2.resize(image, expected_size) 
        
        image = cv2.add(image, pyramid[i])
    return image

R = pyramid_reconstruct(Rr)
G = pyramid_reconstruct(Rg)
B = pyramid_reconstruct(Rb)

fusion = np.stack([R, G, B], axis=-1)

# Display final image.
import matplotlib.pyplot as plt
plt.imshow(fusion)
plt.title('Multi-scale Fusion Output')
plt.axis('off')
plt.show()
