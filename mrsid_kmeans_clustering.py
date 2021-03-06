# Sarah Nassar (Queen's University), February 2021

# in command prompt (python directory):
#pip3 install numpy
#pip3 install opencv-python
#pip3 install matplotlib

from osgeo import gdal
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def cluster_image(no_of_clusters, no_of_iterations, image_shape, image_values):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, no_of_iterations, 1)
    compactness, labels, centers = cv2.kmeans(image_values, no_of_clusters, None, criteria, 1000, cv2.KMEANS_PP_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    print('centers.shape, centers: ', centers.shape, centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_data = centers[labels]
    # reshape back to the original image dimension
    segmented_image = segmented_data.reshape(image_shape)
    cv2.imwrite('C://...//clusters//clustered_image.tif', segmented_image) #make a new folder named "clusters" and complete the directory
    # show the image
    plt.imshow(segmented_image)
    plt.show()
    return segmented_data, labels

def extract_individual_clusters(cluster_no, segmented_data, labels, image_shape):
    segmented_image_data = np.copy(segmented_data)
    segmented_image_data[labels != cluster_no] = [255, 255, 255] #white
    segmented_image_data[labels == cluster_no] = [0, 0, 0] #black
    segmented_image = segmented_image_data.reshape(image_shape)
    cv2.imwrite('C://...//clusters//image_cluster'+str(cluster_no)+'.tif', segmented_image) #complete the directory to the new folder, "clusters" 
    plt.imshow(segmented_image)
    plt.show()

fn = "C://...//image.sid" #complete the directory
ds = gdal.Open(fn, gdal.GA_ReadOnly)
if ds is None:
    print('Could not open ' + fn)
    sys.exit(1)

#####################################################################
#for testing with lower memory:

cols = ds.RasterXSize
rows = ds.RasterYSize
bands = ds.RasterCount
print('cols: ', cols)
print('rows: ', rows)
print('bands: ', bands)

band1 = ds.GetRasterBand(1)
band2 = ds.GetRasterBand(2)
band3 = ds.GetRasterBand(3)

cols1 = int(cols/5) #can change
rows1 = int(rows/5) #can change
xoff = 8000 #can change
yoff = 4000 #can change
print('cols1, rows1, xoff, yoff: ', cols1, rows1, xoff, yoff)

b1 = band1.ReadAsArray(xoff, yoff, cols1, rows1)
b2 = band2.ReadAsArray(xoff, yoff, cols1, rows1)
b3 = band3.ReadAsArray(xoff, yoff, cols1, rows1)
image = (np.dstack((b1, b2, b3))).astype(np.uint8)

plt.imshow(image)
plt.show()

image_data = image.reshape((-1, 3))
image_data = np.float32(image_data)

#####################################################################

'''
#for testing with whole image (works on 16GB-memory computer, comment the code between the lines of #)

image = ds.ReadAsArray().astype(np.uint8)
print('image.shape: [channels first] ', image.shape)
image = np.moveaxis(image, 0, 2)  # change channels first to channels last
print('image.shape: [channels last]', image.shape)

plt.imshow(image)
plt.show()

image_data = image.reshape((-1, 3))
image_data = np.float32(image_data)
print('image_data.shape: ', image_data.shape)
'''

no_of_clusters = 3 # can change
no_of_iterations = 3 # can change
print('no_of_clusters, no_of_iterations: ', no_of_clusters, no_of_iterations)
segmented_image_data, labels = cluster_image(no_of_clusters, no_of_iterations, image.shape, image_data)

for i in range(0, no_of_clusters):
    extract_individual_clusters(i, segmented_image_data, labels, image.shape)