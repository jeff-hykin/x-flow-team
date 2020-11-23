from scipy import ndimage as ndi
from skimage import data, exposure
from skimage.feature import hog
from skimage.filters import gabor_kernel
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np

from a_image_preprocessing import only_keep_every_third_pixel, get_preprocessed_train_test
from misc_tools import split_into_columns

def get_hog_train_test(hog_options={}, preprocess_options={}):
    train_features, train_labels, test_features = get_preprocessed_train_test(**preprocess_options)
    # hog + flatten
    transformation = lambda each: hog_feature(each, **hog_options).flatten()
    train_features['images'] = train_features['images'].transform(transformation)
    test_features['images']  = test_features['images'].transform(transformation)
    # give every image-feature its own column (a lot of columns)
    train_features = split_into_columns(train_features, "images")
    test_features  = split_into_columns(test_features,  "images")
    return train_features, train_labels, test_features

def get_gabor_train_test(gabor_options={}, preprocess_options={}):
    train_features, train_labels, test_features = get_preprocessed_train_test(**preprocess_options)
    
    # 
    # create a progress bar cause its really slow
    # 
    # (function doesn't touch the data, just passes the data through while keeping count)
    image_count = 0
    image_total = len(train_features['images']) + len(test_features['images'])
    def progress(arg):
        nonlocal image_count, image_total
        image_count += 1
        percent = (image_count/image_total)*100
        width = 30 # characters
        left = width * percent // 100
        right = width - left
        print(
            '\r[', '#' * int(left), ' ' * int(right), ']',
            f' {percent:.0f}%',
            sep='',
            end='',
            flush=True
        )
        # clean up when last step
        if image_count == image_total:
            print("")
        # just pass the data through without touching it
        return arg
    
    # gabor + flatten
    transformation = lambda each: progress(np.array(gabor_feature(each, **gabor_options)).flatten())
    train_features['images'] = train_features['images'].transform(transformation)
    test_features['images']  = test_features['images'].transform(transformation)
    # give every image-feature its own column (a lot of columns)
    print('train_features = ', len(train_features['images'][0]))
    print('splitting into columns')
    train_features = split_into_columns(train_features, "images")
    print('splitting first')
    test_features  = split_into_columns(test_features,  "images")
    return train_features, train_labels, test_features

def hog_feature(image, **options):
    """
    example usage:
        feature_mapped_images = np.array([ hog_feature(image) for image in train_images ])
    source: 
        https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/?utm_source=blog&utm_medium=3-techniques-extract-features-from-image-data-machine-learning
    """
    options = {} if options is None else options
    return hog(image, **{
        "orientations": 8,
        "pixels_per_cell": (16, 16),
        "cells_per_block": (1, 1),
        "visualize": True,
        "multichannel": False,
        **options
    })[1]

def visualize_hog(original_image, hog_image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True) 

    ax1.imshow(original_image, cmap=plt.cm.gray) 
    ax1.set_title('Original Image') 

    # Rescale histogram for better display 
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) 

    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray) 
    ax2.set_title('Histogram of Oriented Gradients')

    plt.show()

def gabor_feature(image, include_kernel=False):
    '''
    image: the 2D image array

    Extract 40 Gabor features, 8 different direction and 5 different frequency
    '''
    results = []
    kernels = []
    kernel_params = []
    # 8 direction
    for theta in range(8):
        theta = theta / 4. * np.pi
        # 5 frequency
        for frequency in range(1, 10, 2):
            frequency = frequency * 0.1
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            results.append(power(image, kernel))
            kernels.append(kernel)
    
    if include_kernel:
        return results, kernels, kernel_params
    else:
        return results        
    
def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    # Using the mod of real part and imag part of image after gabor filter
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

def gabor_plot(kernel_params, kernels, results, image):
    '''
    kernel_params: the title of kernel
    results: (kernel, power) the Gabor kernel and the image feature
    image: original 2D image array
    '''
    fig, axes = plt.subplots(nrows=11, ncols=8, figsize=(55, 40))
    plt.gray()
    fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)
    # Plot original image
    ax = axes[0][0] 
    ax.imshow(image)
    ax.set_title('original image', fontsize=8)
    for i in range(8):
        axes[0][i].axis('off')

    for i in range(1, 10, 2):
        for j in range(8):
            # Plot Gabor kernel
            ax = axes[i][j]
            # shape of results and kernel_params are both (40,)
            kernel = kernels[i//2 + 5*j]
            ax.imshow(np.real(kernel))
            ax.set_title(kernel_params[i//2 + 5*j])
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot Gabor responses with the contrast normalized for each filter
            ax = axes[i+1][j]
            power = results[i//2 + 5*j]
            ax.imshow(power)
            ax.axis('off')
    plt.savefig("../graphs/gabor_features.png")
 
if __name__=="__main__":
    brick = only_keep_every_third_pixel(data.brick())
    # a single test of Gabor Extraction
    results, kernels, kernel_params = gabor_feature(brick, include_kernel=True)
    gabor_plot(kernel_params, kernels, results, brick)
