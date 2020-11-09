from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt

def hog_feature(image):
    """
    example usage:
        feature_mapped_images = np.array([ hog_feature(image) for image in train_images ])
    source: 
        https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/?utm_source=blog&utm_medium=3-techniques-extract-features-from-image-data-machine-learning
    """
    # NOTE: while these arguments work (for grayscale),
    #       I don't have a good 'feel' as to what they 
    #       actually do
    return hog(
        image,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        multichannel=False
    )[1]


def visualize(original_image, hog_image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True) 

    ax1.imshow(original_image, cmap=plt.cm.gray) 
    ax1.set_title('Original Image') 

    # Rescale histogram for better display 
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) 

    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray) 
    ax2.set_title('Histogram of Oriented Gradients')

    plt.show()