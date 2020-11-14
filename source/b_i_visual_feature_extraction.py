from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.util import img_as_float
from skimage.filters import gabor_kernel

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
    
def gabor_feature(image):
    '''
    image: the 2D image array
    
    Extract 40 Gabor features, 8 different direction and 5 different frequency
    '''
    results = []
    kernel_params = []
    # 8 direction
    for theta in range(8):
        theta = theta / 4. * np.pi
        # 5 frequency
        for frequency in range(1,10,2):
            frequency = frequency * 0.1
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            results.append((kernel, power(image, kernel)))
    gabor_plot(kernel_params, results, image)
    return results

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    # Using the mod of real part and imag part of image after gabor filter
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

def gabor_plot(kernel_params, results, image):
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
            kernel = results[i//2 + 5*j][0]
            ax.imshow(np.real(kernel))
            ax.set_title(kernel_params[i//2 + 5*j])
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot Gabor responses with the contrast normalized for each filter
            ax = axes[i+1][j]
            power = results[i//2 + 5*j][1]
            ax.imshow(power)
            ax.axis('off')
    plt.show()

# creates dataframe of features given function and image path
def create_feature_df(feature_function, image_path):
    feature_data = []
    for i in os.listdir(image_path):
        image_file = image_path + i
        image_file = cv2.imread(image_file, 0)
        image = img_as_float(image_file)[shrink]
        # makes dataframe row with filename and image features, and flattens each feature
        feature_data.append([i] + list(np.array(feature_function(image)).flatten()))

    df_feat = pd.DataFrame(feature_data)
    df_feat_classified = df_hog.merge(
        df_from_csv[['filename', 'covid(label)']], left_on=0, right_on="filename")
    return df_feat_classified

 
if __name__=="__main__":
    shrink = (slice(0, None, 3), slice(0, None, 3))
    brick = img_as_float(data.brick())[shrink]
    # a single test of Gabor Extraction
    gabor_feature(brick)
