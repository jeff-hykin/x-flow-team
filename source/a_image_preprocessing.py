import cv2
import os
from skimage.util import img_as_float
import numpy as np

from misc_tools import split_data, get_train_test, images_in, flatten, is_grayscale

def preprocess_images(images, new_image_size=100):
    """
    default method for preprocessing images
    - grayscale
    - crop
    """
    results = []
    for each_image in images:
        new_image = np.copy(each_image)
        # grayscale-ify
        if not is_grayscale(new_image):
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        new_image = crop_resize(new_image, new_image_size)
        results.append(new_image)
    
    return results

def crop_resize(image, new_image_size):
    # shape like (1024,1007,3)
    size = image.shape
    if size[0] == size[1]:
        crop = image
    elif size[0] > size[1]:
        center = size[0] // 2
        half = size[1] // 2
        # crop the center of image
        if is_grayscale(image):
            crop = image[center-half:center+(size[1]-half),:]
        else:
            crop = image[center-half:center+(size[1]-half),:,:]
    else:
        center = size[1] // 2
        half = size[0] // 2
        # crop the center of image
        if is_grayscale(image):
            crop = image[:, center-half:center+(size[0]-half)]
        else:
            crop = image[:, center-half:center+(size[0]-half), :]
    # resize the image
    return cv2.resize(crop, (new_image_size, new_image_size))

def get_cropped_and_resized_images(name, new_image_size):
    '''
    name : 'train' or 'test'
    new_image_size: integer, the length of the resize image
    '''
    new_images = []
    image_folder = name + '/'
    new_path = 'new_' + name + str(new_image_size) + '/'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    
        for each_image_name in os.listdir(image_folder):
            new_images.append(crop_resize(cv2.imread(image_folder + each_image_name), new_image_size))
            # save best quality of image
            cv2.imwrite(new_path + each_image_name, new_images[-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    else:
        # load all of them from a file
        for each_image_name in os.listdir(new_path):
            new_images.append(cv2.imread(new_path + each_image_name))
    
    return new_images

def only_keep_every_third_pixel(image):
    # my understanding is that this has a step size of 3
    # for both dimensions so only 1 out of three in each direction
    # will be kept
    shrink = (slice(0, None, 3), slice(0, None, 3))
    return img_as_float(image)[shrink]

if __name__=="__main__":
    get_cropped_and_resized_images('train', 100)
