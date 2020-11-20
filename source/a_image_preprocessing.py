import cv2
import os
from skimage.util import img_as_float

def crop_resize(image, new_image_size):
    # shape like (1024,1007,3)
    size = image.shape
    if size[0] == size[1]:
        crop = image
    elif size[0] > size[1]:
        center = size[0] // 2
        half = size[1] // 2
        # crop the center of image
        crop = image[center-half:center+(size[1]-half),:,:]
    else:
        center = size[1] // 2
        half = size[0] // 2
        # crop the center of image
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
