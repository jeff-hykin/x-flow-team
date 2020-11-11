import cv2
import os

def crop_resize(name, target):
    '''
    name : 'train' or 'test'
    target: the target length of the resize image
    '''
    image_path = name + '/'
    new_path = 'new_' + name + str(target) + '/'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for i in os.listdir(image_path):
        image_file = image_path + i
        image = cv2.imread(image_file)
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
        image_new = cv2.resize(crop, (target, target))
        # save best quality of image
        cv2.imwrite(new_path + i, image_new, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    

if __name__=="__main__":
    crop_resize('train', 448)
