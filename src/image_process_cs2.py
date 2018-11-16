import numpy as np
import skimage.io
from os import listdir
from skimage.transform import resize


def data_preprocess(folders, dimensions, flat=False, path = '/Users/millie/Documents/Galvanize/Capstone_2/images/select/'):

    images = []
    components = []
    labels = []

    for folder in folders:
        path += folder
        directory = listdir(path) #[:max_images]
        #print(directory)

        for item in directory:
            if item == '.DS_Store':
                pass
            else:
                img = skimage.io.imread(path+'/'+item)
                print (item)

            img_resized = resize(img, dimensions, mode='constant')
            flatten = img_resized.reshape(img_resized.size)

            images.append(img_resized)
            labels.append(folder)
            components.append(flatten)

    if flat:
        return np.array(components), labels
    else:
        return np.array(images), labels

if __name__ == '__main__':
    components, y = data_preprocess(['japanese_beetle', 'cucumber_beetle', 'ladybug'], (100,100,3), True)
