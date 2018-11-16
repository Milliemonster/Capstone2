import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from os import listdir
from skimage.transform import resize
from sklearn import preprocessing

'''Opens images from selected folder and displays them. Photos can be selected
by number for copy into a new folder'''

def image_read(folder, max_images=10):

    labels = []
    full_images = []

    path = '/Users/millie/Documents/Galvanize/Capstone_2/images/'+folder

    if max_images == False:
        directory = listdir(path)
    else:
        directory = listdir(path)[:max_images]


    for item in directory:
        if item == '.DS_Store':
            pass
        else:
            img = skimage.io.imread(path+'/'+item)

        full_images.append(img)
        labels.append(folder)

    return np.array(full_images),  labels

def select_to_save(folder, components):

    keep_indexes = []
    start = 0

    for j in range (int(len(components)/10)):

        slice = components[start:start+10]
        fig = plt.figure(figsize = (20,5))
        for i in range(0, 10):
            ax = fig.add_subplot(2,5,i+1)

            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlabel(i)
            ax.imshow(slice[i], aspect = 'equal')

        plt.show(block=False)

        inds = input('input indices to keep:')
        for ind in inds:
            if ind.isdigit():
                keep = int(ind)+start
                keep_indexes.append(keep)

        start += 10

        for im in keep_indexes:
            skimage.io.imsave('/Users/millie/Documents/Galvanize/Capstone_2/images/select/'+folder+'/'+str(im)+'.png', components[im])

if __name__ == '__main__':
    #folder = input('Which bug?')
    components, y = image_read('ladybug', False)
    select_to_save('ladybug', components)
