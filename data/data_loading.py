import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

DATA_PATH = "../images/mirflickr/"
ANNOTATION_PATH = "../images/mirflickr25k_annotations_v080/"


# Takes in jpg file, returns file as opencv array
def readfile(file, dim=256):

    image = cv2.imread(file, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    H, W, C = image.shape

    # if image is large enough, scale it down
    if H >= dim and W >= dim:
        image = cv2.resize(image, (dim, dim))
        return image

    # if not, discard the image
    else:
        return np.empty((dim, dim, C))


def load_dataset(path=DATA_PATH):
    import glob
    import re
    import math
    from pathlib import Path
    file_pattern = re.compile(r'.*?(\d+).*?')

    def get_order(file):
        match = file_pattern.match(Path(file).name)
        if not match:
            return math.inf
        return int(match.groups()[0])

    data_files = sorted(glob.glob(path + "*.jpg"), key=get_order)

    # min H  = 42
    # min W = 118

    images_list = list()

    count = 0
    for file in data_files:
        image = readfile(file)
        if image is not None:
            images_list.append(image)

        # Report how many images have been processed
        count += 1
        if count % 1000 == 0:
            print("Finished processing image {}".format(count))

    images = np.array(images_list)
    return images

def load_dataset_annotation(annotation, path=DATA_PATH):
    import glob
    import re
    import math
    import os
    from pathlib import Path
    file_pattern = re.compile(r'.*?(\d+).*?')

    def get_order(file):
        match = file_pattern.match(Path(file).name)
        if not match:
            return math.inf
        return int(match.groups()[0])

    data_files = sorted(glob.glob(path + "*.jpg"), key=get_order)
    data_files = np.array(data_files)

    # Filter by annotations
    data_files = find_by_class(data_files, annotation)

    # min H  = 42
    # min W = 118

    images_list = list()

    count = 0
    for i, file in enumerate(data_files):
        image = readfile(file)
        if image is not None:
            images_list.append(image)

        # Report how many images have been processed
        count += 1
        if count % 100 == 0:
            print("Finished processing image {}".format(count))

    images = np.uint8(images_list)
    return images

def retrieve_data(category):

    try:
        train = np.load("processed_images_{}_train.npy".format(category))
        val = np.load("processed_images_{}_val.npy".format(category))
        test = np.load("processed_images_{}_test.npy".format(category))
        print("Retrieving stored images")

    except:
        print("Unable to find save")
        print("Processing images for {}".format(category))
        images = load_dataset_annotation(category)
        images = filter_bw(images)
        train, val, test = split_train_val_test(images)

        np.save("processed_images_{}_train.npy".format(category), train)
        np.save("processed_images_{}_val.npy".format(category), val)
        np.save("processed_images_{}_test.npy".format(category), test)
        print("Saved processed images")

    return train, val, test


def filter_bw(dataset):
    a = np.mean(np.abs(dataset[:, :, :, 1] - 128), axis=(1, 2))
    b = np.mean(np.abs(dataset[:, :, :, 2] - 128), axis=(1, 2))
    filtered_data = dataset[(a > 20) | (b > 20)]
    return filtered_data


def split_train_val_test(dataset):
#TODO: @Neethu implement
#     # split dataset into Training Set(80%), Validation Set(10%), and Testing
#     # Set(10%)
#     # Using random function to get the indeces and split the dataset
#     # Return 3 arrays of shape (x, 250, 250, 3)

    #idx = np.random.randint(0, dataset.shape[0], size=int(dataset.shape[0] * 0.8))
    idx = list(range(0, int(dataset.shape[0] * 0.8)))
    training = dataset[idx,:,:,:]

    #idx2 = np.random.randint(0, temp.shape[0], size=int(dataset.shape[0] * 0.1))
    idx2 = list(range(idx[-1], idx[-1] + int(dataset.shape[0] * 0.1)))
    validation = dataset[idx2,:,:,:]

    #idx3 = np.random.randint(0, temp.shape[0], size=int(dataset.shape[0] * 0.1))
    idx3 = list(range(idx2[-1], idx2[-1] + int(dataset.shape[0] * 0.1)))
    testing = dataset[idx3,:,:,:]

    return training, validation, testing

    
# def load_category_data(category):
#TODO: @Eric implement
def find_by_class(dataset, annotation):

    import os
    path = os.path.join(ANNOTATION_PATH, "{}.txt".format(annotation))

    # Retrieve all the indexes
    with open(path) as f:
        idxs = f.readlines()

    # must convert to int and subtract 1 for 0-based indexing
    idxs =[int(i.strip()) - 1 for i in idxs]

    return dataset[idxs]

def do_PCA(img):
    pca1 = PCA(0.99)
    n, d, c = img.shape
    max_dim = 0
    
    for i in range(0, c):
        a, b = pca1.fit_transform(img[:, :, i]).shape
        max_dim = max(max_dim, b)
    
    new_dim_image = np.zeros((n, d, c))
    
    pca2 = PCA(n_components = max_dim)
    
    for i in range(0, c):
        temp = pca2.fit_transform(img[:, :, i])
        approx = pca2.inverse_transform(temp)
        #print(approx.shape)
        approx = approx.reshape(n, d)
        new_dim_image[:, :, i] = approx
    
    return new_dim_image

# Main method for testing
if __name__ == "__main__":
    # usable dataset as a variable - shape = (m:24676, H:256, W:256, channel:3)
    train, val, test = retrieve_data("bird")

    image = test[4]

    print(image.dtype)

    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    cv2.imshow("", image)
    cv2.waitKey(0)
    #
    # filter_data = filter_bw(dataset)
    # training, validation, testing = split_train_val_test(dataset)
    #
    # print(dataset.shape)
    # print(filter_data.shape)
    # print(validation.shape)
    # print(training.shape)
    # print(testing.shape)
