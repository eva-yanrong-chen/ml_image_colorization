import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "../images/mirflickr/"


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
        return None


def load_dataset(path=DATA_PATH):
    import glob

    data_files = glob.glob(path + "*.jpg")

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


def retrieve_data():

    try:
        images = np.load("processed_images.npy")
        print("Retrieving stored images")

    except:
        print("Unable to find save")
        print("Processing images from scratch")
        images = load_dataset()
        np.save("processed_images.npy", images)
        print("Saved processed images")

    return images


def filter_bw(dataset):
    a = np.mean(np.abs(dataset[:, :, :, 1] - 128), axis=(1, 2))
    b = np.mean(np.abs(dataset[:, :, :, 2] - 128), axis=(1, 2))
    filtered_data = dataset[(a > 20) | (b > 20)]
    return filtered_data


# def split_train_val_test(dataset):
#TODO: @Neethu implement
#     # split dataset into Training Set(80%), Validation Set(10%), and Testing
#     # Set(10%)
#     # Using random function to get the indeces and split the dataset
#     # Return 3 arrays of shape (x, 250, 250, 3)

    
# def load_category_data(category):
#TODO: @Eric implement

# Main method for testing
if __name__ == "__main__":
    # usable dataset as a variable - shape = (m:24676, H:256, W:256, channel:3)
    dataset = retrieve_data()
    filter_data = filter_bw(dataset)
    
    print(dataset.shape)
    print(filter_data.shape)