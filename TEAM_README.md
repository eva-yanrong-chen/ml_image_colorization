# Setting up the Repository

Select where you want to put the repository and run the following command to clone it:

`git clone https://github.com/eva-yanrong-chen/ml_image_colorization.git`

# Download the dataset

Download the following files from http://press.liacs.nl/mirflickr/mirdownload.html:

* mirflickr25k.zip
* mirflickr25k_annotations_v080.zip

Unzip and place into images folder


# Download required packages

Highly recommend using python virtual environments: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/.

Install the requirements:

`pip install -r requirements.txt`

# Making Changes

Tips:
* Always make sure you are up to date by running `git pull`. 
* Be sure you are working on the correct branch. The easiest way to check is with a `git status`

To create a new feature branch:

1. Make sure you start from `master` (you can branch from other branches but not recommended)
2. Run `git checkout -b 'branch-name'`
3. Make changes 
4. Push your changes up to origin using `git push`
5. Make a pull request on https://github.com/eva-yanrong-chen/ml_image_colorization/pulls for your branch.
6. Let everyone else know
7. Merge the branch back in