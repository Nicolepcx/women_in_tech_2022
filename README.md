# Women in tech conference 2022  


This repo contains the slides and the accompanying code for the 
technical workshop about the attention mechanism @ [women in tech
global conference 2022](https://www.womentech.net/women-tech-conference). 


## Downloading the repo
This repo uses Git LFS find [here](https://www.atlassian.com/git/tutorials/git-lfs) more info how to use and install it,
you can clone a Git LFS repository as normal using git clone.



## Content

- Jupyter Notebook explaining the basics of the attention mechanism with NumPy
- Jupyter Notebook for An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale ([paper](https://arxiv.org/abs/2010.11929))
- Jupyter Notebooks and custom library for Unsupervised Anomaly Detection in Energy Time Series Data using Variational Recurrent Autoencoders with Attention ([paper](https://ieeexplore.ieee.org/document/8614232))



## Getting started

The easiest way to use the custom library and the Jupyter Notebooks
of this repo is to install pipenv as this repo contains a pipfile
with all installed packages. If you haven't installed pipenv on your computer yet, 
please follow the instructions [here](https://pipenv.pypa.io/en/latest/).


## Using the Jupyter Notebooks

To use the notebooks, you can use your IDE's terminal or the terminal of your choice on 
your computer. Just navigate to the repos folder, like for instance with the following 
terminal command: ```cd ~/Documents/02_code/women_in_tech_2022 ``` and use then use this command: 
```pipenv shell``` to start the virtual environment and use ```jupyter notebooks``` to 
start the server for the Jupyter Notebooks server. 

### Running the notebooks

The notebooks for the anomaly detection at the vison transformer will run rather longer on your computer.
To speed things up, you might want to use [Google's Colab](https://colab.research.google.com/). 
Here you can simply upload the notebooks, but be aware that you will have to alter the paths
for the notebooks of the anomaly detection notebook and also have to create some folders on your Google drive. 


## Using pipenv with PyCharm

The notebooks mostly use custom functions which are located in the folder 
```library```. To alter the code it is useful to use an IDE like
[PyCharm](https://www.jetbrains.com/pycharm/) for it. To use the pipenv setup
with PyCharm follow [this steps](https://www.jetbrains.com/help/pycharm/pipenv.html).




