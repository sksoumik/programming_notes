##                                            

## Computer shortcuts

- Select the whole line: if the cursor is at the beginning of the line, then `shift+down arrow` if the cursor is at the end of a line then, `shit+up arrow`
- Show line numbers in colab: `ctrl + M + L`
- Insert a row or column in excel: `ctrl + (+)`
-  Havit keyboard shortcuts: fn + so that we can work around it. I have no problem. But I can ensure that might be a good keyboard. 
- Go end of the page: `ctrl + fn + right arrow` 
- Go at the beginning of a list: `ctrl + fn + left arrow`
- Move among opened tabs: `ctrl + fn + up/down arrow` 
- Change workspace: `ctrl + alt + right/left arrow` 
- Selecting multiple place mouse cursor: press `alt + shift + D` and then use `up/down` arrow. 
- Vscode: select the same word in the whole file: point the cursor in any word that you want to select then press  `ctrl + D`
- Vscode: Open User Settings: `CTRL + ,`
- VSCode: Quick Open a File: `CTRL + P` 
- VScode: Tab Through Open Files: `CTRL + Tab`
- Vscode: Move File to Split Windows = `CTRL + \ `
- Vscode: Navigating Text/ move a line up or down: Hold `ALT` when moving `left and right` through text to move faster.  

## Terminal Commands

###### Linux

- delete a folder from linux including all files`sudo rm -r folder_name` 

- Delete all files from  the current directory `sudo rm ./*` 

- Clean up root disk in Linux | dev/sda1 disk full problem

  `sudo apt-get install ncdu`
  `sudo ncdu /  (too see all files size in root dir)`
  or
  `ncdu`   (see files sizes in the current directory)

- Download youtube videos as mp3 youtube-dl 

​          `youtube-dl -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 <URL>`

- Create a new file:  `cat filename`



###### Anaconda commands

- create a new venv
  `conda create -n myenv python=3.6`

- create  anaconda env file from terminal 
  `conda env export > conda.yaml`

- remove a venv from anaconda
  `conda env remove -n env_name`

- Remove any anaconda env:    `conda env remove -n env_name`

###### GCP commands

- Connect with the instance from terminal: 
  `gcloud compute config-ssh`
  `ssh soumik.us-west1-b.project_name`

- After creating VM Instance configure from the local PC:
  `gcloud config set project project_name`
  `gcloud compute config-ssh`

- Copy file from instance to GCP workspace: 
  `sudo gsutil cp gs://ceri_snow_unified/CERI_snow_10sec/csv/kfold.csv .`

- Copying files from GCP instance to local computer:
  `gsutil -m cp -r gs://project_location/ ${loca_pc_dest_path}`

  for example:
  `gsutil -m cp -r gs://ceri_snow_unified/CERI_snow_10sec/ C:/Users/Dell/code/ceri_test_windows`

- See how much space left on GCP:
  `df -h`

- Upload files from bucket to instance:
  ``gsutil -m cp -R gs://bucket_location_gs_path .`

- Download file from gcp bucket to an instance (. means current directory)
  `gsutil -m cp -r gs://bucket_location .` 

- Untar files

  `tar -xf filename.xz`

###### Git

- create a new branch without adding any content from the master branch (empty brach)

  `git checkout --orphan branchname` 

  `git rm -rf .`

## Instance segmentation:

- object localization and detect boundaries. 



## NLP Intro

- Concepts of Bag-of-Words (BoW) and TF-IDF come into play. Both BoW and TF-IDF are techniques that help us convert text sentences into **numeric vectors**. [Read](https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/)
- BERT tokenizer does the preprocessing by itself, so usually you don't benefit from standard preprocessing.



## SVM

Support Vector Machine (SVM) is a supervised machine learning algorithm that can be used for **both classification or regression** challenges. The model extracts the best possible hyper-plane / line that segregates the two classes.



## Random Forest Model

Random Forest models are a type of **ensemble** models, particularly **bagging** models. They are part of the tree-based model family.



## Microsoft -vs- Google

Microsoft is an enterprise software company driven by **license revenue**, while Google is a consumer Internet company driven by **advertising revenue**.



## Enterprise software vs Consumer software

Enterprise software is just another term for business software. This is software that is **sold to (or targeted at) companies, not to individuals.** So, all the software which you use on a general basis like Windows or **Google or Quora is consumer software.**

Enterprise software is sold to companies to solve their problems. This can cover a wide range of applications, from software to manage the employees like payroll, attendance, promotions etc. (HRM), interacting with customers like the one’s marketing, sales.



## Text classification

###### **Approaches to automatic text classification can be grouped into three categories:**

- Rule-based methods
- Machine learning (data-driven) based methods
- Hybrid methods

###### **neural network architectures, such as models based on** 

- recurrent neural networks (RNNs),
- Convolutional neural networks (CNNs), 
- attention, 
- Transformers, 
- Capsule Nets



## Batch Size

- **Batch Gradient Descent**. Batch size is set to the total number of examples in the training dataset.
- **Stochastic Gradient Descent**. Batch size is set to one.
- **Minibatch Gradient Descent**. Batch size is set to more than one and less than the total number of examples in the training dataset.



## Python’s built-in `sorted()` function

The built-in sorting algorithm of Python uses a special version of merge sort, called Timsort, which runs in O(n log n) on average and worst-case both. 



## Permutations vs Anagrams vs Palindromes                  

Check Permutation: Given two strings, write a method to decide if one is a permutation of the other.

I’m working through algorithm exercises with a group of people, and there was a lot of confusion about what permutation means, and how it differs from anagrams and palindromes.

So, to clarify:

A permutation is one of several possible variations, in which a set of things (like numbers, characters or items in an array) can be ordered or arranged. A permutation of characters does not have to have meaning.

Example: Given the string abcd, the permutations are abcd, abdc, acbd, acdb, adbc, adcb, bacd, badc, bcad, bcda, bdac, bdca, cabd, cadb, cbad, cbda, cdab, cdba, dabc, dacb, dbac, dbca, dcab and dcba

An anagram is a word, phrase, or name formed by rearranging the characters of a string. An anagram must have meaning, it can’t just be gibberish.

Example: These words are anagrams of carets: caters, caster, crates, reacts, recast, traces

A palindrome is a word, phrase, or sequence that reads the same backward as forward. A palindrome must have meaning, it can’t just be gibberish.

Example: Civic, level, madam, mom and noon are all palindromes.

All palindromes and anagrams are permutations, but not all permutations are either anagrams or palindromes.



## What to do when you can’t find the solution?

- Use a jupyter notebook to debug the code. 
- Search similar problems in GitHub, Kaggle, Medium, YouTube, StackOverflow
- Break-down the problem into smaller parts and understand what you really need to do. 



### Multi-class Text Classification

- For multi-class classification: loss-function: categorical cross entropy (For binary classification: binary cross entropy loss). 
- BERT: Take a pre-trained BERT model, add an untrained dense layer of neurons, train the layer for any downstream task, … 



## Backpropagation

The backward function contains the backpropagation algorithm, where the goal is to essentially minimize the loss with respect to our weights. In other words, the weights need to be updated in such a way that the loss decreases while the neural network is training (well, that is what we hope for). All this magic is possible with the gradient descent algorithm. 



## Activation function vs Loss function

An *Activation function* is a property of the neuron, a function of all the inputs from previous layers and its output, is the input for the next layer.

If we choose it to be linear, we know the entire network would be linear and would be able to distinguish only linear divisions of the space.

Thus we want it to be non-linear, the traditional choice of function (tanh / sigmoid) was rather arbitrary, as a way to introduce non-linearity.

One of the major advancements in deep learning, is using ReLu, that is easier to train and converges faster. but still - from a theoretical perspective, the only point of using it, is to introduce non-linearity.

On the other hand, a *Loss function*, is the goal of your whole network.

it encapsulate what your model is trying to achieve, and this concept is more general than just Neural models.

A *Loss function*, is what you want to minimize, your error.

Say you want to find the best line to fit a bunch of points:

*D*={(*x*1,*y*1),…,(*x**n*,*y**n*)}
$$
D={(x1,y1),…,(x**n,y**n)}
$$
Your model (linear regression) would look like this:

`y=mx+n`

And you can choose several ways to measure your error (loss), for example L1:

or maybe go wild, and optimize for their harmonic loss:



## t-SNE algorithm

(**t**-**SNE**) **t**-Distributed Stochastic Neighbor Embedding is a non-linear dimensionality reduction algorithm **used for** exploring high-dimensional data. It maps multi-dimensional data to two or more dimensions suitable for human observation.



## Cross-Validation data

We should not use augmented data in cross validation dataset. 



## Concurrency and parallelism

**Concurrency** and **parallelism** both relate to "different things happening more or less at the same time. 

https://fastapi.tiangolo.com/async/#in-a-hurry



## Pickle vs JSON for serialization 

https://docs.python.org/3/library/pickle.html#comparison-with-json



## Hyperparameter optimization techniques

- Grid Search
- Bayesian Optimization. 
- Random Search 



## Normalization in ML

Normalizing helps keep the network weights near zero which in turn makes backpropagation more stable. Without normalization, networks will tend to fail to learn.



## Why do call `scheduler.step()` in pytorch?

If you don’t call it, the learning rate won’t be changed and stays at the initial value. 



## Momentum and Learning rate dealing 

If the LR is low, then momentum should be high and vice versa. The basic idea of momentum in ML is to increase the speed of training. 



## YOLO

You only look once (YOLO) is SOTA real-time object detection system. 



## Object Recognition

[Ref link](https://machinelearningmastery.com/object-recognition-with-deep-learning/)

Object classification + Object localization (bbox) = Object detection 

Object classification + Object localization + Object detection = Object Recognition (Object detection) 



## CRUD

In computer programming, **create, read, update, and delete** (CRUD) are the four basic functions of persistent storage. 



## SQL vs NoSQL 

**SQL:** 

- Relational database. 
- Organize data into one or more tables. 
- Each table has rows and columns.
- An unique identifier is added in each row. 

Relational database management systems: mysql, postgresql, mariadb, oracle etc. 

**NoSQL**

- Non-relational. 
- Organize data in a key:value pair. 
- Mainly documents in JSON/XML format. 

NoSQL management systems: MongoDB, firebase, apache cassandra etc. 

**Foreign key** is just the primary key of another table. So that we can make a relationship between two tables.