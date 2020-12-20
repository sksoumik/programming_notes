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

#### Linux Terminal

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
- Zip a folder with a name: `zip -r file_name.zip targer_folder_name/`
- 



#### Anaconda commands

- create a new venv
  `conda create -n myenv python=3.6`

- create  anaconda env file from terminal 
  `conda env export > conda.yaml`

- remove a venv from anaconda
  `conda env remove -n env_name`

- Remove any anaconda env:    `conda env remove -n env_name`



#### GCP commands

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

- See how much space left on GCP: `df -h`
  
- Upload files from bucket to instance: `gsutil -m cp -R gs://bucket_location_gs_path .`
  
- Download file from gcp bucket to an instance (. means current directory)
  `gsutil -m cp -r gs://bucket_location .` 

- Untar files: `tar -xf filename.xz`

- Transfer (copy) file from GCP instance to Cloud Storage Bucket: (run from the instance workplace) 

  `gsutil cp filename gs://bucket_name`

- Transfer/Copy file/folder from Cloud Storage Bucket to the local computer: (Open terminal in your local computer and navigate to the folder where you need the file to be copied and run this command):

   `gsutil -m cp -r gs://bucket_name/folder .` 

- 

  



#### Git

- create a new branch without adding any content from the master branch (empty branch)

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

## SQL

##### JOIN statement

A SQL Join statement is used to combine data or rows from two or more tables based on a common field between them. Different types of Joins  are:

- INNER JOIN
- LEFT JOIN
- RIGHT JOIN
- FULL JOIN

##### Aggregate Functions

An aggregate function performs a calculation on a set of values, and returns a single value. Except for `COUNT(*)`, aggregate functions ignore null values. Aggregate functions are often used with the `GROUP BY` clause of the SELECT statement.

Various aggregate functions are: 

```sql
1) Count()
2) Sum() 
3) Avg()
4) Min()
5) Max()
```

## Target Value Types

Categorical variables can be:

1. Nominal
2. Ordinal 
3. Cyclical 
4. Binary 

Nominal variables are variables that have two or more categories which do not have any kind of order associated with them. For example, if gender is classified into two groups, i.e. male and female, it can be considered as a nominal variable.Ordinal variables, on the other hand, have “levels” or categories with a particular order associated with them. For example, an ordinal categorical variable can be a feature with three different levels: low, medium and high. Order is important.As far as definitions are concerned, we can also categorize categorical variables as binary, i.e., a categorical variable with only two categories. Some even talk about a type called “cyclic” for categorical variables. Cyclic variables are present in “cycles” for example, days in a week: Sunday, Monday, Tuesday, Wednesday, Thursday, Friday and Saturday. After Saturday, we have Sunday again. This is a cycle. Another example would be hours in a day if we consider them to be categories.



## Confusion Matrix

Let's say, we have a dataset which contains cancer patient data (Chest X-ray image), and we have built a machine learning model to predict if a patient has cancer or not. 

**True positive (TP):** Given an image, if your model predicts the patient has cancer, and the actual target for that patient has also cancer, it is considered a true positive. Means the prediction is True. 

**True negative (TN):** Given an image, if your model predicts that the patient does not have cancer and the actual target also says that patient doesn't have cancer it is considered a true negative. Means the prediction is True. 

**False positive (FP):** Given an image, if your model predicts that the patient has cancer but the the actual target for that image says that the patient doesn't have cancer, it a false positive. Means the model prediction is False. 

**False negative (FN):** Given an image, if your model predicts that the patient doesn't have cancer but the actual target for that image says that the patient has cancer, it is a false negative. This prediction is also false.



## When not to use accuracy as Metric

If the number of samples in one class outnumber the number of samples in another class by a lot. In these kinds of cases, it is not advisable to use accuracy as an evaluation metric as it is not representative of the data. So, you might get high accuracy, but your model will probably not perform that well when it comes to real-world samples, and you won’t be able to explain to your managers why. In these cases, it’s better to look at other metrics such as precision.



## Common Evaluation Metrics in ML

If we talk about **classification problems**, the most common metrics used are:

- Accuracy

- Precision (P)
- Recall (R)
- F1 score (F1)
- Area under the ROC (Receiver Operating Characteristic) curve or simply AUC (AUC)
- Log loss- Precision at k (P@k)
- Average precision at k (AP@k)
- Mean average precision at k (MAP@k)

When it comes to **regression**, the most commonly used evaluation metrics are:

- Mean absolute error (MAE)
- Mean squared error (MSE)
- Root mean squared error (RMSE)
- Root mean squared logarithmic error (RMSLE)
- Mean percentage error (MPE)
- Mean absolute percentage error (MAPE)- R2

## Autoencoder

An **autoencoder** is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an **autoencoder** is to learn a representation (encoding) for a set of data, typically for dimensionality reduction. 

[See](https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder) visual representation and code in pytorch. A great [notebook](https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html) from cstoronto. 

##### Difference between AutoEncoder(AE) and Variational AutoEncoder(VAE):

The key difference between and autoencoder and variational autoencoder is autoencoders learn a “compressed representation” of input (*could be image,text sequence etc.)* automatically by first compressing the input (*encoder*) and decompressing it back (*decoder*) to match the original input. The learning is aided by using distance  function that quantifies the information loss that occurs from the lossy compression. So learning in an autoencoder is a form of unsupervised  learning (*or self-supervised as some refer to it*) - there is no labeled data.

Instead of just learning a function representing the data ( *a compressed representation*) like autoencoders, variational autoencoders learn the parameters of a  probability distribution representing the data. Since it learns to model the data, we can sample from the distribution and generate new input  data samples. So it is a generative model like, for instance, GANs

So, VAE are generative autoencoders, meaning they can generate new instances  that look similar to original dataset used for training. VAE learns **probability distribution** of the data  whereas autoencoders learns a function to map each input to a number and decoder learns the reverse mapping.

