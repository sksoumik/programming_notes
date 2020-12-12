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

​      `youtube-dl -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 <URL>`

- Create a new file:  `cat filename`



###### Anaconda commands

- create a new venv
  `conda create -n myenv python=3.6`

- create  anaconda env file from terminal 
  `conda env export > conda.yaml`

- remove a venv from anaconda
  `conda env remove -n env_name`

- Remove any anaconda env: 

​       `conda env remove -n env_name`

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

