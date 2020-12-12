### Programming Notes

##### Computer shortcuts

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

##### Commands

###### Linux

- delete a folder from linux including all files

- delete a folder from linux including all files
  `sudo rm -r folder_name` 

- Delete all files from  the current directory 
  `sudo rm ./*` 

- Clean up root disk in Linux | dev/sda1 disk full problem

  ``sudo apt-get install ncdu`
  `sudo ncdu /  (too see all files size in root dir)`
  `or`
  `ncdu` . (see files sizes in the current directory)

- Download youtube videos as mp3 youtube-dl 

​      `youtube-dl -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 <URL>`

- Create a new file

​      `cat filename`



###### Anaconda commands

- create a new venv
  `conda create -n myenv python=3.6`

- create  anaconda env file from terminal 
  `conda env export > conda.yaml`

- remove a venv from anaconda
  `conda env remove -n env_name`

- Remove any anaconda env: 

​       `conda env remove -n env_name`

