# COMPUTER SCIENCE NOTES

Maintained by [**Sadman Kabir Soumik**](https://www.linkedin.com/in/sksoumik/)

---

# ML Basics

Read [here](https://developers.google.com/machine-learning/glossary)

# Difference between constant time vs linear time

In time-complexity,

**Constant time** O(1) means, the algorithm doesn't depend on the size of the input. If the size of the data container(e.g arrays) grow/increase, the execution time for some operation will remain same.

**Linear time** O(n) means, the algorithm depends on the size of the input. If the size of the data container (e.g. arrays) increases, the execution time for some operation will also increase.

# Time complexity

`n`: input size.

| Name        | Running time                                                 | example algorithms                                    |
| ----------- | ------------------------------------------------------------ | ----------------------------------------------------- |
| constant    | O(1)                                                         | Finding the median value in a sorted array of numbers |
| logarithmic | O(log n) \|   every time n increases by an amount k, the time or space increases by k/2. | Binary search                                         |
| Linear      | O(n)                                                         | Find duplicate elements in array with hash map        |
| Loglinear   | O(n log n) \|   implies that O(log n) operations will occur n times. | Merge Sort, Heap Sort, Quick Sort                     |
| Quadratic   | O(n^2)                                                       | Bubble sort, Insertion sort                           |
| Cubic       | O(n^3)                                                       |                                                       |
| Exponential | O(2^n)                                                       | Find all subsets                                      |
| Factorial   | O(n!)                                                        | Find all permutations of a given set/string           |

See the time and space complexities [chat](https://en.wikipedia.org/wiki/Sorting_algorithm#Stability)

# Linear vs Non-linear data structures

|            Key            |                    Linear Data Structures                    |                  Non-linear Data Structures                  |      |
| :-----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | ---- |
| Data Element Arrangement  | In linear data structure, data elements are **sequentially connected** and each element is traversable through a single run. | In non-linear data structure, data elements are hierarchically connected and are present at various levels. |      |
|          Levels           | In linear data structure, all data elements are present at a single level. | In non-linear data structure, data elements are present at multiple levels. |      |
| Implementation complexity |       Linear data structures are easier to implement.        | Non-linear data structures are difficult to understand and implement as compared to linear data structures. |      |
|         Traversal         | Linear data structures can be traversed completely in a single run. | Non-linear data structures are not easy to traverse and needs multiple runs to be traversed completely. |      |
|    Memory utilization     | Linear data structures are not very memory friendly and are not utilizing memory efficiently. |   Non-linear data structures uses memory very efficiently.   |      |
|      Time Complexity      | Time complexity of linear data structure often increases with increase in size. | Time complexity of non-linear data structure often remain with increase in size. |      |
|         Examples          |                  Array, List, Queue, Stack.                  |                      Graph, Map, Tree.                       |      |

# Difference between tree and graph

In tree, there is no cycles. In Graphs, cycles may form. 

# DFS, BFS

| DFS                                                          | BFS                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Stack                                                        | Queue                                                        |
| LIFO                                                         | FIFO                                                         |
| Stacking Plates                                              | Queue in front of a elevator                                 |
| DFS is more suitable when there are solutions away from source. | BFS is more suitable for searching vertices which are closer to the given source. |
| when we want to know the all possible results                | when we want to find the shortest path (simple graph). we usually use bfs,it can guarantee the 'shortest'. |



# Polymorphism vs Overriding vs Overloading

Polymorphism means more than one form, same object performing different operations according to the requirement.

Polymorphism can be achieved by using two ways, those are

1. Method overriding
2. Method overloading

*Method overloading* means writing two or more methods **in the same class** by using same method name, but the passing parameters are different.

*Method overriding* ability of any object-oriented programming language that allows a subclass or child class to provide a specific implementation of a method that is already provided by one of its super-classes or parent classes. When a method in a subclass has the same name, same parameters or signature and same return type(or sub-type) as a method in its super-class, then the method in the subclass is said to **override** the method in the super-class.

# Computer shortcuts

#### Chrome

- Show all bookmarks to search

  ```
  ctrl + shift + O
  ```

  

- Move between tabs: 

  ```
  ctrl + fn + up/down arrow
  ```

  

- Go end of the page: 

  ```
  ctrl + fn + right arrow
  ```

  

- Go at the beginning of a list: 

  ```
  ctrl + fn + left arrow
  ```

  

#### vscode

- Vscode: select the same word in the whole file: point the cursor in any word that you want to select then press `ctrl + D`
- Vscode: Open User Settings: `CTRL + ,`
- VSCode: Quick Open a File: `CTRL + P`
- VScode: Tab Through Open Files: `CTRL + Tab`
- Vscode: Move File to Split Windows = `CTRL + \ `
- Vscode: Navigating Text/ move a line up or down: Hold `ALT` when moving `left and right` through text to move faster.

#### Linux

- Move among opened tabs: `ctrl + fn + up/down arrow`
- Change workspace: `ctrl + alt + right/left arrow`
- Selecting multiple place mouse cursor: press `alt + shift + D` and then use `up/down` arrow.

#### Linux Terminal

- add a custom command in linux terminal: 

  ```bash
  alias custom_command='original_command'
  ```

- delete a folder from linux including all files 

  ```bash
  sudo rm -r folder_name
  ```

  

- get a notification with a voice after a process gets finished:

  ```bash
  some-command; spd-say "Any Voice Message"
  ```

  

- Delete all files from the current directory 

  ```bash
  sudo rm ./*
  ```

  

- Clean up root disk in Linux | dev/sda1 disk full problem

  ```bash
  sudo apt-get install ncdu
  sudo ncdu /  # too see all files size in root dir
  or
  ncdu       # (see files sizes in the current directory)
  ```

  

- Download youtube videos as mp3 youtube-dl 

  ```bash
  youtube-dl -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 <URL>
  ```

  

- Create a new file:

  ```bash
  cat filename
  ```

  

- find absolute location of a file: 

  ```bash
  readlink -f file.txt
  ```

  

- Zip a folder with a name: 

  ```bash
  zip -r file_name.zip targer_folder_name/
  ```

  

- Open a folder in file explorer using Linux Terminal:

  ```bash
  xdg-open folder
  ```

  

- copy or move files from subdirectories that ends with a common extension: `mv **/*.csv target_dir`

- Install VS Code from terminal: [read here](https://linuxize.com/post/how-to-install-visual-studio-code-on-ubuntu-18-04/)

- Get the size of a file (human-readable): `du -sh <file-name>`

- Search for a file in the current directory: `find *<file-name-portion>*`

- rename a folder: `vm old_name new_name`

- `Ctrl + L` : equivalent to `clear`

- `Ctrl + U`: This shortcut erases everything from the current cursor position to the beginning of the line.

- `Ctrl + A`: Move the cursor to the beginning of the line.

- `Ctrl + E`: This shortcut does the opposite of `Ctrl + A`. It moves the cursor to the end of the line.

- Display first 3 lines of a file in the terminal: `head -3 filename`

- ```
  sudo apt update        # Fetches the list of available updates
  sudo apt upgrade       # Installs some updates; does not remove packages
  sudo apt full-upgrade  # Installs updates; may also remove some packages, if needed
  sudo apt autoremove    # Removes any old packages that are no longer needed
  ```

- see the laptop hardware information: `sudo lshw`

- find cpu configuration: `lscpu`

- 

#### Anaconda commands

- create a new venv
  `conda create -n myenv python=3.6`
- create anaconda env file from terminal
  `conda env export > conda.yaml`
- Creating new anaconda environment using a yaml file: `conda env create --file dependency.yaml`
- remove a venv from anaconda
  `conda env remove -n env_name`
- Remove any anaconda env: `conda env remove -n env_name`
- Remove any package from an environment: activate the target env and `conda remove package_name`

#### GCP commands

- Create a GCP VM Instance: [link](https://github.com/cs231n/gcloud/)
  
- Connect with the instance from terminal:
  
  ```
  gcloud compute config-ssh
  ssh soumik.us-west1-b.project_name
  ```
  
  
  
- After creating VM Instance configure from the local PC:
  
  ```
  gcloud config set project project_name
  gcloud compute config-ssh
  ```
  
  
  
- Copy file from instance to GCP workspace:
  
  ```
  sudo gsutil cp gs://ceri_snow_unified/CERI_snow_10sec/csv/kfold.csv .
  ```
  
  
  
- Copying files from GCP instance to local computer:
  
  ```
  gsutil -m cp -r gs://project_location/ ${loca_pc_dest_path}
  ```

  for example:
  
  ```
  gsutil -m cp -r gs://ceri_snow_unified/CERI_snow_10sec/ C:/Users/Dell/code/ceri_test_windows
  ```
  
  
  
- See how much space left on GCP: 

  ```
  df -h
  ```

  

- Upload files from bucket to instance: 

  ```
  gsutil -m cp -R gs://bucket_location_gs_path .
  ```

  

- Download file from gcp bucket to an instance (. means current directory)
  
  ```
  gsutil -m cp -r gs://bucket_location .
  ```
  
  
  
- Untar files: `tar -xf filename.xz`

- Transfer (copy) file from GCP instance to Cloud Storage Bucket: (run from the instance workplace)

  `gsutil cp filename gs://bucket_name`

- Transfer/Copy file/folder from Cloud Storage Bucket to the local computer: (Open terminal in your local computer and navigate to the folder where you need the file to be copied and run this command):

  `gsutil -m cp -r gs://bucket_name/folder .`

- Clean up root disk in Linux | dev/sda1 disk full problem | see which files are taking what space in the disk:

  - install ncdu: `sudo apt-get install ncdu`
  - to see all file/folder size in the root directory of disk: `sudo ncdu /`
  - to see all file/folder size in the current directory of disk: `ncdu .`

#### Git

Create ssh key for github

```
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Could not open a connection to your authentication agent

```
eval `ssh-agent -s`
```





- See a git cheatsheet: [[here](/static/git-cheat-sheet-education.pdf)]

- github authentication error: use `gh` client. 

  ```bash
  sudo apt update
  sudo apt install gh
  gh auth login
  ```

- create a new branch without adding any content from the master branch (empty branch)

  `git checkout --orphan branchname`

  `git rm -rf .`

- revert back to a specific commit:

  `git reset --hard <commit_id>`

  `git push -f origin master`

- create a new branch and switch to it at the same time:

`git checkout -b <branch_name>`

- Add agent:

  ```shell
  eval `ssh-agent -s`
  ssh-add
  ```

- update a branch with the master:

  ```bash
  git checkout <branch_name>
  git merge main
  ```

- update the current branch with the recent changes:

  ```
  git pull origin <remote branch name>
  ```

- update the current branch with the master (if changes later after pulling the master code)

  ```
  git rebase master
  ```

  

# Microsoft -vs- Google

Microsoft is an enterprise software company driven by **license revenue**, while Google is a consumer Internet company driven by **advertising revenue**.

# Enterprise software vs Consumer software

Enterprise software is just another term for business software. This is software that is **sold to (or targeted at) companies, not to individuals.** So, all the software which you use on a general basis like Windows or **Google or Quora is consumer software.**

Enterprise software is sold to companies to solve their problems. This can cover a wide range of applications, from software to manage the employees like payroll, attendance, promotions etc. (HRM), interacting with customers like the one’s marketing, sales.

# Permutations vs Anagrams vs Palindromes

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



# Concurrency and parallelism

**Concurrency** and **parallelism** both relate to "different things happening more or less at the same time.

https://fastapi.tiangolo.com/async/#in-a-hurry

# Daemon in Linux

A **daemon** (pronounced DEE-muhn) is a program that runs continuously and exists for the purpose of handling periodic service requests that a computer system expects to receive. The **daemon** program forwards the requests to other programs (or processes) as appropriate. For example, the **Cron** daemon is a built-in **Linux** utility that runs processes on your system at a scheduled time. We can configure a **cron** job to schedule scripts or other commands to run automatically.

# Pickle vs JSON for serialization

https://docs.python.org/3/library/pickle.html#comparison-with-json



# CRUD

In computer programming, **create, read, update, and delete** (CRUD) are the four basic functions of persistent storage.

# SQL vs NoSQL

**SQL:**

- Relational database.
- Organize data into one or more tables.
- Each table has rows and columns.
- An unique identifier is added in each row.
- B+ tree is the main data structure

Relational database management systems: mysql, postgresql, mariadb, oracle etc.

**NoSQL**

- Non-relational.
- Organize data in a key:value pair.
- Mainly documents in JSON/XML format.
- LSM-tree (Long structured merge -tree) is the main data structure.

NoSQL management systems: MongoDB, firebase, apache cassandra etc.

**Foreign key** is just the primary key of another table. So that we can make a relationship between two tables.

**LSM-tree**: read [here](http://www.shafaetsplanet.com/?p=3796)

# SQL Basics

##### Basic statements

- SQL statements fall into two different categories: Data Definition Language (DDL) statements and Data Manipulation Language (DML) statements.

  ![](static/dbms-sql-command.png)

Read more details from [here](https://www.javatpoint.com/dbms-sql-command)

**Query statement (retrieve data): ** SELECT

**DDL (data definition language)**: Create, Drop, Alter, Truncate

**DML (Data manipulation language) statement:** INSERT, UPDATE, DELETE

##### Select Rule | Query data from a table

```sql
SELECT column_name
FROM Table_name
WHERE Conditions;
```

##### Insert Rule | Inser new data in a table

```sql
INSERT INTO table_name (column1, column2, ... )
VALUES (value1, value2, ... );
```

Example, insert one row

```sql
INSERT INTO Instructor(ins_id, lastname, firstname, city, country)
VALUES(4, 'Saha', 'Sandip', 'Edmonton', 'CA');
```

Example, insert multiple row

```sql
INSERT INTO Instructor(ins_id, lastname, firstname, city, country)
VALUES(5, 'Doe', 'John', 'Sydney', 'AU'), (6, 'Doe', 'Jane', 'Dhaka', 'BD');
```

##### Update Rule | Alter information in a table

```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition;
```

Example, update data

```sql
UPDATE Instructor
SET city='Toronto'
WHERE firstname="Sandip";
```

Example, update multiple columns

```sql
UPDATE Instructor
SET city='Dubai', country='AE'
WHERE ins_id=5;
```

##### Delete | Remove one or more rows from a table

```sql
DELETE FROM table_name
WHERE condition;
```

Example, delete one row from the table

```sql
DELETE FROM Instructor
WHERE ins_id = 6;
```

##### JOIN statement

A SQL Join statement is used to combine data or rows from two or more tables based on a common field between them. Different types of Joins are:

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

##### Few clauses that is used with the select statement

**COUNT:** retrieves the number of rows

**DISTINCT** is used to remove duplicate values from a result set.

**LIMIT**: restricting the number of rows retrieved from the database.

## Entity -relationship diagrams

Entity: Table

Attribute: Columns

##### Data type, CHAR vs VARCHAR

A CHAR field is a *fixed* length, and VARCHAR is a *variable* length field.

This means that the storage requirements are different - a CHAR always takes the same amount of space regardless of what you store, whereas the storage requirements for a VARCHAR vary depending on the specific string stored. 

CHAR fields are stored inside the register due to its size being known, this makes searching and indexing faster. 

##### Indexing in DB

Consider a "Book" of 1000 pages, divided by 10 Chapters, each section with 100 pages.

Now, imagine you want to find a particular Chapter that contains a word "**Alchemist**". Without an index page, you have to scan through the entire book/Chapters. i.e: 1000 pages.

This analogy is known as **"Full Table Scan"** in database world.

![book index](static/indexing.jpg)

But with an index page, you know where to go! And more, to lookup any particular Chapter that matters, you just need to look over the index page, again and again, every time. After finding the matching index you can efficiently jump to that chapter by skipping the rest.

But then, in addition to actual 1000 pages, you will need another ~10 pages to show the indices, so totally 1010 pages.

> Thus, the index is a separate section that stores values of indexed column + pointer to the indexed row in a sorted order for efficient look-ups.



##### Normalization

Normalization is a **database design technique** that reduces data redundancy and eliminates undesirable characteristics like Insertion, Update and Deletion Anomalies. Normalization rules divides larger tables into smaller tables and links them using relationships.

If a table is not properly normalized and have data redundancy then it will not only eat up extra memory space but will also make it difficult to handle and update the database, without facing data loss. Insertion, Updation and Deletion Anomalies are very frequent if database is not normalized.

###### 1NF

1. It should only have single(atomic) valued attributes/columns.
2. All the columns in a table should have unique names.

###### 2NF

1. Should be in 1NF.
2. Single column primary column. There should be no partial dependency. 

##### 3NF

1. satisfy 2NF.
2. 

5. 

# SQLite Database Creation: Flask

##

```sql
$ sqlite3 database.db
$ .tables
$ .exit
# define the database path code.
$ python
$ from app import db
$ db.create_all()
$ exit()
# open the database
$ sqlite3 database.db
$ .tables
$ select * from table_name
```



# CMOS

Stands for "Complementary Metal Oxide Semiconductor." It is a technology used to produce integrated circuits. **CMOS** circuits are found in several types of electronic components, including microprocessors, batteries, and digital camera image sensors.

# Profiling

In software engineering, profiling is a form of dynamic program analysis that measures, for example, the space or time complexity of a program, the usage of particular instructions, or the frequency and duration of function calls. Most commonly, profiling information serves to aid program optimization.

# Babel

**Babel** is a **transpiler** that converts our ultra-modern JavaScript syntax to browser-readable JavaScript, HTML, and CSS.

# HTML class vs ID

The **difference** between an **ID** and a **class** is that an **ID** is only used to identify **one single element** in our **HTML**. ... However, a **class** can be used to identify more than one **HTML** element.

# Vue.js commands

```bash
# check vue version
$ vue --version
# create the app from the current directory
$ vue create <app-name>
# run the app to browser
$ npm run serve

```

# Abstract class

An **abstract class** is a **class** that is declared **abstract** —it may or may not include **abstract** methods. **Abstract classes** cannot be instantiated, but they can be subclassed. Abstract classes are classes that contain one or more abstract methods. An abstract method is a method that is declared, but contains **no implementation**. Abstract classes **cannot be instantiated**, and require subclasses to provide implementations for the abstract methods.

Python on its own doesn't provide abstract classes. Yet, Python comes with a module which provides the infrastructure for defining Abstract Base Classes (ABCs). This module is called - for obvious reasons - **abc**.

The following Python code uses the abc module and defines an abstract base class:

```python
from abc import ABC, abstractmethod

class AbstractClassExample(ABC):

    def __init__(self, value):
        self.value = value
        super().__init__()

    @abstractmethod
    def do_something(self):
        pass
```

We will define now a subclass using the previously defined abstract class. You will notice that we haven't implemented the `do_something` method, even though we are required to implement it, because this method is **decorated** as an abstract method with the decorator "`abstractmethod`". We get an exception that Add42 can't be instantiated.

```python
class Add42(AbstractClassExample):
    pass

x = Add42(4)
```

Output:

```bash
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-4-2bcc42ab0b46> in <module>
      2     pass
      3
----> 4 x = Add42(4)

TypeError: Can't instantiate abstract class Add42 with abstract methods do_something
```

We will do it the correct way in the following example, in which we define two classes inheriting from our abstract class:

```python
class Add42(AbstractClassExample):

    def do_something(self):
        return self.value + 42

class Mul42(AbstractClassExample):

    def do_something(self):
        return self.value * 42

x = Add42(10)
y = Mul42(10)

print(x.do_something())
print(y.do_something())
```

```bash
52
420
```

Read [more...](https://www.python-course.eu/python3_abstract_classes.php)

# Python collections module

Read [here](https://docs.python.org/3/library/collections.html#module-collections)

### 

# Static method vs Instance method

##### Static Method

Static methods are methods that can be called without creating an object of the class. They can be just called by referring the class name. For example: if we have a class:

```javascript
class ClassName {
  static getPosts() {}
}
```

Then, I don't need to create an object like `obj = new ClassName()` something like this. I can directly call `ClassName.getPosts()`.

Pythonic way to create a static method: Most common form is to put a `@staticmethod` decorator on top of the function.

```python
class MyClass:
    @staticmethod
    def hello():
        print('static method called')
```

```bash
# way to call a static method
>>> MyClass.hello()
```

Output:

```
static method called
```

##### Instance Method

Pythonic way to create an instance method: Instance method must contain a self parameter.

```python
class MyClass:
    def instance_method(self):
        return 'instance method called', self
```

#### When do we want to use a `static` method?

1. Static methods are used when we don't want subclasses of a class change/override a specific implementation of a method.
2. A particular piece of code is to be shared by all the instance methods.
3. If you are writing utility classes and they are not supposed to be changed.

# What is the purpose of self keyword in Python?

`self` represents the instance of the class. By using the “self” keyword we can access the attributes and methods of the class in python.

Let's say you have a class `ClassA` which contains a method `methodA` defined as:

```python
def methodA(self, arg1, arg2):
    # do something
```

and `ObjectA` is an instance of this class.

Now when `ObjectA.methodA(arg1, arg2)` is called, python internally converts it for you as:

```python
ClassA.methodA(ObjectA, arg1, arg2)
```

The `self` variable refers to the object itself. The `self` parameter is a reference to the **current instance of the class**, and is used to access variables that belongs to the class. It does not have to be named `self` , you can call it whatever you like, but it has to be the first parameter of any function in the class.

# None keyword in python

The `None` keyword is used to define a null value, or no value at all. `None` is not the same as 0, `False`, or an empty string. None is a data type of its own (NoneType) and only None can be `None`. 

# list pop()

Code:

```python
queue = [1, 2, 3]
queue.pop()  # delete the last item
print(queue)
```

Output:

```
[1, 2]
```

Code:

```python
queue = [1, 2, 3]
queue.pop(0)  # delete the first item
print(queue)
```

Output:

```
[2, 3]
```

# List comprehension in Python

code:

```python
words = ['data','science','machine','learning']

#for loop
a = []
for word in words:
   a.append(len(word))

#list comprehension
b = [len(word) for word in words]

print(f"a is {a}")
print(f"b is {b}"
```

output:

```bash
a is [4, 7, 7, 8]
b is [4, 7, 7, 8]
```

code:

```python
#for loop
a = []
for word in words:
   if len(word) > 5:
    	a.append(word)

#list comprehension
b = [word for word in words if len(word) > 5]

print(f"a is {a}")
print(f"b is {b}")
```

output:

```bash
a is ['science', 'machine', 'learning']
b is ['science', 'machine', 'learning']
```

code:

```python
#for loop
a = []
for word in words:
  for letter in word:
    if letter in ["a","e","i"]:
       a.append(letter)


# list comprehension
b = [letter for word in words for letter in word if letter in ["a","e","i"]]
```

Read [more](https://towardsdatascience.com/crystal-clear-explanation-of-python-list-comprehensions-ac4e652c7cfb)..

# Recurrent Neural Network

##### Use cases

1. sentiment analysis
2. text mining
3. and image captioning
4. time series problems such as predicting the prices of stocks in a month or quarter
5.



# List and Tuple difference

| List                                        | Table                                       |
| ------------------------------------------- | ------------------------------------------- |
| Lists are mutable, means they can be edited | Tuples are immutble, means can't be edited. |
| Lists are slower than tuples.               | Tuples are faster.                          |

# What is the difference between Python Arrays and lists?

Arrays and lists, in Python, have the same way of storing data. But, arrays can hold only a **single data type** elements whereas lists can hold **any data type** elements.

# Difference between HashTable and HashMap

| Hash MAP                                      | Hash Table                            |
| --------------------------------------------- | ------------------------------------- |
| Not synchronized. So, it's not thread-safe.   | Synchronized. Means it's thread-safe. |
| Allows one null key and multiple null values. | Doesn't allow any null key or value.  |

Python dictionaries are based on a well-tested and finely tuned **hash table** implementation that provides the performance characteristics you’d expect: **_O_(1)** time complexity for lookup, insert, update, and delete operations in the average case.

# Synchronized

`synchronized` means that in a multi threaded environment, an object having `synchronized` method(s)/block(s) does not let two threads to access the `synchronized` method(s)/block(s) of code at the same time. This means that one thread can't read while another thread updates it.

# How does the Google "Did you mean?" algorithm work?

Basically and according to Douglas Merrill former CTO of Google it is like this:

1. You write a ( misspelled ) word in Google.

2. You don't find what you wanted ( don't click on any results ).

3. You realize you misspelled the word so you rewrite the word in the search box.

4. You find what you want ( you click in the first links ).

This pattern multiplied millions of times, shows what are the most common misspells and what are the most "common" corrections.

This way Google can almost instantaneously, offer spell correction in every language.

Also this means if overnight everyone start to spell night as "nigth", Google would suggest that word instead. Douglas describe it as "statistical machine learning".

# Proxy Server

PS is an intermediate server between client and the Internet. Proxy servers offers the following basic functionalities:

- Firewall and network data filtering.
- Network connection sharing
- Data caching

##### Purpose of Proxy Servers:

- Monitoring and Filtering:
- Improving performance
- Translation
- Accessing services anonymously
- Security

Read more from [here](https://www.tutorialspoint.com/internet_technologies/proxy_servers.htm)

# SQL vs NoSQL: what’s the best option for you?

**1 . Data structure**

The first and primary factor in making the SQL vs. NoSQL decision is what your data looks like. If your data is primarily structured, a SQL database is likely the right choice. A SQL database is a great fit for transaction-oriented systems such as customer relationship management tools, accounting software, and e-commerce platforms. Each row in a SQL database is a distinct entity (e.g. a customer), and each column is an attribute that describes that entity (e.g. address, job title, item purchased, etc.). [Read more ...](https://www.thorntech.com/sql-vs-nosql/)

NoSQL examples:

1. Big data applications.
2. Rapidly growing application that needs scalability.
3. Social Media (e.g. Facebook)
4.

SQL (MongoDB, Redis, Cassandra) examples:

1. Transaction systems.
2. Banking systems.
3. Customer relationship systems.
   1. E-commerce.


# Difference between JPG and PNG

**PNG** stands for Portable Network Graphics, with so-called “lossless” compression.

JPEG or **JPG** stands for Joint Photographic Experts Group, with so-called “lossy” compression.

**JPEG** uses lossy compression algorithm and image may lost some of its data whereas **PNG** uses lossless compression algorithm and no image data loss is present in **PNG** format.



# Web Server

1. Apache HTTP Server
2. Gunicorn
3. Nginx (Engine-X) requires a JSON configuration
4. Unicorn

ASGI (Asynchronous Server Gateway Interface) server implementation

5. Uvicorn

# Metadata

Metadata is "data that provides information about other data". In other words, it is "data about data".

# Docker Basics

###### Containerization

Usually, in the software development process, code developed on one machine might not work perfectly fine on any other machine because of the dependencies. This problem was solved by the containerization concept. So basically, an application that is being developed and deployed is bundled and wrapped together with all its configuration files and dependencies. This bundle is called a container. Now when you wish to run the application on another system, the container is deployed which will give a bug-free environment as all the dependencies and libraries are wrapped together. Most famous containerization environments are Docker and Kubernetes.

###### What is Docker Compose? What can it be used for?

Docker Compose is a tool that lets you define multiple containers and their configurations via a YAML or JSON file. The most common use for Docker Compose is when your application has one or more dependencies, e.g., MySQL or Redis. Normally, during development, these dependencies are installed locally—a step that then needs re-doing when moving to a production setup. You can avoid these installation and configuration parts by using Docker Compose.

Once set up, you can bring all of these containers/dependencies up and running with a single `docker-compose up` command.

###### If you wish to use a base image and make modifications or personalize it, how do you do that?

You pull an image from docker hub onto your local system

It’s one simple command to pull an image from docker hub:

```
$ docker pull <image_name>
```

###### How do you create a docker container from an image?

Pull an image from docker repository with the above command and run it to create a container. Use the following command:

```
$ docker run -it -d <image_name>
```

`-d` means the container needs to start in the detached mode.

###### How do you list all the running containers?

The following command lists down all the running containers:

```
$ docker ps
```

###### Suppose you have 3 containers running and out of these, you wish to access one of them. How do you access a running container?

The following command lets us access a running container:

```
$ docker exec -it <container id> bash
```

###### **Can I use JSON instead of YAML for my compose file in Docker?**

You can use JSON instead of YAML for your compose file, to use JSON file with compose, specify the JSON filename to use, for eg:

```
$ docker-compose -f docker-compose.json up
```



# Class Method vs Static Method in Python

A **staticmethod** is a method that knows nothing about the class or instance it was called on. It just gets the arguments that were passed, no implicit first argument. We can use static method to create utility functions. It's a way of putting a function into a class (because it logically belongs there), while indicating that it does not require access to the class.

**With classmethods**, the class of the object instance is implicitly passed as the first argument instead of `self`.

To decide whether to use [@staticmethod](https://docs.python.org/3/library/functions.html?highlight=staticmethod#staticmethod) or [@classmethod](https://docs.python.org/3.5/library/functions.html?highlight=classmethod#classmethod) you have to look inside your method. **If your method accesses other variables/methods in your class then use @classmethod**. On the other hand, if your method does not touches any other parts of the class then use @staticmethod.

```python
class Apple:

    _counter = 0

    @staticmethod
    def about_apple():
        print('Apple is good for you.')

        # note you can still access other member of the class
        # but you have to use the class instance
        # which is not very nice, because you have repeat yourself
        #
        # For example:
        # @staticmethod
        #    print('Number of apples have been juiced: %s' % Apple._counter)
        #
        # @classmethod
        #    print('Number of apples have been juiced: %s' % cls._counter)
        #
        #    @classmethod is especially useful when you move your function to other class,
        #       you don't have to rename the class reference

    @classmethod
    def make_apple_juice(cls, number_of_apples):
        print('Make juice:')
        for i in range(number_of_apples):
            cls._juice_this(i)

    @classmethod
    def _juice_this(cls, apple):
        print('Juicing %d...' % apple)
        cls._counter += 1
```

# Database basics

###### Volatile vs Non-volatile

Volatile: Power-off -> Data lost

Non-Volatile: Power-off -> Still data remains.

###### Different ERD schemas:

1. Star schema
2. Constellation schema
3. Snowflake schema

###### Database vs Data warehouse

Database is a collection of related data that represents some elements of the real world whereas Data warehouse is an information system that stores historical and commutative data from single or multiple sources. Database is designed to record data whereas the Data warehouse is designed to analyze data.

DB: designed for record/store data.

DW: designed for analyzing data.



# How Django works

1. The entry point to Django applications are URLs.  URLs could be as simple as www.example.com, or more complex like www.example.com/whatever/you/want/. When a user accesses a URL, Django will pass it to a view for processing.
2. Requests are Processed by Views. Django Views are custom Python code that get executed when a certain URL is accessed. Views can be as simple as returning a string of text to the user. They can also be made complex, querying databases, processing forms, processing credit cards, etc. Once a view is done processing, a web **response** is provided back to the user. 
3. Most often these web responses are HTML web page, showing a combination of text and images. These pages are created using Django's templating system.

# RSS feeds

Its another format like html pages. RSS feeds are created using XML. 

## What is a back-end?

The back-end is all of the technology required to **process the incoming request and generate and send the response to the client**. This typically includes three major parts:

- The server. This is the computer that receives requests.
- The app. This is the application running on the server that listens for requests, retrieves information from the database, and sends a response.
- The database. Databases are used to organize and persist data.
- The middlewares. Middleware is any code that executes between the server receiving a request and sending a response. 

###### Server

A server is simply a computer that listens for incoming requests. The server runs an app that contains logic about how to respond to various requests based on the [HTTP verb](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods) and the [Uniform Resource Identifier (URI)](https://developer.mozilla.org/en-US/docs/Glossary/URI). The server should not send more than one response per request. 

###### Routing

The pair of an HTTP verb and a URI is called a *route* and matching them based on a request is called *routing*.

###### Middlewares

Middleware is any code that executes between the server receiving a request and sending a response. These middleware functions might modify the request object, query the database, or otherwise process the incoming request. Middleware functions typically end by passing control to the next middleware function, rather than by sending a response.

Eventually, a middleware function will be called that ends the request-response cycle by sending an HTTP response back to the client.

# Transfer data between client and server

HTTP, FTP, SCP are the common File Transfer Protocols. 

The basic point that distinguishes HTTP and FTP is that **HTTP** on request provides a web page from a web server to web browser. On another side, **FTP** is used to upload or download file between client and server.

# The difference between SOAP and REST 

Web services are categorised into two types: SOAP and REST. Typically SOAP and REST are the methods used to call the web services. There are several differences between SOAP and REST. Firstly SOAP relies on XML to assist the services while REST can support various formats such as HTML, XML, JSON, etc. Another significant difference is that SOAP is a protocol. 

SOAP -> XML

REST -> JSON, HTML, XML 

# REST API using Flask

##### When to create an API

In general, consider an API if:

1. Your data set is large, making download via FTP unwieldy or resource-intensive.
2. Your users will need to access your data in real time, such as for display on another website or as part of an application.
3. Your data changes or is updated frequently.
4. Your users only need access to a part of the data at any one time.
5. Your users will need to perform actions other than retrieve data, such as contributing, updating, or deleting data.

If you have data you wish to share with the world, an API is one way you can get it into the hands of others. However, APIs are not always the best way of sharing data with users. If the size of the data you are providing is relatively small, you can instead provide a “data dump” in the form of a downloadable JSON, XML, CSV, or SQLite file. Depending on your resources, this approach can be viable up to a download size of a few gigabytes.

##### **REST (REpresentational State Transfer)** 

is a philosophy that describes some best practices for implementing APIs.

REST means when a client machine places a request to obtain information about resources from a server, the server machine then transfers the current state of the resource back to the client machine.

There are a few methods in this which are as follows.

- **GET** –  select or retrieve data from the server
- **POST** – send or write data to the server
- **PUT** – update existing data on the server
- **DELETE** – delete existing data on the server



##### Create REST using Flask

Flask-RESTful  can be used to build REST APIs. 

# Why do we need to define a constructor

A constructor is generally used to set initial values for any of the fields (aka variables). It may also be used to “set up” anything you need for a class when you instantiate it.

A constructor is a method that is only called at the time of instantiation. You cannot ever explicitly call it, therefore, if you ever want to change the value of a field, you have to create methods other than the constructor to do so.

You don’t have to create a constructor at all. A default constructor will run anyway and set all fields to zero, null, etc., so if that’s all you plan to do, don’t bother.

Additionally, you can create overloaded constructors for different situations. You can have constructors that set values for any combination of variables and you can specify whether those values come from the program instantiating the class.

# Compile time vs Run time

**Compile-time:** the time period in which you, the developer, are compiling your code.

**Run-time:** the time period which a user is running your piece of software.

# interpreted language vs compiled language

We need to convert our source code (high-level language) into binary machine code (low-level), so that our computer can understand it. There are mainly two ways to do these translations. 

1. Compiling the source code. 
2. Interpreting the source code.

Luckily as a programmer, we don't need to worry about these things, because the languages themselves take care of these things, unless we are designing a programming language by ourselves. 


Now let's think of a scenario where I am the programmer and you are a consumer. Now I want to send my coded application to you.

One way to do this is that I compile my source code in my computer using a compiler, which will take my human readable source code, and translate it into a binary machine code. At this point, I have two files, one is the original source code, another one is the machine executable binary code. Now, I can't send my executable binary file to the consumers so that the consumers can run my application. I don't need to send the source code to the consumers. Compiled languages mainly work in this way. Examples of compiled languages: C, C++, Rust, Go. 

Second way to distribute my program to the consumers is to give the source code to the consumer by interpreting my program. In this case, I send the actual source code to the consumer instead of the executable binary file. Then the consumer can download an interpreter that can execute my source code and run it **on the fly**. In this case, the interpreter goes through one line at a time of the source code and convert it to the equivalent binary code, and run it immediately before going to the next line. Examples of interpreted languages: Python, JavaScript, Ruby, PHP. 

Benefits of compiled languages:

1. It's always ready to run. Once it is compiled and I have the executable binary file, I can send that file to millions of consumers immediately. 
2. It can be optimized for CPU usage. So, it is often faster. 
3.  The source code is private. 

Disadvantages/ downsides of compiled languages:

1. If I compile it on PC, then that executable file will not work on Mac. It often needs to execute separately even for different types of CPU on the same operating system. 

Benefits of interpreted languages:

1. We don't need to care about what kind of machine we are working on. Because we don't distribute the executable file, we only send the source code. So, it is more portable and flexible across different platforms. 
2. It is also easier to test and debug because you only need to write your source code and test it. 

Disadvantages/ downsides of interpreted languages:

1. Slower compared to compiled languages.
2. An interpreter is required. 
3. Source code is public. 

But nowadays, most interpreted languages uses JIT (Just-in-time compilation), which makes interpreted languages faster. Read [here](https://medium.com/young-coder/the-difference-between-compiled-and-interpreted-languages-d54f66aa71f0). 

# Loose Coupling vs Tight Coupling

Loose coupling implies that services are independent so that changes in one service will not affect any other. The more dependencies you have between services, the more likely it is that changes will have wider, unpredictable consequences.

In a tightly coupled system, your performance is largely dictated by your slowest component. For example, microservice architectures with services that collaborate via HTTP-based APIs can be vulnerable to cascading performance problems where one component slows down. If your services are decoupled, you will have more freedom to optimise them individually for specific workloads.



# Web Scraping Best Practices

- Never scrape more frequently than you need to.
- Consider [caching](https://pypi.org/project/requests-cache/) the content you scrape so that it’s only downloaded once.
- Build pauses into your code using functions like [`time.sleep()`](https://docs.python.org/3/library/time.html) to keep from overwhelming servers with too many requests too quickly.
