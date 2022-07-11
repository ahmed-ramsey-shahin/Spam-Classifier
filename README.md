# Spam-Classifier
A spam classifier based on SVM with Gaussian Kernel

# The dataset
The dataset I used to train the mode is the Spam Assassin dataset you can find it here
https://spamassassin.apache.org/old/publiccorpus/

# The hierarchy of the project
```bash
Spam-Classifier:.
├───Spam Assassin Dataset
│   ├───non-spam
│   └───spam
├───classifier.obj
├───dataset.npy
├───email.txt
├───predict.py
├───preprocess_data.py
├───train.p
└───vocab_list.json
```

Note that the dataset.npy file is compressed into a file called dataset.7z with compression level "9 - Ultra" thus it might take sometime to be extracted, the extracted file will be 282 MB

# The files used in the project
### Spam Assassin Dataset
#### non-spam
This folder contains all the non-spam emails (You can download them from the link above if you want to preprocess the data yourself)
#### spam
This folder contains all the spam emails (You can download them from the link above if you want to preprocess the data yourself)

----
###  classifier.obj
This file contains a pre-trained model with 0.504% error rate on the cross validation set, and 0.620% error rate on the test set. Both sets are provided as a numpy array in the dataset.npy file.
This file is saved and loaded using ```pickle```

----
### dataset.npy
This file contains the dataset after doing pre-processing and feature extraction as explained in the preprcoess_data.py data section

----
### email.txt
A file containing a spam email which is not part of the dataset I just got it from my own email to test the model

----
### predict.py
A python script used to classify emails, in order for this script to work it needs three extra files
| File | Content |
| ---- | ------- |
| email.txt | Contains the email we need to predict |
| classifier.obj | Contains the classifier object which is saved using ```pickle``` |
| vocab_list.json | Contains the vocabulary list used to extract the features |
if these files are given you can classify any given email

----
### preprocess_data.py
This file does all the pre-processing required on the dataset, and it outputs 2 files, The ```vocab_list.json```, and the ```dataset.npy```.
1. First the script starts to remove the headers of the email by removing any data before the first empty line.
2. Second it starts pre-processing the emails content by doing these steps
  * Convert all the letters to lower case letters
  * Remove any tag of the form &nbsp;
  * Replce any numbers with the actual word "number" so we can use any number as a feature
  * Replace any url with the word "httpaddr" so we can use any url as a feature
  * Replace any email address with the word "emailaddr" so we can use any email address as a feature
  * Replace prices with the word "dollar" so we can use any price as a feature
  * Remove punctuation
  * Remove extra spaces
  * Put the whole email on one line
  * Use the ```PorterStemmer``` from the ```nltk``` package to do a word by word [stemming](https://www.wikiwand.com/en/Stemming)
3. Extract the vocabulary list which contains any word that was used more than 100 times in the emails (This list is saved in the file ```vocab_list.json``` along with and an index with every word for example ```{"i": 1}``` which means that the index of "i" is 1) There are 2,873 words in this file.
5. Extracting the features using vocabulary list mentioned above, since there are 2,873 words in the vocabulary list, then there are 2,873 features for every email. The index of the word from the json file is the index of the feature for example the first three features are "i", "cant", and "thi" thus the first three features are the same. The feature value contains 1 if the word is present, and 0 otherwise. For example if the words "i", and "cant" are present, and "thi" is not, then the first three features are going to be [1, 1, 0]
6. Another row is added for every feature vector which contains the classification of the email 1 if spam, and 0 otherwise
7. After extracting the features for all the emails in the dataset they are transposed, and put into a numpy array where the rows of the array are the features of the emails ```X = np.array([features_of_1st_email, features_of_2nd_email, ..., features_of_nth_email])``` the final shape of the numpy array is (12896, 2874)
8. Shuffle the rows of the numpy array
9. The features are saved to the file ```dataset.npy```

----
### train.py
The train.py file loads the ```dataset.npy``` file which contains the shuffled data, and it splits the data into 3 sets, train set, cross validation set, and test set the size of these sets are as follows 60%, 20%, and 20%. The train set is used to train the SVM model with a Gaussian Kernel, the cross validation set is used to pick the value of C, and gamma, and then the test set model is used to generalize the error rate. Then it saves the model to the classifier.obj file using ```pickle```.

----
### vocab_list.json
Contains the vocabulary list mentioned above
