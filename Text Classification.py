import os
import math
import numpy as np
import glob
  
# Folder Path - change path to run 
path_spam = 'C:/Users/pam'
path_ham = 'C:/Users/ham'
path_mix = 'C:/Users/mix'
path_train = 'C:/Users/train'
path_test = 'C:/Users/test'
  
each_spam = []
each_ham = []
voc_spam = []
voc_ham = []
voc_total = []

# Hard limit on the number of iterations for updating W
hard_limit = 5

# Remove stopwords
def load_stopwords():
    path_stopwords = 'C:/Users/suhyu/Desktop/11th week/stopwords.txt'
    stopwords = []    
    with open(path_stopwords, 'r', errors="ignore") as f:
        for word in f:
            stopwords.append(word.strip('\n'))
            
    return stopwords

# Read spam files
def read_text_file_spam(file_path):
    with open(file_path, 'r', errors="ignore") as f:
        for line in f:
            for word in line.split():
                voc_spam.append(word)
                each_spam.append(word)
                voc_total.append(word)

# Read ham files
def read_text_file_ham(file_path):
    with open(file_path, 'r', errors="ignore") as f:
        for line in f:
            for word in line.split():
                voc_ham.append(word)
                each_ham.append(word)
                voc_total.append(word)

# Perform naive bayes
def naive_bayes(path, unique_words_ham, unique_words_spam, unique_words_total, stopwords):
    os.chdir(path)
    predicted_correctly = 0
    fileCounter = len(glob.glob1(path,"*.txt"))
    for file in os.listdir():
        each_file = []
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            with open(file_path, 'r', errors="ignore") as f:
                for line in f:
                    for word in line.split():
                        if word not in stopwords:
                            each_file.append(word)
                
                # Calculate probabilities for spam and ham
                spam_prob = math.log(len(unique_words_spam) / len(unique_words_total))
                ham_prob = math.log(len(unique_words_ham) / len(unique_words_total))
                
                for word in each_file:
                    try:    
                        spam_prob = spam_prob + math.log((dic_spam[word] + 1) / (len(unique_words_spam) + len(unique_words_total)))
                        ham_prob = ham_prob + math.log((dic_ham[word] + 1) / (len(unique_words_ham) + len(unique_words_total)))
                    except:
                        pass
                
                # Count true positive by comparing spam_prob and ham_prob
                if spam_prob <= ham_prob:
                    if 'ham' in file:
                        predicted_correctly = predicted_correctly + 1
                else:
                    if'spam' in file:
                        predicted_correctly = predicted_correctly + 1
                
                each_file = []
    
    # Naive Bayes accuracy for test sets 
    if 'test' in path:
        print('Naive Bayes accuracy for Test: ', predicted_correctly / fileCounter)

# Generate X matrix with words in all the text files for Logistic Regression
# Row; files (spam + ham randomly)
# Column; distinct words in all files
def generate_matrix(path, stopwords):
    fileCounter = len(glob.glob1(path,"*.txt"))
    # Initialize a matrix with 0s
    data = np.zeros((len(unique_words_total) + 1, fileCounter), dtype=int)
    # Initialize W vector with 1s
    w = [1 for i in range(len(unique_words_total))]
    os.chdir(path)
    file_count = 0
    for file in os.listdir():
        each_file = {}
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            with open(file_path, 'r', errors="ignore") as f:
                for line in f:
                    for word in line.split():
                        if word not in stopwords:
                            each_file[word] = each_file.get(word, 0) + 1
                
                for word in unique_words_total:
                    each_file[word] = each_file.get(word, 0)
                
                # Set true y values for each file indicating it is spam or ham
                if 'spam' in file:
                    each_file['is_spam'] = 1
                elif 'ham' in file:
                    each_file['is_spam'] = 0
       
        # Create a matrix    
        for i in range(len(unique_words_total)):
            data[i][file_count] = each_file[unique_words_total[i]]  
        
        # Train the model (Fit the optimal W vector)
        if 'train' in path:
            count = 0
            summation = 0
            
            # Calculate the summation of (Wi*Xi) where i=1 to N
            for word in unique_words_total:
                try:
                    summation = w[count] * each_file[word]
                except:
                    pass
                count = count + 1
            
            # Calculate a probability for spam given X; P(Y=1|X)
            prob_spam = np.exp(w[0] + summation) / (1 + np.exp(w[0] + summation))
            
            # Gradient descent
            for x in range(len(unique_words_total)):
                gradient_descent = 0
                for f in range(hard_limit):
                    gradient_descent = gradient_descent + data[x][f] * (each_file['is_spam'] - prob_spam)
                
                w[x] = w[x] + (nu_value * gradient_descent) - (nu_value * lambda_value * w[x])
                
            print('Training... ', file_count, 'out of', fileCounter, 'files')
        
        file_count = file_count + 1
                
    return data, w, fileCounter

# Main
if __name__ == '__main__':
    
    ### Naive Bayes
    # Iterate through all file in spam
    # Spam
    dic_spam = {}
    os.chdir(path_spam)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path_spam}\{file}"
            # call read text file function
            read_text_file_spam(file_path)

    # Ham 
    dic_ham = {}
    os.chdir(path_ham)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path_ham}\{file}"
            # call read text file function
            read_text_file_ham(file_path)
            each_ham = []

    for word in voc_ham:
        dic_ham[word] = dic_ham.get(word, 0) + 1
    
    for word in voc_spam:
        dic_spam[word] = dic_spam.get(word, 0) + 1
    
    unique_words_ham = set(voc_ham)
    unique_words_spam = set(voc_spam)
    unique_words_total = list(set(voc_total))
    
    # Use load_stopwords() to see what happens after removing stopwords
    #stopwords = load_stopwords()
    stopwords = []
    
    # Train
    naive_bayes(path_train, unique_words_ham, unique_words_spam, unique_words_total, stopwords)
    
    # Test
    naive_bayes(path_test, unique_words_ham, unique_words_spam, unique_words_total, stopwords)


    ### Logistic Regression
    # Choose different lambda and nu values; 0.01, 0.05, 0.1
    nu_value = 0.01
    lambda_value = 0.01
    #lambda_value = 0.05
    #lambda_value = 0.1

    # Load and generate matrix for train and test
    train_data, w, file_count = generate_matrix(path_train, stopwords)
    test_data, w_empty, file_count2 = generate_matrix(path_test, stopwords)
    
    # Test     
    predicted_correctly = 0
    classification = 0
    count = 0
    
    for file in os.listdir():
        if file.endswith(".txt"):    
            for x in range(len(unique_words_total)):    
                classification = classification + w[x] * test_data[x][count] 
            
            classification = w[0] + classification
            
            if classification > 0 and 'spam' in file or classification <= 0 and 'ham' in file:
                predicted_correctly = predicted_correctly + 1
            count = count + 1
            classification = 0
        
    print('Logistic Regression Accuracy:', predicted_correctly / file_count2)
        


