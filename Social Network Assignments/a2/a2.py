# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", True)
    array(['necronomicon', 'geträumte', 'sünden.<br>hi'], 
          dtype='<U13')
          
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", False)
    array(['necronomicon', 'geträumte', 'sünden', 'br', 'hi'], 
          dtype='<U12')
    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
      
    """
    ###TODO

    word_list = []
    
    if keep_internal_punct==True :
     
       for word in doc.split():
          match = re.sub(r'^\W+', r'',word.lower())
          match = re.sub(r'\W+$', r'',match)
          word_list.append(match)
       
       word_list = list(filter(None, word_list))    
      
    else:
     
       #\W -> Matches any non-alphanumeric character; 
       #this is equivalent to the class [^a-zA-Z0-9_]. 
       
       word_list = re.sub('\W+', ' ', doc.lower()).split()
      
    return(np.array(word_list))   


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    
    for token in tokens :
       t = 'token=' + token
          
       if t not in feats.keys() : 
           feats.setdefault(t,1)
       else :
           feats[t] += 1      
    


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    
    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(['a', 'b', 'a'], feats, 3)
    >>> sorted(feats.items())
    [('token_pair=a__a', 1), ('token_pair=a__b', 1), ('token_pair=b__a', 1)]
    
    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd' ,'e']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 2), ('token_pair=c__e', 1), ('token_pair=d__e', 1)]

    """
    ###TODO
    #print('tokens=',tokens)
    #print('feats=',feats)
    
    index = 0
    while(index < (len(tokens)-k+1)):

       #print('\tIndex=',index)  
          
       window = tokens[index:index+k]                       
       #print('window = ',window)    
       
       window_comb = []         
       comb = combinations(window,2)   
       for c in comb:
          window_comb.append(c) 
          pair = 'token_pair=' + c[0] + '__' + c[1]
          #print(pair)
          
          if pair not in feats.keys() : 
               feats.setdefault(pair,1)
          else :
               #print('Duplicate')
               feats[pair] += 1
                                 
       #print('window_comb',window_comb)
                       
       index += 1
       
    #print('feats=',feats)
    
       

neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]

    """
    ###TODO

    # step 1 -> make lower-case
    # not getting why need to make lower case here -> doc-test need to check
    word_list = [x.lower() for x in tokens]
    
    
    nw = 0
    pw = 0
    
    # step 2 -> count pos/neg words
    for token in word_list:
        if token in neg_words: # returns True/False -> faster
            nw += 1
        if token in pos_words:
            pw += 1

    # step 3 -> add feature to feats
    feats.setdefault('neg_words',nw)
    feats.setdefault('pos_words',pw)
    
    pass



def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    
    # step 1 -> feats creation
    feats = defaultdict(lambda: 0)
    
    # step 2 -> call particular feature function for each feature
    for feature in feature_fns :        
        feature(tokens,feats)

    # step 3 -> sort before return
    return(sorted(feats.items(), key=lambda x: x[0]))

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
 
    """
    ###TODO
    
    features =  []
    feature_freq = {}
    vocabulary = {}
    
    # 2 case : for vocab
    # case 1: 
    if (vocab == None):
        
       for doc in tokens_list: 
          #print('doc#=%d tokens=%s'%(i,doc))  
          data = featurize(doc,feature_fns)
          #print('data=',data)
                    
          for feature in data:   
                if feature[1] > 0 :                  
                   if feature[0] not in feature_freq.keys():
                      feature_freq.setdefault(feature[0],1)  
                   else :
                      feature_freq[feature[0]] += 1
                           
                   if feature[0] not in vocabulary.keys() :
                      vocabulary.setdefault(feature[0], None)                   
                 
          features.append(data)
          
       # sort vocab according to features (alphabetical order)
       vacab_list = sorted(feature_freq.keys(), key =lambda x: x,reverse=False)
        
       for colIndex,term in enumerate(vacab_list) :
           #print('colIndex = %d, term = %s'%(colIndex,term))
           vocabulary[term] = colIndex

    else: # case 2        
         
         # vocab already present
         #print('Vocab already present')
         vocabulary = vocab.copy()         
        
         
         for doc in tokens_list:             
            data = featurize(doc,feature_fns)  
            
            test_data = []                                 
            for feature in data:                 
                # only take feature present in vocab                
                if feature[0] in vocabulary.keys():
                   #print('feature = ',feature)  
                   if feature[1] > 0 :  
                      test_data.append(feature)            
                      if feature[0] not in feature_freq.keys():
                         feature_freq.setdefault(feature[0],1)  
                      else :
                         feature_freq[feature[0]] += 1
                         
            #print('test_data = ',len(test_data))                  
            features.append(test_data)
            #test_data.clear()
         #print('features = ',features)
    
     
    # build a csr_matrix    
    row = []
    col = []
    data = []  
     
    for docID,feat_list in enumerate(features) :
       for term in feat_list:
           if (feature_freq[term[0]] >= min_freq): # (zero values are not stored)
                                                  
               row.append(docID)
               col.append(vocabulary[term[0]])
               data.append(term[1])
     
    #print('row =',row)
    #print('col =',col)
    #print('data=',data)
   
    X = csr_matrix((data, (row, col)), shape=(len(features), len(vocabulary)), dtype=np.int64)
    
    #print('X ->')
    #print(X.toarray())
    #print('      size of X = ',X.get_shape())
    
    return(X, vocabulary)    

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO

    cv = KFold(n=len(labels),n_folds=k)
    accuracies = []

     
    for train_indices, test_indices in cv:
        
        clf.fit(X[train_indices], labels[train_indices])
        predicted = clf.predict(X[test_indices])
        acc = accuracy_score(labels[test_indices], predicted)
        accuracies.append(acc)
    
    #print('accuracies = ',accuracies) 
    #avg = np.mean(accuracies,dtype=np.float64)
    return(np.mean(accuracies,dtype=np.float64))   
    
def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
      
    """
    ###TODO
    
    # gettting feature's 7 combinations
    feature_comb = [] 
    i = 1
    
    while i <= len (feature_fns) :
       comb = combinations(feature_fns,i)
       i += 1 
   
       for c in comb:
          feature_comb.append(c) 
          #print(c)

    #for option in feature_comb:
         #print(option)
 
    # LogisticRegression object
    

    keys = ['punct','features','min_freq','accuracy']
    results = []
    dicts = []       
    feature_dict = {}
    
    # setting on punct,mi_freq and conmbination of feature
    for punct in punct_vals:       
       tokens_list = [tokenize(d,punct) for d in docs]
       
       for freq in min_freqs :       

           for features in feature_comb:
                              
               #print('MinFreq = %d Punct = %s fetures = %s'%(freq,punct,features))
               X, vocabulary = vectorize(tokens_list,features,freq)
               clf = LogisticRegression()
               avg_acc = cross_validation_accuracy(clf, X, labels, k=5)
               #print('Avg accuracy = %f'%(avg_acc))
               #print('vocab size =',len(vocabulary)) 
            
               result = [punct,features,freq,avg_acc]

               feature_dict = dict(zip(keys, result))

               dicts.append(feature_dict)
    
   
    # sort dict on accuracy     
    dicts.sort(key=lambda x:(-x['accuracy'],-x['min_freq']))
    #print('dicts = ',dicts)
        
    return(dicts)

def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    #print(results)
        
    #step 1 -> sort accuracies and get x and y
                # x = setting
                # y = sorted list of accuracies
    #results.sort(key=lambda x:(x['accuracy'])) 
    # don't use it ->it will change results from main as well
    
    #print(results)

    acc = []
    
    x = list(range(len(results)))
    
    for d in results:
        #print('dict=',d)
        acc.append(d['accuracy'])
    
    acc.sort(key=lambda x:(x))
    #print('acc =  ',acc)
    
    #step 2 -> plot figure
    fig1 = plt.figure(1)   
    plt.plot(x,acc)
    plt.ylabel('accuracy')
    plt.xlabel('settings')
    
    plt.show()
    
    fig1.savefig('accuracies.png')
    


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    min_freq = {}
    feature = {}
    punct = {}
    
    #step 1 -> loop over results to get list of values for particular 
             # setting of punct, features,min_freq 
    #keys = ['punct','features','min_freq','accuracy']
    
    for d in results:
    
        if d['min_freq'] not in min_freq.keys():
           min_freq.setdefault(d['min_freq'],[]).append(d['accuracy'])
        else :
           min_freq[d['min_freq']].append(d['accuracy'])
           
        if d['punct'] not in punct.keys():
           punct.setdefault(d['punct'],[]).append(d['accuracy'])
        else :
           punct[d['punct']].append(d['accuracy'])
               
        if d['features'] not in feature.keys():           
           feature.setdefault(d['features'],[]).append(d['accuracy'])
        else :           
           feature[d['features']].append(d['accuracy'])
           
                                  
    #print('min_freq = ',min_freq)
    #print('feature  = ',feature)
    #print('punct    = ',punct)  
    
    # step 2 -> find average for each setting
    tuple_list = [] 
    for fet in feature.keys():
      
        t1 =  'features='
        for f in fet:
            t1 += f.__name__  +  ' '
            
        #print(t1)
        avg = np.mean(feature[fet],dtype=np.float64)        
        tuple_list.append((avg,t1))


    #print('After features result = ',result) 
    
    for freq in min_freq.keys():
        t1 =  'min_freq=' + str(freq)
        avg = np.mean(min_freq[freq],dtype=np.float64)        
        tuple_list.append((avg,t1))
        
    #print('After mean_freq result = ',result)        
   
    for pun in punct.keys():
        t1 =  'punct=' + str(pun)
        avg = np.mean(punct[pun],dtype=np.float64)        
        tuple_list.append((avg,t1))
 
    #print('After punct result = ',result)               
    
         
    tuple_list.sort(key=lambda x:(-x[0]))
    #print('2.Sorted result = ',result)    
     
     
    return(tuple_list)   
     
    pass


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    #print('best_result = ',best_result)
    #print('labels = ',labels)
    
    #step 1 -> call tokenize
    #keys = ['punct','features','min_freq','accuracy']
    tokens_list = [tokenize(d,best_result['punct']) for d in docs]
    
    #step 2 -> call vectorize 
    #vocabulary = {}    
    X, vocab = vectorize(tokens_list, best_result['features'], best_result['min_freq'], vocab=None) # vocab = None
    
    #step 3 -> do LogisticRegression
    clf = LogisticRegression()
    clf.fit(X,labels)
    
    #predictions = clf.predict(X)
    #print('Testing accuracy=%f' %
          #accuracy_score(labels, predictions))

    return (clf,vocab) #sending clf too ***


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
      
    """
    ###TODO
    
    # step 1 -> get .coef_
    coefficient = clf.coef_[0]  #***** 
    
    # step 2 -> check label and sort
    if label == 1: # positive class -> descending sorting
        # get indices of sorted list i.e. [2,3,1] -> sorting [1,2,3] -> indices[3,1,2]
        top_coef_ind = np.argsort(coefficient)[::-1][:n] # requires very less time by this methos of sorting and get sorted element's indices       
    
    if label == 0: # negative class -> ascending sorting
        top_coef_ind = np.argsort(coefficient)[::1][:n]
        
    
    #step 3 -> get all top coefficient' indices
    #print('top_coef_ind = ',top_coef_ind)
    top_coef = abs(coefficient[top_coef_ind])
    #print('top_coef = ',top_coef)
    
    #step 4 -> get all top coefficient' terms i.e. tokens
    rev_Vocab = {}
    
    for term,colId in vocab.items():
        rev_Vocab.setdefault(colId,term)
    #alternatives -> check for fasted 
    #vocab.__class__(map(reversed, vocab.items()))
    #rev_Vocab = lambda vocab: {v:k for k, v in vocab.items()}
    #rev_Vocab = lambda vocab: dict( zip(vocab.values(), vocab.keys()) )
    
       
    top_coef_terms = []
    
    for colId in top_coef_ind:
         top_coef_terms.append(rev_Vocab[colId])
    
    #step 5 -> get touple (top_coef_terms, top_coef) and send
    return ([x for x in zip(top_coef_terms, top_coef)])

    

def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    
    # step 1 -> read data
    test_docs, test_labels = read_data(os.path.join('data','test'))
            
    # step 2 -> call tokenize
    #keys = ['punct','features','min_freq','accuracy']
    tokens_list = [tokenize(d,best_result['punct']) for d in test_docs]
    
    # step 3 -> call vectorize  ->vocab is not None
    X_test, vocab = vectorize(tokens_list, best_result['features'], best_result['min_freq'], vocab)
    
    #print('Sizes-> test_docs = %d, test_labels = %d, tokens_list =%d'%(len(test_docs),len(test_labels),len(tokens_list))) 
    #print('Setting -> feature=',best_result['features'])
    #print('punc =',best_result['punct'])
    #print('Min_freq =',best_result['min_freq'])     
    
    return (test_docs, test_labels, X_test)



def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    #print('test_labels =',test_labels)        

    #step 1 -> find missclassified
    predicted = clf.predict(X_test)
    
    #print('predicted = ',predicted)
    #acc = accuracy_score(test_labels, predicted)
    #print('acc = ',acc )
    
    misclassified = np.where(predicted != test_labels)
    
    #print('misclassified = ',misclassified)
    #print('misclassified = ',misclassified[0])
    #print('misclassified = ',misclassified[0][0])

    #step 2 -> find predicted probabilities
    probab = clf.predict_proba(X_test)
    
    #print('probab = ',probab)
    
    #step 3 -> collect all misclassified docs with all required info
    misclassified_docs = []
    
    for i in misclassified[0]:
        #print(i)
        misclassified_docs.append( ( test_labels[i], predicted[i], probab[i][predicted[i]], test_docs[i] ) ) 
		
    #step 4 -> sort in descending order of the predicted probability for the incorrect class 	
    sorted_docs = sorted(misclassified_docs,key=lambda x:(-x[2]))[:n]

    #step 5 -> print all value
    for doc in sorted_docs :
        print('\n',"truth=",doc[0]," predicted=",doc[1]," proba=",doc[2])
        print(str(doc[3])) #.encode("utf-8")
    


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))    
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))
    
    
    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))
    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    #print('CSR Test ->')
    #print(X_test.toarray())
    #print('predictions = ',predictions)
    #print('test_labels = ',test_labels)

    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))
          
    
    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)
    
            
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)

if __name__ == '__main__':
    main()
