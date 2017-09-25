# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    
    # step 1 -> do tokenize_string for each row in movies['genres']

    all_genres = []
    for row in movies['genres']:
      #genre_list = re.sub(r'[||)|(]', r' ',row.lower()).split()
      genre_list = tokenize_string(row)
      #print(genre_list)
      #print(len(genre_list))
      all_genres.append(genre_list)
    
    # step 2 -> add column tokens in movies
    array = np.array(all_genres)
    
    #print(array[:5])
    #print('#list = ',len(array))
    
    new_movies = movies.assign(tokens = array)
    
    #print(new_movies.head(5))
    return(new_movies)  
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
   
    """
    ###TODO
    
    #print(movies[:5]) 
    
    #step 1 -> build a vocab and df(term)
    vocab = {}
    vocab_list = []
    df = {}
    
    for row1 in movies['tokens']:
       row = list(set(row1))
       for term in row:
          if term not in vocab.keys():
             vocab.setdefault(term,-1)
             
          if term not in df.keys(): 
             df.setdefault(term,1)
          else :
             df[term] += 1
             
             
    #print('vocab = ', vocab)
    
    vocab_list = sorted(vocab.keys(), key = lambda x:x)
    #print('vocab_list = ', vocab_list)
    
    for i,term in enumerate(vocab_list):
         vocab[term] = i
         
    #print('Sorted vocab = ', sorted(vocab.items()))
    #print('df = ',sorted(df.items(), key=lambda x:x[0]))
    
    # step 2 -> Build a csr_matrix for each row of movies['tokens']
    N = len(movies)
    #print('N = ',N)
    
    #[comedy, comedy, comedy, horror]  -> max_k tf(k, d) = 3 
    #[action, comedy,thriller] -> tf(action, d) =1
    # df(i) ->
    #num_features is the total number of unique features across all documents.
    
    csr_array =[]
    
    for row1 in movies['tokens']:
       csr_row = []
       csr_col = []
       csr_data = []
       max_k = 0
       
       max_k = Counter(row1).most_common()[:1][0][1]
       row = list(set(row1))

       #print('removed duplicates =',row)
       for term in row:       
         csr_row.append(0)
         csr_col.append(vocab[term])
         #tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
         tf = Counter(row1)[term]
         #max_k = max_k.most_common()[:1][0][1]
         
         #print('term = %s ---> tf = %d ---> max_k = %d'%(term,tf,max_k))
         tfidf = (tf / max_k) * math.log10(N/df[term])
         csr_data.append(tfidf)
           
         
       #print('csr_row = ',csr_row) 
       #print('csr_col = ',csr_col)
       #print('csr_data=',csr_data)
       X = csr_matrix((csr_data, (csr_row, csr_col)), shape=(1, len(vocab)), dtype=np.float128)
       
       #print('X ->\n',X.toarray())
       csr_array.append(X)
    

    # step 3 -> add column features to movies 
    #print('size of csr_array = ',len(csr_array)) 
    #print('CSR = ',csr_array[:2])  
    new_movies = movies.assign(features = csr_array)
    #print(new_movies.head(2))
     
    return(new_movies,vocab)   
    
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
      


    >>> a = csr_matrix([[ 0.0,  0.0,  0.0,  0.0, 0.43974934,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.3955668,  0.0]] )
    >>> b = csr_matrix([[ 0.0,  0.0,  0.0,  0.0,  0.43974934,  0.0,  0.0,  0.32024863,  0.0, 0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]] )
    >>> sim = cosine_sim(a, b)
    >>> print(sim)
    0.242942001121

    >>> a =  csr_matrix([[ 0.0,  0.0,        0.0,        0.0,        0.43974934,  0.0,  0.0,  0.0,         0.0, 0.0,  0.0,  0.0,  0.0,        0.0,  0.0,        0.0,  0.0,  0.0,  0.0,  0.0,  1.3955668,  0.0]] )
    >>> b =  csr_matrix([[ 0.0,  0.9121797,  1.3099253,  1.1945643,  0.0,         0.0,  0.0,  0.32024863,  0.0,  0.0,  0.0,  0.0,  1.7755414,  0.0,  1.3647367,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,        0.0]] )
    >>> sim = cosine_sim(a, b)
    >>> print(sim)
    0.0

    """
    ###TODO
    #print(type(a))
    #print(type(b))
    #print('a = ',a)
    #print('b = ',b)
    
    #step 1 -> calculate ||a||
    x = (a.data * a.data)
    X = math.sqrt(x.sum())    
    #print('X = ',X)

    #step 2 -> calculate ||b||
    y = (b.data * b.data)
    Y = math.sqrt(y.sum())    
    #print('Y = ',Y)

    # step 3 -> calculate dot(a, b)
    #print('a = ',a.todense())
    #print('b = ',b.todense())
    #print('a = ',a.shape)
    #print('b = ',b.shape)
    
    
    dotProduct = (a).dot(b.transpose())
    #print('dotProduct = ',dotProduct)

    Sum_dot = dotProduct.sum()
    #print('Sum of dotProduct = ',Sum_dot)
    
    # step 4 -> calculate cosine similarity
    if (X != 0) | (Y !=0) :
       Cos_Sim = Sum_dot / ( X * Y )
    else :
       Cos_Sim = 0
    
    #print('Cos_Sim = ',Cos_Sim)
    return(Cos_Sim)
    
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.
   
    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    #print('movies = \n',movies.head(2))
    #print('ratings_train = \n',ratings_train.head(2))
    #print('ratings_test = \n',ratings_test.head(2))
    
    # steps ->  traverse each row of ratings_test 
    # find user's movie To predict and his movie list that he rated 
    # and then find cosine sim

    result = [] 
    for index, row in ratings_test.iterrows():
        test_user = row['userId']
        test_movie = row['movieId'] 
        #print(test_user)
        user_movieTopredict = movies[(movies.movieId == test_movie)]
        a =  user_movieTopredict['features'].values[0]
        #print(user_movieTopredict)
        #print('test_user=%s test_movie=%d'%(test_user,test_movie))

        trains_user = ratings_train[(ratings_train.userId == test_user) & (ratings_train.movieId != test_movie)]
        #print(trains_user)

        wt_avg = 0.0
        sum_cos = 0.0
        for index1, row1 in trains_user.iterrows():          
              #print('\nUserID=%d --> RatedMovie=%s Rating=%f'%(row['userId'],row1['movieId'],row1['rating']))
        
              user_movieHist = movies[(movies.movieId == row1['movieId'])]  
              b =  user_movieHist['features'].values[0]
              #print(type(a))
              #print(type(b))
              #print(a.toarray())
              #print(b.toarray())

              cos_sim = cosine_sim(a,b)
              #print(cos_sim) 
              if cos_sim > 0 :
                 wt_avg += row1['rating'] * cos_sim
                 sum_cos += cos_sim

        if sum_cos > 0:  
            wt_avg = wt_avg / sum_cos
        else : # take a mean if there is no single movie with positive cos-sim
            mean_rating = trains_user['rating']
            #print('mean_rating =',mean_rating)
            #print(np.mean(mean_rating))
            #print(type(mean_rating))
            wt_avg = np.mean(mean_rating)

        #print('test_user=%s wt_avg=%f'%(test_user,wt_avg)) 
        #print('test_user=%d test_movie=%d Actual_rating=%f Pred_rating=%f'%(test_user,test_movie,row['rating'],wt_avg))      
        result.append(wt_avg)

    return(result)    

    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
