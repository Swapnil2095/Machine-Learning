"""
sumarize.py
"""

# Imports needed.

from operator import itemgetter, attrgetter
from collections import Counter, defaultdict
import pickle
import math

def read_pickle_dump(filename):
    
    file = open(filename,'rb')
    fl = pickle.load(file)
    file.close()

    #print(fl[:2])
    return(fl)

    pass



def write_summary(users,messages,comm,classes):

    #`summary.txt`
    #Number of users collected:
    #Number of messages collected:
    #Number of communities discovered:
    #Average number of users per community:
    #Number of instances per class found:
    #One example from each class:
    num_clust = len(comm)
    i = 0
    size = 0

    avg_followers = 0
    for user in users:
        avg_followers += len(user['Followers'])

    avg_followers = avg_followers/len(users)


    # Write a file
    with open("summary.txt", "wt") as out_file:
        out_file.write("\t---Summary---" + "\n")
        out_file.write("Number of users collected: " + str(len(users)) + "\n")
        out_file.write('Average Number of Followers collected per user:' + str(math.floor(avg_followers)) + "\n")

        out_file.write("Number of messages collected: " + str(len(messages)) + "\n")
        out_file.write("Number of communities discovered: "+ str(len(comm)) + "\n") 

        out_file.write("Number of cluster = " + str(num_clust) + "\n")

        while i < num_clust :
             out_file.write("Cluster " + str(i) +  " Size = " + str(comm[i].order()) + "\n") 
             size += comm[i].order()  
             i += 1

        out_file.write('Average number of users per community:' + str(math.floor(size/len(comm))) + "\n")

        out_file.write('Number of Classes = ' + str(len(classes)) + "\n")
        out_file.write('Number of instances per class found: ' + "\n")
  
        for clas in classes.keys():

            if clas == 'positive' :
               out_file.write('Size of positive class = ' + str(len(classes['positive'])) + "\n")

            if clas == 'negative' :
               out_file.write('Size of negative class = ' + str(len(classes['negative'])) + "\n")

            if clas == 'neutral' :
               out_file.write('Size of neutral class = ' + str(len(classes['neutral'])) + "\n")

        out_file.write('One example from each class:' + "\n")
        out_file.write('Top Positive Tweets ->' + "\n") 
        for tweet, pos, neg in sorted(set(classes['positive']), key=lambda x: x[1], reverse=True)[:1]:
            out_file.write('(pos,neg) = ('+ str(pos) +',' + str(neg) +') -> Tweet = '+ tweet + "\n")

        out_file.write('Top Negative Tweets ->' + "\n")
        for tweet, pos, neg in sorted(set(classes['negative']), key=lambda x: x[2], reverse=True)[:1]:
            out_file.write('(pos,neg) = ('+ str(pos) +',' + str(neg) +') -> Tweet = '+ tweet + "\n")
    
        out_file.write('Top Neutral Tweets ->' + "\n")
        for tweet, pos, neg in sorted(set(classes['neutral']), key=lambda x: x[2], reverse=True)[:1]:
            out_file.write('(pos,neg) = ('+ str(pos) +',' + str(neg) +') -> Tweet = '+ tweet + "\n")
 

    pass

def print_summary(users,messages,comm,classes):

    #`summary.txt`
    #Number of users collected:
    #Number of messages collected:
    #Number of communities discovered:
    #Average number of users per community:
    #Number of instances per class found:
    #One example from each class:


    print('Number of users collected: ',len(users))
    avg_followers = 0
    for user in users:
        avg_followers += len(user['Followers'])

    avg_followers = avg_followers/len(users)

    print('Average Number of Followers collected per user:',math.floor(avg_followers))
    print('Number of messages collected: ',len(messages))
    print('Number of communities discovered: ',len(comm))

    num_clust = len(comm)
    print('Number of cluster = ',num_clust)

    i = 0
    size = 0
    while i < num_clust :
          print('Cluster %d Size = %d'%(i,comm[i].order()))  
          size += comm[i].order()  
          i += 1

    print('Average number of users per community:', math.floor(size/len(comm)))

    print('Number of Classes = ',len(classes))
    print('Number of instances per class found: ')
  
    for clas in classes.keys():

        if clas == 'positive' :
           print('Size of positive class = ',len(classes['positive']))

        if clas == 'negative' :
           print('Size of negative class = ',len(classes['negative']))

        if clas == 'neutral' :
           print('Size of neutral class = ',len(classes['neutral']))

    print('One example from each class:')
    print('Top Positive Tweets ->')
    for tweet, pos, neg in sorted(set(classes['positive']), key=lambda x: x[1], reverse=True)[:1]:
        print('(pos,neg) = (%d,%d) -> Tweet = %s'%(pos, neg, tweet))

    print('Top Negative Tweets ->')
    for tweet, pos, neg in sorted(set(classes['negative']), key=lambda x: x[2], reverse=True)[:1]:
        print('(pos,neg) = (%d,%d) -> Tweet = %s'%(pos, neg, tweet))
    
    print('Top Neutral Tweets ->')
    for tweet, pos, neg in sorted(set(classes['neutral']), key=lambda x: x[2], reverse=True)[:1]:
        print('(pos,neg) = (%d,%d) -> Tweet = %s'%(pos, neg, tweet))
 
    pass


def main():
    """ Main method. You should not modify this. """


    # read dump files
    Channels = read_pickle_dump('Channels.pkl')
    Tweets = read_pickle_dump('Tweets.pkl')
    Components = read_pickle_dump('Components.pkl')
    Classes = read_pickle_dump('Classes.pkl')

    # print summary to file
    print_summary(Channels,Tweets,Components,Classes)
    
    # write summary to file
    write_summary(Channels,Tweets,Components,Classes)

    print('Summary Done Successfully!!')

if __name__ == '__main__':
    main()

