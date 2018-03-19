import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata
import os
#import pickle
#from decimal import Decimal
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #Initalize parameters
        self.vocab_len = data.X.shape[1] #data.vocab.GetVocabSize()
        self.count_positive = {} #n+
        self.count_negative = {} #n-
        self.num_positive_reviews = 0 #sum(d+)
        self.num_negative_reviews = 0 #sum(d-)
        self.total_positive_words = 0 #unique n+
        self.total_negative_words = 0 #unique n-
        self.P_positive = float(0.0) #P(+)
        self.P_negative = float(0.0) #P(-)
        self.deno_pos = float(1.0)
        self.deno_neg = float(1.0)
        self.Train(data.X,data.Y)
        

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #Estimate Naive Bayes model parameters
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
       
        self.num_positive_reviews = len(positive_indices)
        self.num_negative_reviews = len(negative_indices)

        keys = range(0, X.shape[1])    
        for key in keys:
            word = self.data.vocab.GetWord(key)
            self.count_positive.setdefault(key,(word, 0))
            self.count_negative.setdefault(key,(word, 0))
            
        for i in positive_indices:              
            A = X[i]
            doc_i = [((i, j), A[i,j]) for i, j in zip(*A.nonzero())]
            for data_pt, count in doc_i:
                row_index = data_pt[0]
                col_index = data_pt[1]
                #vocab_word = self.data.vocab.GetWord(col_index)
                word, count_pos = self.count_positive[col_index]
                add_pos_count = count_pos + count
                self.count_positive[col_index] = (word, add_pos_count)
                self.total_positive_words += count

        for i in negative_indices:
            A = X[i]
            doc_i = [((i, j), A[i,j]) for i, j in zip(*A.nonzero())]
            for data_pt, count in doc_i:
                row_index = data_pt[0]
                col_index = data_pt[1]
                #vocab_word = self.data.vocab.GetWord(col_index)
                word, count_neg = self.count_negative[col_index] 
                add_neg_count = count_neg + count
                self.count_negative[col_index] = (word, add_neg_count)
                self.total_negative_words += count

        for key in keys:
            v_word = self.data.vocab.GetWord(key)
            p_word = self.count_positive[key]
            n_word = self.count_negative[key]
            
        self.deno_pos = float(self.total_positive_words + (self.ALPHA * self.vocab_len))
        self.deno_neg = float(self.total_negative_words + (self.ALPHA * self.vocab_len))
        
        print("self.vocab_len = ", self.vocab_len)
        print("self.total_positive_words = ", self.total_positive_words)
        print("self.total_negative_words = ", self.total_negative_words)
        print("self.deno_pos = ", self.deno_pos)
        print("self.deno_neg = ", self.deno_neg)
        
        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X):
        #Implement Naive Bayes Classification
        self.P_positive = float(self.num_positive_reviews / (self.num_positive_reviews + self.num_negative_reviews))
        self.P_negative = float(self.num_negative_reviews / (self.num_positive_reviews + self.num_negative_reviews))
        print("self.P_positive = ", self.P_positive)
        print("self.P_negative = ", self.P_negative)

        pred_labels = []

        log_deno_pos = log(self.deno_pos)
        log_deno_neg = log(self.deno_neg)
            
        P_doc_positive = 0.0
        P_doc_negative = 0.0
        
        sh = X.shape[0]
        for i in range(sh):
            test_doc = X[i].nonzero()
            P_doc_positive = log(self.P_positive)
            P_doc_negative = log(self.P_negative)
            for j in range(len(test_doc[0])):
                # Look at each feature
                row_index = test_doc[0][j]
                col_index = test_doc[1][j]
                word1, count_pos = self.count_positive[col_index] 
                word2, count_neg = self.count_negative[col_index] 
                P_doc_positive += (log(count_pos + self.ALPHA) - log_deno_pos)
                P_doc_negative += (log(count_neg + self.ALPHA) - log_deno_neg)
                pass

            if P_doc_positive >= P_doc_negative:  # Predict positive
                pred_labels.append(1.0)
            else:                                 # Predict negative
                pred_labels.append(-1.0)
        
        return pred_labels

    def Normalize(self, a ,b):

        x = float (a / (a + b))
        y = float (b / (a + b))
        return(x, y)

        pass

    def confusionMatrix(self,Y, predicted_labels ):
        y_actu = pd.Series(Y, name='Actual')
        y_pred = pd.Series(predicted_labels, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
        print(df_confusion)
        pass

    def EvalPrecision(self, TP, TN, FP, FN):

        if TP + FP == 0:
            precision = 0.0
        else:
            precision = TP/(TP + FP)

        return(precision)
        pass


    def EvalRecall(self, TP, TN, FP, FN):

        if TP + FN == 0:
            recall = 0.0
        else:
            recall = TP/(TP + FN)
        return(recall)
        pass


    def LogSum(self, logx, logy):
        #Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)
        return m + log(exp(logx - m) + exp(logy - m))

    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, test, indexes, probThresh = None):

        predicted_labels = []
        TP = FP = TN = FN = 0

        log_deno_pos = log(self.deno_pos)
        log_deno_neg = log(self.deno_neg)         

        for i in indexes:
            #Predict the probability of the i_th review in test being positive review
            #Use the LogSum function to avoid underflow/overflow
            P_doc_positive = log(self.P_positive)
            P_doc_negative = log(self.P_negative)
            test_doc = test.X[i].nonzero()
                        
            predicted_label = 0
            for j in range(len(test_doc[0])):
                row_index = i
                col_index = test_doc[1][j]

                if col_index in self.count_positive.keys():
                    word, count_pos = self.count_positive[col_index]
                    P_doc_positive += log((count_pos + self.ALPHA ) / self.deno_pos)
                else:
                    P_doc_positive += log((self.ALPHA) / self.deno_pos)
                
                if col_index in self.count_negative.keys():
                    word, count_neg = self.count_negative[col_index] 
                    P_doc_negative += log((count_neg + self.ALPHA ) / self.deno_neg)
                else:
                    P_doc_negative += log((self.ALPHA) / self.deno_neg)

            log_sum_exp_deno = self.LogSum(P_doc_positive, P_doc_negative)

            predicted_prob_positive = exp(P_doc_positive - log_sum_exp_deno)
            predicted_prob_negative = exp(P_doc_negative - log_sum_exp_deno)
            

            if(probThresh): #probThresh set
                print(probThresh)
                if predicted_prob_positive >= probThresh:
                    predicted_label = +1.0
                    predicted_labels.append(+1.0)

                    if test.Y[i] == +1.0:
                        TP += 1
                    else:
                        FP += 1

                else:
                    predicted_label = -1.0
                    predicted_labels.append(-1.0)

                    if test.Y[i] == -1.0:
                        TN += 1
                    else:
                        FN += 1

            else: #probThresh not set

                if predicted_prob_positive > predicted_prob_negative:
                    predicted_label = +1.0
                    predicted_labels.append(+1.0)

                    if test.Y[i] == +1.0:
                        TP += 1
                    else:
                        FP += 1

                else:
                    predicted_label = -1.0
                    predicted_labels.append(-1.0)

                    if test.Y[i] == -1.0:
                        TN += 1
                    else:
                        FN += 1

            #print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])

        self.confusionMatrix(test.Y, predicted_labels)
        precision = self.EvalPrecision(TP, TN, FP, FN)
        print("Precision = ",precision)
        
        recall = self.EvalRecall(TP, TN, FP, FN)
        print("Recall = ",recall)
        
        return(precision, recall)
        pass

    # Evaluate performance on test data
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        self.confusionMatrix(test.Y, Y_pred)
        return ev.Accuracy()
    
    def DrawGraph(self, precision, recall):
                
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.title("Precision Vs Recall")
        plt.plot(precision,recall)
        plt.savefig('PrecisionVsRecall.png')
        plt.show()                
        pass


    def Top20Words(self):
        
        pos_wt = {}
        neg_wt = {}
       
        log_deno_pos = log(self.deno_pos)
        log_deno_neg = log(self.deno_neg)

        for col_index in self.count_positive.keys():
            word, count_pos = self.count_positive[col_index]
            P_word_positive = (log(count_pos + self.ALPHA ) - log_deno_pos)

            word, count_neg = self.count_negative[col_index] 
            P_word_negative = (log(count_neg + self.ALPHA ) - log_deno_neg)
            
            pos_wt[word] = (P_word_positive - P_word_negative) 
            neg_wt[word] = (P_word_negative - P_word_positive)             

        sorted_pos_wt = sorted(pos_wt.items(), key=lambda x: x[1], reverse = True)[:20]        
        print("Top 20 +ve words = ",sorted_pos_wt)
        
        sorted_neg_wt = sorted(neg_wt.items(), key=lambda x: x[1], reverse = True)[:20]        
        print("Top 20 -ve words = ",sorted_neg_wt)        
        
        pass
    
if __name__ == "__main__":
    
    path = os.path.dirname(os.path.abspath(__file__))  
    #path = os.getcwd()
    data_path = os.path.join(path,"../" + sys.argv[1])    
    print("Data Path = ",data_path)
    
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" %data_path)
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" %data_path, vocab=traindata.vocab)
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    print("Test Accuracy: ", nb.Eval(testdata))
    
    print("Top +ve and -ve words-")
    nb.Top20Words()
    
    print("Evaluating NaiveBayes accuracy for different smoothing(alpha) values-")        
    alpha = [0.1, 0.5, 1.0, 5.0, 10.0]
    for a in alpha:
        print("Evaluating for alpha = ", a)
        nb = NaiveBayes(traindata, a)
        accuracy = nb.Eval(testdata)
        print("Test Accuracy: ", accuracy)

    print("PredictProb for different probThresh values-")
    precision = [] 
    recall = []
    indexes = range(testdata.X.shape[0])
    #indexes = range(10)
    probThresh = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for th in probThresh:
        print("Prob Threshould set = ", th)
        p, r = nb.PredictProb(testdata, indexes, th)
        precision.append(p)
        recall.append(r)
        print("precision = %s\trecall = %s" %(p, r))
    
    print("Graph : Precicion vs. Recall")
    nb.DrawGraph(precision, recall)    