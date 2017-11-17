# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library(factoextra)
library(cluster)
library(NbClust)
library(pryr)

#library("lsa")
#library("lda")
mem_used()

newsGroup <- c("F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\comp.sys.ibm.pc.hardware",
               "F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\comp.sys.mac.hardware",
               "F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\sci.electronics",
               "F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\comp.windows.x",
               "F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\rec.sport.baseball",
               "F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\rec.sport.hockey")


newsGroup [1:6]

news <- Corpus(DirSource(newsGroup, recursive=TRUE),readerControl = list(reader=readPlain))

#news



# data preprocessing
news <- tm_map(news, removeWords,"Subject") 
news <- tm_map(news, removeWords,"Organization") 
news <- tm_map(news, removeWords,"writes") 
news <- tm_map(news, removeWords,"From") 
news <- tm_map(news, removeWords,"lines") 
news <- tm_map(news, removeWords," NNTP-Posting-Host") 
news <- tm_map(news, removeWords,"article")


news <- tm_map(news, tolower) ## Convert to Lower Case 

news <- tm_map(news, removeWords, stopwords("english")) ## Remove Stopwords 

news <- tm_map(news, removePunctuation) ## Remove Punctuations 

news <- tm_map(news, stemDocument) ## Stemming 

news <- tm_map(news, removeNumbers) ## Remove Numbers 

news <- tm_map(news, stripWhitespace) ## Eliminate Extra White Spaces 

news <- tm_map(news , PlainTextDocument)

mem_used()

#DocumentTermMatrix

dtm <- DocumentTermMatrix(news,control=list(wordLengths=c(3,Inf)))
dtm
dim(dtm)

#inspect(dtm[1:5,1:20])
mem_used()


mat4 <- weightTfIdf(dtm)
mat4 <- as.matrix(mat4)

rm(dtm)

mem_used()



set.seed(123)
# Compute and plot wss for k = 1 to k = 15
k.max <- 15
# Maximal number of clusters
data <- mat4
wss <- sapply(1:k.max, function(k){kmeans(mat4, k, nstart=10 )$tot.withinss})
plot(1:k.max, wss, type="b", pch = 19, frame = FALSE, xlab="Number of clusters K", ylab="Total within-clusters sum of squares")
abline(v = 3, lty =2)









norm_eucl <- function(m)
  m/apply(m,1,function(x) sum(x^2)^.5)

mat_norm <- norm_eucl(mat4)

rm(mat4)
mem_used()

set.seed(5)
k <- 3
kmeansResult <- kmeans(mat_norm, k)

kmeansResult$cluster[1:5]

count(kmeansResult$cluster)

library(cluster)
library(NbClust)
library(factoextra)

fviz_cluster(kmeansResult, data = mat_norm , geom = "point", stand = FALSE , ellipse.type = "norm" )



wssplot <- function(dtm, nc=15, seed=1234){

  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(dtm, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")}









#dtms <- removeSparseTerms(dtm, 0.15) # Prepare the data 

freq1 <- colSums(as.matrix(dtm)) # Find word frequencies
#freq2 <- colSums(as.matrix(dtms)) # Find word frequencies

dark2 <- brewer.pal(6, "Dark2") 

wordcloud(names(freq1), freq1, max.words=100, rot.per=0.2, colors=dark2)
#dark2 <- brewer.pal(6, "Dark2")
#wordcloud(names(freq2), freq2, max.words=100, rot.per=0.2, colors=dark2)


#dtm <- TermDocumentMatrix(myCorpus,control = list(minWordLength = 4))
tdm <- TermDocumentMatrix(myCorpus, control=list(wordLengths=c(3,Inf))) #Term Document Matrix 
tdms <- removeSparseTerms(tdm, 0.5) # Prepare the data 

dim(tdm)
dim(tdms)


mem_used()

#convert dtm to matrix
m <- as.matrix(dtm)
dim(m)
#m[1:5,1:5]
  #write as csv file (optional)
  #write.csv(m,file="dtmEight2Late.csv")
#shorten rownames for display purposes
mem_used()
rm(dtm)
mem_used()

#rownames(m) <- paste(substring(rownames(m),1,5),rep("..",nrow(m)),substring(rownames(m), nchar(rownames(m))-12,nchar(rownames(m))-4))
#compute distance between document vectors
d <- dist(m)

m[1:3,1:5]

mem_used()

#run hierarchical clustering using Ward's method
groups <- hclust(d,method="ward.D")
#plot dendogram, use hang to ensure that labels fall below tree
plot(groups, hang=-1)


#k means algorithm, 2 clusters, 100 starting configurations
#kfit <- kmeans(d, 3, nstart=100)
#plot - need library cluster
library(cluster)
#clusplot(m, kfit$cluster, color=T, shade=T, labels=3, lines=0)




#kmeans - determine the optimum number of clusters (elbow method)
#look for "elbow" in plot of summed intra-cluster distances (withinss) as fn of k
wss <- 2:29
for (i in 2:29) wss[i] <- sum(kmeans(d,centers=i,nstart=25)$withinss)
plot(2:29, wss[2:29], type="b", xlab="Number of Clusters",ylab="Within groups sum of squares")

