
# Load
library("tm")
#library("SnowballC")

#library("wordcloud")
#library("RColorBrewer")
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

inspect(dtm[1:5,1:20])
mem_used()

#TermDocumentMatrix
tdm <- TermDocumentMatrix(news, control=list(wordLengths=c(3,Inf))) #Term Document Matrix 
tdm
inspect(tdm[1:5,1:20])
mem_used()

#Verify Frequent Terms 
m <- as.matrix(tdm)
 
v <- sort(rowSums(m), decreasing=TRUE) 

#data frame
d <- data.frame(word = names(v),freq=v) 

str(d)

head(d, 10)

dtms <- removeSparseTerms(dtm, 0.15) # Prepare the data 

mem_used()

str(dtms)
dim(dtms)
inspect(tdm[1:5,1:20])

gc()
#rm()
#gc()
mem_used()

dtm_tfxidf <- weightTfIdf(tdm)
m1 <- as.matrix(dtm_tfxidf)
m<-t(m1)
rownames(m) <- 1:nrow(m)

norm_eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)
m_norm <- norm_eucl(m)
num_cluster<-6
cl <- kmeans(m_norm, num_cluster)
round(cl$centers, digits = 1)

for (i in 1:num_cluster) {
  cat(paste("cluster ", i, ": ", sep = ""))
  s <- sort(cl$centers[i, ], decreasing = T)
  cat(names(s)[1:5], "\n")
  gc()
}

library(factoextra)
library(cluster)
library(NbClust)

rm(d)
gc()
tdm.m <- as.matrix(tdm)
fviz_cluster(cl,data = tdm.m, geom = "point", stand = FALSE , ellipse.type = "norm")


colnames(dtms)
rownames(dtms)
dimnames(dtms)
dimnames(tdm)



freq <- colSums(as.matrix(dtm)) # Find word frequencies 
str(freq)


dark2 <- brewer.pal(6, "Dark2") 

wordcloud(names(freq), freq, max.words=100, rot.per=0.2, colors=dark2)


#dtm_tfxidf2<- weightTfIdf(dtm)

#dtm_tfxidf2

#svd_out <- svd(scale(dtm))

#str(svd_out)

#str(svd_out)

#s <- svd(dtms,nu=5,nv=5,LINPACK = FALSE)


#str(s)

#s <- svd(dtm,nu=15,nv=15,LINPACK = FALSE)


s = svd(dtm)

ds <- diag(s$d[1:50])  # let's now use the first three values
us <- as.matrix(s$u[,1:50])
vs <- as.matrix(s$v[,1:50])
m.approx2 <- us %*% ds %*% t(vs)
m.approx2   # m.approx2 will never be a worst approximation than m.approx1



# we could compute the sum of squared errors
approx.error <- function(m1,m2) {
  sum((m1-m2)^2)
}
approx.error(dtm,m.approx2)

ds <- diag(s$d[1:100])  # let's now use the first three values
us <- as.matrix(s$u[,1:100])
vs <- as.matrix(s$v[,1:100])
m.approx3 <- us %*% ds %*% t(vs)
m.approx3   # m.approx2 will never be a worst approximation than m.approx1



# we could compute the sum of squared errors
approx.error <- function(m1,m2) {
  sum((m1-m2)^2)
}
approx.error(dtm,m.approx3)

library(cluster)
library(NbClust)

#dtm

set.seed(1234)
nc <- NbClust(dtm, min.nc=2, max.nc=15, method="kmeans") 
table(nc$Best.n[1,])

set.seed(123) 
km.res <- kmeans(dtm, k=100, nstart = 25)
# k-means group number of each observation
km.res$cluster

# Visualize k-means clusters
fviz_cluster(km.res, data = dtm, geom = "point", stand = FALSE, frame.type = "norm")
