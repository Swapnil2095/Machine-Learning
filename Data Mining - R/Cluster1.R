# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")

newsGroup <- c("F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\comp.sys.ibm.pc.hardware",
               "F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\comp.sys.mac.hardware",
               "F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\sci.electronics",
               "F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\comp.windows.x",
               "F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\rec.sport.baseball",
               "F:\\IITC\\ADM\\HW\\HW1\\20news-bydate\\20news-bydate-test\\rec.sport.hockey")


newsGroup [1:6]

myCorpus <- Corpus(DirSource(newsGroup, recursive=TRUE),readerControl = list(reader=readPlain))

#str(myCorpus)

myCorpus <- tm_map(myCorpus, tolower)
myCorpus <- tm_map(myCorpus, PlainTextDocument)
myCorpus<- tm_map(myCorpus,removePunctuation)
myCorpus <- tm_map(myCorpus, removeNumbers)
myCorpus <- tm_map(myCorpus, removeWords,stopwords("english"))
myCorpus <- tm_map(myCorpus, stripWhitespace)


dtm <- TermDocumentMatrix(myCorpus,control = list(minWordLength = 4))
dtm_tfxidf <- weightTfIdf(dtm)
m1 <- as.matrix(dtm_tfxidf)
rm(dtm_tfxidf)

m<-t(m1)
rm(m1)

mem_used()

rownames(m) <- 1:nrow(m)
norm_eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)
m_norm <- norm_eucl(m)
num_cluster<-6

mem_used()

cl <- kmeans(m_norm, num_cluster)
round(cl$centers, digits = 1)


mem_used()

for (i in 1:num_cluster) {
  cat(paste("cluster ", i, ": ", sep = ""))
  s <- sort(cl$centers[i, ], decreasing = T)
  cat(names(s)[1:5], "\n")
  mem_used()
  
}