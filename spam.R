
setwd("E:/SS/AV/OnlineHack/Kaggle/SMS Spam Collection Dataset")

#Classifying SMS messages as SPAM/NON-SPAM based on their content.

#Load libraries

library(readr)
library(caTools)
library(e1071)
library(randomForest)
library(rpart)
library(rpart.plot)
library(wordcloud)
library(tm)
library(SnowballC)
library(ROCR)
library(pROC)
library(RColorBrewer)
library(stringr)
library(ggplot2)
library(plotly)
library(lattice)

#Get input

sms <- read.csv("spam.csv", stringsAsFactors = F)
str(sms)


# remove empty columns
sms$X <- NULL
sms$X.1 <- NULL
sms$X.2 <- NULL

names(sms) <- c("label","message")
levels(as.factor(sms$label))

sms$label[sms$label == "ham"] <- "non-spam"
sms$label[sms$label == "spam"] <- "spam"

sms$label <- factor(sms$label)

#Text Analysis
#Clean text for analysis

# create bag of words from text

bag <- Corpus(VectorSource(sms$message))
bag <- tm_map(bag, tolower)
bag <- tm_map(bag, PlainTextDocument)
bag <- tm_map(bag, removePunctuation)
bag <- tm_map(bag, removeWords, c(stopwords("english")))
bag <- tm_map(bag, stripWhitespace)
bag <- tm_map(bag, stemDocument)

#Convert bag of words to data frame

frequencies <- DocumentTermMatrix(bag)

# look at words that appear atleast 200 times
findFreqTerms(frequencies, lowfreq = 200)
##  [1] "call" "can"  "come" "day"  "dont" "free" "get"  "good" "got"  "ill" 
## [11] "just" "know" "like" "love" "ltgt" "now"  "send" "text" "time" "want"
## [21] "will"
sparseWords <- removeSparseTerms(frequencies, 0.995)

# convert the matrix of sparse words to data frame
sparseWords <- as.data.frame(as.matrix(sparseWords))

# rename column names to proper format in order to be used by R
colnames(sparseWords) <- make.names(colnames(sparseWords))

str(sparseWords)


sparseWords$label <- sms$label

#Predicting whether SMS is spam/non-spam

#split data into 75:25 and assign to train and test.

set.seed(987)
split <- sample.split(sparseWords$label, SplitRatio = 0.75)
train <- subset(sparseWords, split == T)
test <- subset(sparseWords, split == F)

#Baseline Model(predicting every message as non-spam)
table(test$label)
## 
## non-spam     spam 
##     1206      187
print(paste("Predicting all messages as non-spam gives an accuracy of: ",
            100*round(table(test$label)[1]/nrow(test), 4), "%"))
## [1] "Predicting all messages as non-spam gives an accuracy of:  86.58 %"

#Logistic Regression Model
glm.model <- glm(label ~ ., data = train, family = "binomial")
glm.predict <- predict(glm.model, test, type = "response")

### ROC curve
glm.ROCR <- prediction(glm.predict, test$label)
print(glm.AUC <- as.numeric(performance(glm.ROCR,"auc")@y.values))
## [1] 0.9580329
glm.prediction <- prediction(abs(glm.predict), test$label)
glm.performance <- performance(glm.prediction,"tpr","fpr")
plot(glm.performance)


### selecting threshold = 0.75 for spam filtering
table(test$label, glm.predict > 0.9)
##           
##            FALSE TRUE
##   non-spam  1176   30
##   spam        26  161
glm.accuracy.table <- as.data.frame(table(test$label, glm.predict > 0.9))
print(paste("logistic model accuracy:",
            100*round(((glm.accuracy.table$Freq[1]+glm.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))
## [1] "logistic model accuracy: 95.98 %"

#Support Vector Machine Model

svm.model <- svm(label ~ ., data = train, kernel = "linear", cost = 0.1, gamma = 0.1)
svm.predict <- predict(svm.model, test)
table(test$label, svm.predict)
##           svm.predict
##            non-spam spam
##   non-spam     1184   22
##   spam           26  161
svm.accuracy.table <- as.data.frame(table(test$label, svm.predict))
print(paste("SVM accuracy:",
            100*round(((svm.accuracy.table$Freq[1]+svm.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))
## [1] "SVM accuracy: 96.55 %"

#Decision Trees
tree.model <- rpart(label ~ ., data = train, method = "class", minbucket = 35)

# visualize the decision tree. It tells us about significant words.
prp(tree.model) 


tree.predict <- predict(tree.model, test, type = "class")
table(test$label, tree.predict)
##           tree.predict
##            non-spam spam
##   non-spam     1180   26
##   spam           96   91
rpart.accuracy.table <- as.data.frame(table(test$label, tree.predict))
print(paste("rpart (decision tree) accuracy:",
            100*round(((rpart.accuracy.table$Freq[1]+rpart.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))
## [1] "rpart (decision tree) accuracy: 91.24 %"
#SVM is the most accurate model but rpart is the most interpretable because it tells us about the words that play a significant role in detecting whether a SMS is SPAM or NON-SPAM.


#Random Forest
set.seed(987)
rf.model <- randomForest(label ~ ., data = train, ntree=300, mtry=16, importance=T)

# Importance  tells us about significant words.
importance(rf.model)
varimp <- varImpPlot(rf.model,main = "Importance of each variable")

# Plot error
plot(rf.model, main ="Evolution of the error")

# significant predictors based on MeanDecreaseGini
#dotplot(sort(varimp[,2]),
#        xlab="Variable Importance in DATA\n(predictors to right of dashed vertical line are significant)",
#        panel = function(x,y){
#            panel.dotplot(x, y, col='darkblue', pch=16, cex=1.1)
#            panel.abline(v=abs(min(varimp)),
#                         col='red',
#                         lty='longdash', lwd=2)
#        }
#)

rf.predict <- predict(rf.model, test, type = "class")
table(test$label, rf.predict)
##           tree.predict
##            non-spam spam
##   non-spam     1199    7
##   spam           37  150
rf.accuracy.table <- as.data.frame(table(test$label, rf.predict))
print(paste("random forest accuracy:",
            100*round(((rf.accuracy.table$Freq[1]+rf.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))
#"random forest accuracy: 96.84 %"

## Since tagging a non-spam as spam incurs more cost than otherwise, we can follow
rf.predict <- predict(rf.model, test, type = "prob")
table(test$label, rf.predict[,2] >0.7)

#         FALSE TRUE
#non-spam  1206    0
#spam        53  134

accuracy <- (1206+134)/nrow(test)
accuracy
# 0.9619526


#Wordcloud
bag <- TermDocumentMatrix(bag)
bag <- as.matrix(bag)
bag <- sort(rowSums(bag), decreasing = T)
bag.df <- data.frame(word = names(bag), freq = bag)

set.seed(154)
str(bag)
##  Named num [1:7804] 653 478 447 405 384 366 297 279 276 275 ...
##  - attr(*, "names")= chr [1:7804] "call" "now" "get" "can" ...
wordcloud(words = bag.df$word, freq = bag.df$freq, min.freq = 100,
          max.words=1500, random.order=FALSE, rot.per=0.25,
          colors=brewer.pal(8, "Dark2"),
          scale = c(0.5,3))





























































































































































































































