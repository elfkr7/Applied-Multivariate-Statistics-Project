##MULTIVARIATE POJECT

#load training dataset
data_train=read.table("vowel-train.txt",sep=",",header = T)[,-1]
data_train_vars=data_train[-1]##the same data set except for the class indicator variable y
##1
S=cov(data_train_vars)
round(S,3)
##there is no big difference between the variables' variances which can require any standardization
mean=colMeans(data_train_vars)
mean
##mean is not zero, so the variables can be centralized. According to Janson and Wichern(2007),
##when mu=! 0, it is the mean-centered principal component yi = ei( x - mu) that has mean 0 and 
##lies in the direction ei.  This can be done without loss of generality because the normal random 
##vector X can always be translated to the normal random vector W =X - mu and E(W) = 0. However, Cov(X) = Cov(W). 

#eigenvalues
eigen=eigen(S)
round(eigen$values,3)
round(eigen$vectors,3)
pc1=prcomp(data_train_vars,center =  T)
round(pc1$sdev,3)
round(pc1$center,3)
round(pc1$rotation,3)
summary(pc1)

##As seen in the table of importance of components, 6 principal component would be enough 
##to explaint 90% of the total (standardized) sample variance in the training data.

##2
library(MASS)
library(dplyr)
scored_train_data=pc1$x[,1:6]
lda1= lda(scored_train_data,grouping = data_train$y)
lda1
train_estimated=predict(lda1,scored_train_data)
misclass_rate_train=sum(if_else(train_estimated$class!=data_train$y,1,0))/length(data_train$y)
misclass_rate_train
##test prediction

data_test=read.table("vowel-test.txt",sep=",",header = T)[,-1]
data_test_vars=data_test[-1]##the same data set except for the class indicator variable y
##pca for test data

pc1_test=prcomp(data_test_vars,center =  T)
scored_test_data=pc1_test$x[,1:6]
test_predicted=predict(lda1,scored_test_data)
misclass_rate_test=sum(if_else(test_predicted$class!=data_test$y,1,0))/length(data_test$y)
misclass_rate_test

##3
##QDA for train data
library(MASS)
qda1= qda(scored_train_data,grouping = data_train$y)
qda1
train_estimated_qda=predict(qda1,scored_train_data)
misclass_rate_train_qda=sum(if_else(train_estimated_qda$class!=data_train$y,1,0))/length(data_train$y)
misclass_rate_train_qda

##QDA for test
test_predicted_qda=predict(qda1,scored_test_data)
misclass_rate_test_qda=sum(if_else(test_predicted_qda$class!=data_test$y,1,0))/length(data_test$y)
misclass_rate_test_qda

##Comment:QDA gives the same testing error rate with 0.8614719. Eventhough QDA and LDA predicts the same
##number of class correctly, they predict different observations correctly.

##4
##lda for original train data
lda2_train_org=lda(data_train_vars,grouping=data_train$y)
train_estimated_org=predict(lda2_train_org,data_train_vars)
misclass_rate_train_org_lda=sum(if_else(train_estimated_org$class!=data_train$y,1,0))/length(data_train$y)
misclass_rate_train_org_lda

##lda for original test data

test_predicted_org_lda=predict(lda2_train_org,data_test_vars)
misclass_rate_test_org_lda=sum(if_else(test_predicted_org_lda$class!=data_test$y,1,0))/length(data_test$y)
misclass_rate_test_org_lda

##qda for original train data
qda2_train_org=qda(data_train_vars,grouping=data_train$y)
train_estimated_org_qda=predict(qda2_train_org,data_train_vars)
misclass_rate_train_org_qda=sum(if_else(train_estimated_org_qda$class!=data_train$y,1,0))/length(data_train$y)
misclass_rate_train_org_qda

##qda for original test data

test_predicted_org_qda=predict(qda2_train_org,data_test_vars)
misclass_rate_test_org_qda=sum(if_else(test_predicted_org_qda$class!=data_test$y,1,0))/length(data_test$y)
misclass_rate_test_org_qda

##Comparison Table

##get all rates  together

rates_table=matrix(c(misclass_rate_train,misclass_rate_test,misclass_rate_train_qda,misclass_rate_test_qda,
               misclass_rate_train_org_lda,misclass_rate_test_org_lda,misclass_rate_train_org_qda,
               misclass_rate_test_org_qda),ncol = 2, byrow = T)

colnames(rates_table)=c("Training", "Testing")
rownames(rates_table)=c("LDA after PCA", "QDA after PCA","LDA with original data", "QDA with original data")

round(rates_table,3)

##Comment: Prediction with the model are fed by the original data sets are more successful. Also we can say,
##QDA works slightly better than LDA in classifying the dataset.

##5

table( data_train$y,train_estimated_org$class, dnn = c('Actual Group','Predicted Group'))
table( data_test$y,test_predicted_org_lda$class, dnn = c('Actual Group','Predicted Group'))
table( data_train$y,train_estimated_org_qda$class, dnn = c('Actual Group','Predicted Group'))
table( data_test$y,test_predicted_org_qda$class, dnn = c('Actual Group','Predicted Group'))
yy=as.data.frame(scored_test_data)
plot(yy, yy, pch = as.numeric(1:11), col = as.numeric(1:11))

#Based on the comparisons for train data between the actual values and estimated values, classes of
#2,6 and 9 have the most error rate.So, we can conclude that those class are the most difficult to 
#distinguish from others. Let's remove them.

eliminated_data_train=data_train[-which(data_train$y %in% c(1,2,6,11)),]
eliminated_data_train_vars=eliminated_data_train[-1]

eliminated_data_test=data_test[-which(data_test$y %in% c(1,2,6,11)),]
eliminated_data_test_vars=eliminated_data_test[-1]

##lda for train

lda3_train=lda(eliminated_data_train_vars,grouping=eliminated_data_train$y)
train_estimated_lda_elm=predict(lda3_train,eliminated_data_train_vars)
misclass_rate_train_lda_elm=sum(if_else(train_estimated_lda_elm$class!=eliminated_data_train$y,1,0))/length(eliminated_data_train$y)
misclass_rate_train_lda_elm

##lda for test

test_predicted_lda_elm=predict(lda3_train,eliminated_data_test_vars)
misclass_rate_test_lda_elm=sum(if_else(test_predicted_lda_elm$class!=eliminated_data_test$y,1,0))/length(eliminated_data_test$y)
misclass_rate_test_lda_elm

##qda for train

qda3_train=qda(eliminated_data_train_vars,grouping=eliminated_data_train$y)
train_estimated_qda_elm=predict(qda3_train,eliminated_data_train_vars)
misclass_rate_train_qda_elm=sum(if_else(train_estimated_qda_elm$class!=eliminated_data_train$y,1,0))/length(eliminated_data_train$y)
misclass_rate_train_qda_elm


##qda for test

test_predicted_qda_elm=predict(qda3_train,eliminated_data_test_vars)
misclass_rate_test_qda_elm=sum(if_else(test_predicted_qda_elm$class!=eliminated_data_test$y,1,0))/length(eliminated_data_test$y)
misclass_rate_test_qda_elm


lda3_train_cv=lda(data_train_vars,grouping=data_train$y,cv=T)
train_estimated_lda_cv=predict(lda3_train_cv,data_train_vars)
misclass_rate_train_lda_cv=sum(if_else(train_estimated_lda_cv$class!=data_train$y,1,0))/length(data_train$y)
misclass_rate_train_lda_cv
table( data_train$y,train_estimated_lda_cv$class, dnn = c('Actual Group','Predicted Group'))

##pca for eliminated data_train
pc2=prcomp(eliminated_data_train_vars,center =  T)
round(pc2$sdev,3)
round(pc2$center,3)
round(pc2$rotation,3)
summary(pc2)
##As seen in the table of importance of components, 6 principal component would be enough 
##to explaint 90% of the total (standardized) sample variance in the training data.

scored_train_data_eliminated=pc2$x[,1:6]
##pca for eliminated data_test

pc2_test=prcomp(eliminated_data_test_vars,center =  T)
summary(pc2_test)
scored_test_data_eliminated=pc2_test$x[,1:6]


##lda for train

lda4_train_pc=lda(scored_train_data_eliminated,grouping=eliminated_data_train$y)
train_estimated_lda_elm_pc=predict(lda4_train_pc,scored_train_data_eliminated)
misclass_rate_train_lda_elm_pc=sum(if_else(train_estimated_lda_elm_pc$class!=eliminated_data_train$y,1,0))/length(eliminated_data_train$y)
misclass_rate_train_lda_elm_pc

##lda for test

test_predicted_lda_elm_pc=predict(lda4_train_pc,scored_test_data_eliminated)
misclass_rate_test_lda_elm_pc=sum(if_else(test_predicted_lda_elm_pc$class!=eliminated_data_test$y,1,0))/length(eliminated_data_test$y)
misclass_rate_test_lda_elm_pc

##qda for train

qda4_train_pc=qda(scored_train_data_eliminated,grouping=eliminated_data_train$y)
train_estimated_qda_elm_pc=predict(qda4_train_pc,scored_train_data_eliminated)
misclass_rate_train_qda_elm_pc=sum(if_else(train_estimated_qda_elm_pc$class!=eliminated_data_train$y,1,0))/length(eliminated_data_train$y)
misclass_rate_train_qda_elm_pc


##qda for test

test_predicted_qda_elm_pc=predict(qda4_train_pc,scored_test_data_eliminated)
misclass_rate_test_qda_elm_pc=sum(if_else(test_predicted_qda_elm_pc$class!=eliminated_data_test$y,1,0))/length(eliminated_data_test$y)
misclass_rate_test_qda_elm_pc

##Comparison Table

##get all rates  together

rates_table_after_elimination=matrix(c(misclass_rate_train_lda_elm_pc,misclass_rate_test_lda_elm_pc,misclass_rate_train_qda_elm_pc,misclass_rate_test_qda_elm_pc,
                     misclass_rate_train_lda_elm,misclass_rate_test_lda_elm,misclass_rate_train_qda_elm,
                     misclass_rate_test_qda_elm),ncol = 2, byrow = T)

colnames(rates_table_after_elimination)=c("Training", "Testing")
rownames(rates_table_after_elimination)=c("LDA after PCA with eliminated data", "QDA after PCA with eliminated data","LDA with eliminated data", "QDA with eliminated data")

round(rates_table_after_elimination,3)


table( eliminated_data_train$y,train_estimated_lda_elm$class, dnn = c('Actual Group','Predicted Group'))
table( eliminated_data_test$y,test_predicted_lda_elm$class, dnn = c('Actual Group','Predicted Group'))
table( data_train$y,train_estimated_org_qda$class, dnn = c('Actual Group','Predicted Group'))
table( eliminated_data_test$y,test_predicted_qda_elm_pc$class, dnn = c('Actual Group','Predicted Group'))



#####
##6


simple_data=rbind(data_train[which(data_train$y %in% c(1,3,6,10)),],data_test[which(data_test$y  %in% c(1,3,6,10)),])

##hierarchical clustering

dist_euc = dist(as.matrix(simple_data[,2:11]), "euclidean")
hc_euc_c= hclust(dist_euc, "complete")
hc_euc_c$merge
hc_euc_c$height

plot(hc_euc_c,hang =-1, main="cluster for vowels",
     labels=as.factor(simple_data[,1]))
groups=cutree(hc_euc_c, k=4)
(groups)

rect.hclust(hc_euc_c, k=4,border="orange")

##K means clustering

# Determine number of clusters
wss <- (nrow(simple_data[,2:11])-1)*sum(apply(simple_data[,2:11],2,var))
for (i in 2:15) wss[i] <- sum(kmeans(simple_data[,2:11],
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

## K=4 seems feasible.

clus_km = kmeans(simple_data[,2:11], 4,iter.max = 30)
clus_km
plot(simple_data[,2:11], col=c(rep(1,90),rep(3,90),rep(6,90),rep(10,90)), main="true clusters")
plot(simple_data[,2:11], col = clus_km$cluster, main="clusters from k-means")
points(clus_km$centers, col = 1:4, pch = 8, cex=2)

cluster_kmeans=clus_km$cluster

##model clustering
install.packages("mclust")
library(mclust)

mc = Mclust(simple_data[,2:11],G = 4)
mc$parameters["mean"]
summary(mc)
cluster_model=mc$classification

##Among these results, the most interpretible result is the one belongs to the K-means clustering. 
##Hierarchical clustering is alos good at displayin the results but for this number of classes, it is hard
##distinguish the labels. Model clustering is the one taking the most time to reach out to a result.

cluster_hier=groups
cluster_kmeans
cluster_model

##compare the clustering models

comparison_hier=adjustedRandIndex(simple_data$y,cluster_hier)
comparison_kmeans=adjustedRandIndex(simple_data$y,cluster_kmeans)
comparison_model=adjustedRandIndex(simple_data$y,cluster_model)

##AdjustedRandIndex is a kind of metric indicating the accuracy of the clustering model in comparison with the true labels.
##Based on the results, the best model is the kmeans model with 30 iteration and 4 clusters with 0.40 AdjustedRandIndex.
##Hierarchical clustering follows the kmeans model with an AdjustedRandIndex of 0.24. The least successful model is the model clustering
##with an AdjustedRandIndex of 0.22. Considering its interpretable advantages, K means model works better in the data.

