---
title: "Human Activity Recognition_Machine_Learning"
author: "Niharika"
date: "Thursday, March 17, 2016"
output: word_document
---
#Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

#Data

The training data for this project are available here:

[Training Data]https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

[Testing Data]https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

#Reading Data for Testing and Training data set

Data is being downloaded in the file MLProjectData. Observing data, training data set has values such as "",NA's and #DIV/0, which need to be handeled before using this for analyzing and training data.


```r
if(!file.exists("./MLProjectData")){dir.create("./MLProjectData")}

fileurlTraining<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileurlTraining,destfile="./MLProjectData/pml-training.csv",method="curl")

fileurlTesting<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileurlTesting,destfile="./MLProjectData/pml-testing.csv",method="curl")

training<-read.table("./MLProjectData/pml-training.csv",sep=",",stringsAsFactors = FALSE,strip.white = TRUE,na.strings = c("NA","","#DIV/0!"),header=TRUE)
testing<-read.table("./MLProjectData/pml-testing.csv",sep=",",stringsAsFactors = FALSE,strip.white = TRUE,na.strings=c("NA","","#DIV/0!"),header=TRUE)
```

As there are so many columns with no values or NA therefore we can safely remove those columns to run our machine learning model. This reduced the number of columns from 160 to 53.

#Data Cleaning

```r
library(plyr);library(dplyr)
set.seed(33245)


training<-training[,-which(numcolwise(sum)(training) <0)]
training<-training[,!sapply(training,function(x) any(is.na(x) | x=="" ))]
training$X<-NULL
names(training)
```

```
##  [1] "user_name"            "raw_timestamp_part_1" "raw_timestamp_part_2"
##  [4] "cvtd_timestamp"       "new_window"           "roll_belt"           
##  [7] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
## [10] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_y"        
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [16] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

Created partitioning with 60:40 for training and validation data and trained with Random Forest method. With results the best tuned mtry value is 38 for this model with 53 coluns after cleansing the training data. The accuracy of the model is at 99.4%.
#Model Fitting

```r
library(caret);library(gbm);library(ggplot2);library(MASS);
inTrain = createDataPartition(training$classe, p = .60,list=FALSE)
train<-training[inTrain,]
validation<-training[-inTrain,]
traincontrol <- trainControl(method = "cv", number = 2)

fitRf <- train(train$classe~.,data=train, method="rf",trControl=traincontrol)
fitRf$results
```

```
##   mtry  Accuracy     Kappa   AccuracySD      KappaSD
## 1    2 0.9722317 0.9648566 0.0001134233 0.0001471649
## 2   38 0.9949046 0.9935554 0.0021628954 0.0027357302
## 3   74 0.9948199 0.9934481 0.0003615228 0.0004569071
```
With confusion matrix the accuracy of the validation is 99.8% which is very close to train data set using partitioning.

#Predict on Cross Validation

```r
predCV<-predict(fitRf,newdata=validation)
confusionMatrix(predCV, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    3    0    0    0
##          B    1 1515    4    0    0
##          C    0    0 1361    5    0
##          D    0    0    3 1278    0
##          E    0    0    0    3 1442
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9976          
##                  95% CI : (0.9962, 0.9985)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9969          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9980   0.9949   0.9938   1.0000
## Specificity            0.9995   0.9992   0.9992   0.9995   0.9995
## Pos Pred Value         0.9987   0.9967   0.9963   0.9977   0.9979
## Neg Pred Value         0.9998   0.9995   0.9989   0.9988   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1931   0.1735   0.1629   0.1838
## Detection Prevalence   0.2847   0.1937   0.1741   0.1633   0.1842
## Balanced Accuracy      0.9995   0.9986   0.9971   0.9967   0.9998
```

```r
fitRf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 38
## 
##         OOB estimate of  error rate: 0.23%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3345    3    0    0    0 0.0008960573
## B    4 2273    2    0    0 0.0026327337
## C    0    6 2044    4    0 0.0048685492
## D    0    0    2 1922    6 0.0041450777
## E    0    0    0    0 2165 0.0000000000
```

As seen by the result of the confusionmatrix, the model is good and efficient because it has an accuracy of 0.997 and very good sensitivity & specificity values on the validation dataset. (the lowest value is 0.993 for the sensitivity of the class D) 



Prediction on Test set

```r
predTest<-predict(fitRf, newdata=testing)
predTest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(predTest)
```


