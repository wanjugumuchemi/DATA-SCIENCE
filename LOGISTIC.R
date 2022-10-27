glm_data <- read.csv("Dell/AML/LABS/LAB2-EDA/diabetes_dataset_2019.csv", stringsAsFactors = T)
head(glm_data)
summary(glm_data)
#Imputation using mode


glm_data$PhysicallyActive <- factor(glm_data$PhysicallyActive, levels = c( "less than half an hr" ,"more than half an hr" ,"one hr or more","none"), 
                          labels = c(1,2,3,1))
glm_data$Gender <- factor(glm_data$Gender, levels = c("Male", "Female"), labels = c(0,1))
glm_data$Family_Diabetes <- factor(glm_data$Family_Diabetes, levels = c("no", "yes"), labels = c(0,1))
glm_data$highBP <- factor(glm_data$highBP, levels = c("no", "yes"), labels = c(0,1))
glm_data$Smoking <- factor(glm_data$Smoking, levels = c("no", "yes"), labels = c(0,1))
glm_data$Alcohol <- factor(glm_data$Alcohol, levels = c("no", "yes"), labels = c(0,1))
glm_data$RegularMedicine <- factor(glm_data$RegularMedicine, levels = c("no","o", "yes"), labels = c(0,0,1))
glm_data$JunkFood <- factor(glm_data$JunkFood, levels = c("always","very often","often","occasionally"), labels = c(4,3,2,1))
glm_data$Stress <- factor(glm_data$Stress, levels = c("always","very often","sometimes","not at all"), labels = c(4,3,2,1))
glm_data$BPLevel <- factor(glm_data$BPLevel, levels = c("High","high","Low","low","normal"), labels = c(3,3,2,2,1))
glm_data$UrinationFreq <- factor(glm_data$UrinationFreq, levels = c("not much", "quite often"), labels = c(0,1))
summary(glm_data)
glm_data$Diabetic <- factor(glm_data$Diabetic, levels = c("no"," no","", "yes"), labels = c(0,0,0,1))
glm_data$Pdiabetes <- factor(glm_data$Pdiabetes, levels = c("no","0","", "yes"), labels = c(0,0,1,1))


glm_data$BMI = ifelse(is.na(glm_data$BMI),
                ave(glm_data$BMI, FUN = function(x) mean(x, na.rm = TRUE)),
                glm_data$BMI)

glm_data$Pregnancies = ifelse(is.na(glm_data$Pregnancies),
                   ave(glm_data$Pregnancies, FUN = function(x) mean(x, na.rm = TRUE)),
                   glm_data$Pregnancies)
glm_data$Age <- factor(glm_data$Age, levels=c('less than 40','40-49','50-59','60 or older'), 
                 labels = c(0,1,2,3))
structure(glm_data)
# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)

# Note that this is Stratified sampling method
split = sample.split(glm_data$Diabetic, SplitRatio = 0.7)
training_set = subset(glm_data, split == TRUE)
test_set = subset(glm_data, split == FALSE)
head(training_set)
summary(training_set)
head(test_set)
summary(test_set)

# Checking Class distribution
table(glm_data$Diabetic)
prop.table(table(glm_data$Diabetic))
prop.table(table(training_set$Diabetic))
prop.table(table(test_set$Diabetic))

# Building classifier
classifier = glm(Diabetic ~.,
                 training_set,
                 family = binomial)
summary(classifier)

#Finding which instance belongs to which class
# Predicting the Training set results
pred_prob_training <- predict(classifier, type = 'response', training_set)
pred_prob_training
pred_class_training = ifelse(pred_prob_training > 0.5, 1, 0)
pred_class_training

cbind(pred_prob_training, pred_class_training) #probability vs class

cm_training = table(training_set$Diabetic, pred_class_training)
cm_training

#Evaluation metrics using formula
#accuracy = (cm[1,1] + cm[2,2])/ (cm[1,1] + cm [1,2] + cm [2,1] +cm [2,2])
#accuracy

accuracy_training <- sum(diag(cm_training))/sum(cm_training)
accuracy_training

#Alternately
#cm using caret
confusionMatrix(cm_training)

# Predicting the Test set results
pred_prob_test <- predict(classifier, type = 'response', test_set)
pred_prob_test
pred_class_test = ifelse(pred_prob_test > 0.5, 1, 0)
pred_class_test
cm_test = table(test_set$Diabetic, pred_class_test)
cm_test

# Evaluation metrics using formula
# accuracy = (cm[1,1] + cm[2,2])/ (cm[1,1] + cm [1,2] + cm [2,1] +cm [2,2])
# accuracy

accuracy_test <- sum(diag(cm_test))/sum(cm_test)
accuracy_test

#Evaluation - compare accuracy on training set and test set
# Using formulae compute all other evaluation metrics

# ROC curve on test set
install.packages("ROCR")
library(ROCR)
install.packages("gplots")

# To draw ROC we need to predict the prob values. 

pred = prediction(pred_prob_test, test_set$Diabetic)
perf = performance(pred, "tpr", "fpr")
pred
perf
plot(perf, colorize = T)
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))

# Area Under Curve
auc <- as.numeric(performance(pred, "auc")@y.values)
auc <-  round(auc, 3)
auc




# Task 1: Closely examine cm on training and test sets and comment on model fitness

# Task 2: Variation 2 - Run the Expt by imputing the missing values. comment on this experiment

# Task 3:  Varaition 3 - Run the experiment after scaling the values

# Task 4: Compare results of all experiments. You may do this after appropriate tabulation to summarise the results

