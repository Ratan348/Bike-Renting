rm(list=ls())

#Set working Directory
setwd("D:/Project_2")

#Get Working Directory
getwd()

#Load the data from hard disk

data = read.csv("day.csv", na.strings = c("",NA))
class(data)

#Check dimension of the given dataset

dim(data)    # 731, 16

#Check summary of dataset
summary(data)
str(data)

#Get the column names
names(data)
length(names(data))


#Get number of unique values column wise

cnames = names(data)

for (i in cnames) {
   print(i)
   print(length(unique(data[,i])))
}

#Conversion of data type

num_names = c("instant","temp","atemp","hum","windspeed","casual","registered","cnt")


for (i in num_names) {
  data[,i] = as.numeric(data[,i])
}

cnames

fact_names = c("season","yr","mnth","holiday","weekday","workingday","weathersit")

for (i in fact_names) {
  data[,i] = as.factor(as.character(data[,i]))
}

data$dteday = as.Date(data$dteday)

str(data)

#______________________________________________________________________________________________________

#Check for missing values

sum(is.na(data))

###--- No Missing Value found ### No need to perform Missing value analysis

##____________________________________________________________________________________

##_______ OUTLIER ANALYSIS ______________________ 

library("ggplot2")

  for (i in 1:length(num_names)) {
    assign(paste0("gn",i), ggplot(aes_string(y = (num_names[i]), x = "cnt"),data = subset(data))+
             stat_boxplot(geom = "errorbar" , width = 0.5) + 
             geom_boxplot(outlier.color="red", fill = "grey" , outlier.shape=18, outlier.size=1, notch=FALSE) +
             theme(legend.position="bottom")+
             labs(y=num_names[i],x="cnt")+
             ggtitle(paste("Box Plot of responded",num_names[i])))
            print(i)   
            print(num_names[i])
  }

options(warn = -1)

#__ Drawing the Boxplot ________

gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,ncol=2)
gridExtra::grid.arrange(gn6,gn7,ncol = 2)


#___  Getting the outliers data element from each variable  _________

for (i in num_names) {
  print(i)
  val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
  print(length(val))
  print(val)
}

#Get the positions of outliers in respective variables

which(data$hum %in% boxplot.stats(data$hum)$out) #2 outliers
which(data$windspeed %in% boxplot.stats(data$windspeed)$out)  #13 outliers
which(data$casual %in% boxplot.stats(data$casual)$out)  #44 outliers

## 59 total outliers

###---- Remove all the rows which contains outliers

for (i in num_names) {
  val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
  data = data[which(!data[,i] %in% val),]
                
}

##55 rows got deleted

dim(data) # 676 , 16

#Confirm again if any missing value exists
sum(is.na(data))

#Confirm again if any outlier exists

for (i in num_names) {
  val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
}

length(val)

#Till here we have nor any missing value neither any outliers.

#now we have to consider on FEATURE SELECTION 

library("lsr")

#ANOVA TEST 

fact_names #Contains only the factor variables 
cnames

#Since we need to keep track of registered vehicles on each particular day. It becomes mandatory to keep the dteday variable
# i am not performing test on dteday variable for above stated reason

av_test = aov(cnt ~   season + yr + mnth + holiday + workingday + weekday  +weathersit , data = data)
summary(av_test)

#av_test2 = aov(cnt ~   holiday + workingday + season + yr + mnth + weekday  +weathersit , data = data)
#summary(av_test2)



#Correlation Chart
library(corrgram)
corrgram(na.omit(data))
dim(data)
corrgram(data[,num_names],order = F, upper.panel = panel.pie, text.panel = panel.txt, main = "correlation plot" )

fact_names
num_names

#Dimension reduction

data_selected = subset(data,select = -c(instant,casual,registered,temp))

names(data_selected)

write.csv(data_selected,"Clean_bike_data.csv", row.names = F)

#___ THE DATA HERE WE GOT IN DATA FRAME data_selected IS ALREADY NORMALIZED.

#__ MODEL DEVELOPMENT ___#

# Since we have to predict numerical values, the problem falls under domain of regression

library("rpart")
library("DataCombine")

rmExcept("data_selected")

train_index = sample(1:nrow(data_selected), 0.8* nrow(data_selected))

train = data_selected[train_index,]
test = data_selected[-train_index,]

##__  Decision tree regression model development

reg_model = rpart(cnt ~. , data = train, method = "anova")

#########  predict results for the test case dataset

predicted_data = predict(reg_model , test[,-12])

#names(test)

library("DMwR")

#install.packages("rattle")
#install.packages("mltools")
library("mltools")

#Error coefficient method used here is RMSLE Root Mean Square Log Error

rmsle( predicted_data,test[,12]) #0.25


library("rattle")
fancyRpartPlot(reg_model)

text(reg_model,pretty = 0)

#install.packages("pROC")
library("pROC")

##_______ RANDOM FOREST MODEL DEVELOPMENT  _________________# 

#Random Forest Model
library("randomForest")


rf_model = randomForest(cnt~., train, ntree = 100)
str(data_selected)

#as.Date(data_selected$dteday)


#Extract the rules generated as a result of random Forest model
library("inTrees")
rules_list = RF2List(rf_model)

#Extract rules from rules_list
rules = extractRules(rules_list, train[,-12])
rules[1:2,]

#Convert the rules in readable format
read_rules = presentRules(rules,colnames(train))
read_rules[1:2,]

#Determining the rule metric
rule_metric = getRuleMetric(rules, train[,-12], train$cnt)
rule_metric[1:2,]

#Prediction of the target variable data using the random Forest model
RF_prediction = predict(rf_model,test[,-12])
#regr.eval(test[,12], RF_prediction, stats = 'rmse')
rmsle( RF_prediction , test[,12]) #0.17


##_______ DEVELOPMENT OF LINEAR REGRESSION MODEL ____________


library("usdm")
LR_data_select = subset(data_selected , select = -(dteday))
colnames(LR_data_select)
vif(LR_data_select[,-12])
#vifcor(LR_data_select[,-12], th=0.9)

####Execute the linear regression model over the data
#lr_model = lm(cnt~. , data = train)

#summary(lr_model)

#colnames(test)

#Predict the data 
#LR_predict_data = predict(lr_model, test[,1:12])

#Calculate MAPE
#MAPE(test[,12], LR_predict_data)
#library("Metrics")
#rmsle(LR_predict_data,test[,12])

