
######## 第2章: 数据管理和探索 ########










#### 2.1：数据获取 ####


### 读取各种各样的数据到R中



## 1:读取CSV文件
## 方式1
csvdata <- read.csv("data/chap2/Iris.csv",header = TRUE)
head(csvdata)
str(csvdata)
## 方式2
csvdata <- read.table("data/chap2/Iris.csv",header = TRUE,sep = ",")
head(csvdata)
str(csvdata)

## 方式3  c = character, i = integer, n = number, d = double, l = logical, D = date, ## T = date time, t = time, ? = guess
library(readr)
csvdata <- read_csv("data/chap2/Iris.csv",col_names = TRUE,
                    col_types = list("d","d","d","d","d","c"))
head(csvdata,2)
str(csvdata)

## 数据保存为csv
write_csv(csvdata,"data/chap2/IrisWrite_1.csv")

write.csv(csvdata,"data/chap2/IrisWrite_2.csv",quote = FALSE)

## 2: 读取excel数据
library(readxl)
exceldata <- read_excel("data/chap2/Iris.xlsx",sheet = "Iris")
str(exceldata,2)

## 读取spss数据
library(foreign)
spssdata <- read.spss("data/chap2/Iris_spss.sav",to.data.frame = TRUE)
head(spssdata,2)
str(spssdata)

## 读取spss数据
library(haven)
spssdata <- read_sav("data/chap2/Iris_spss.sav")
head(spssdata,2)
str(spssdata)
## 方法2 
spssdata <- read_spss("data/chap2/Iris_spss.sav")
head(spssdata,2)
## 读取SAS数据
sasdata <- read_sas("data/chap2/iris.sas7bdat")
head(sasdata,2)


## 读取stata数据
dtadata <- read_dta("data/chap2/iris.dta")
head(dtadata,2)
str(dtadata)

dtadata <- read_stata("data/chap2/iris.dta")
head(dtadata,2)
str(dtadata)

## 读取matlab数据文件
library(R.matlab)
matdata <- readMat("data/chap2/ABC.mat")
str(matdata)
head(matdata$A,2)

## 读取图片数据
## 读取png图像
library(png)
impng <- readPNG("data/chap2/Rlogo.png")
r <- nrow(impng) / ncol(impng) # image ratio
plot(c(0,1), c(0,r), type = "n", xlab = "", ylab = "", asp=1)
## 该行在Nootbook中不支持，但是在Console中运行正常
rasterImage(impng, 0, 0, 1, r) 

str(impng)

## load.image 可以读取多种格式的图像
library(imager)
imjpg <- load.image("data/chap2/image.jpg")
imdim <- dim(imjpg)
plot(imjpg,xlim = c(1,width(imjpg)),ylim = c(1,height(imjpg)))




## 通过爬虫获取数据

### 从HTML中获取链接、表格

library(XML)

## 获取网页中的链接,检查R官网都有哪些链接
fileURL <- "https://www.r-project.org/"
fileURLnew <- sub("https", "http", fileURL)
links <- getHTMLLinks(fileURLnew)
length(links)



## 从网页中读取数据表格，公牛队球员的数据
fileURL <- "http://www.stat-nba.com/team/CHI.html"
Tab <- readHTMLTable(fileURL)
length(Tab)
NBAmember <- Tab[[1]]
head(NBAmember)
## 在Windows系统下,输出结果中发现有的汉字显示为乱码,下面给出解决方法
## 先使用guess_encoding()函数检查它的可能编码
library(readr)
guess_encoding(colnames(NBAmember))
## 从输出中说明其很可能是UTF-8的编码，下面对编码进行修改
library(rvest)
colnames(NBAmember) <- repair_encoding(colnames(NBAmember),from = "UTF-8")
head(NBAmember)
## 输出结果中的乱码问题得到解决



## 使用rvest包获取网络数据
library(rvest)
library(stringr)
## 读取网页，获取电影的名称
top250 <- read_html("https://movie.douban.com/top250")
title <-top250 %>% html_nodes("span.title") %>% html_text()
head(title)
## 获取第一个名字
title <- title[is.na(str_match(title,"/"))]
head(title)
## 获取电影的评分
score <-top250 %>% html_nodes("span.rating_num") %>% html_text()
filmdf <- data.frame(title = title,score = as.numeric(score))

## 获取电影的主题
term <-top250 %>% html_nodes("span.inq") %>% html_text()
filmdf$term <- term
head(filmdf)











#### 2.2：数据缺失值处理 ####

### 很多时候数据不会是完整的，会存在有缺失值的情况，这时需要对缺失的数据进行处理。



##读取数据
myair <- read.csv("data/chap2/myairquality.csv")
dim(myair)
summary(myair)
## 1:检查数据是否存在缺失值
library(VIM)
## 可视化查看数据是否有缺失值
aggr(myair)



## complete.cases()输出样例是否包含缺失值
## 输出包含缺失值的样例
mynadata <- myair[!complete.cases(myair),]
dim(mynadata)
head(mynadata)
## matrixplot()可视化缺失值的详细情况
## 红色代表缺失数据的情况
matrixplot(mynadata)  
## 只保留没有缺失值的样例
newdata <- na.omit(myair)
dim(newdata)
head(newdata)

## 简单的方法
## 针对不同的情况和变量属性，可以使用不同的缺失值处理方法
## 1: 填补缺失值：
##    均值，中位数，众数等
## is.na()查看Ozone（臭氧）数据缺失值的位置
myair2 <- myair
## 使用均值填补缺失值
myair2$Ozone[is.na(myair$Ozone)] <- mean(myair$Ozone,na.rm = TRUE)

## 输出哪些位置有缺失值
which(is.na(myair$Solar.R))
## 使用中位数填补缺失值
myair2$Solar.R[which(is.na(myair$Solar.R))] <- median(myair2$Solar.R,na.rm = TRUE)


## 使用前面的或者后面的数据填补缺失值
library(zoo)
## 使用前面或者后面的值来填补缺失值
myair2$Wind <- na.locf(myair$Wind)
myair2$Temp <- na.locf(myair$Temp,fromLast = TRUE)
## 数据中月份数据可以使用前面和后面数据的平均值来填补
## 找到缺失值的位置
naindex <- which(is.na(myair$Month))
newnamonth <- round((myair$Month[naindex-1] + myair$Month[naindex+1]) / 2)
myair2$Month[naindex] <- newnamonth
## 日期数据根据数据情况可以使用前面的数值＋1
naindex <- which(is.na(myair$Day))
newnaday <- myair$Day[naindex-1] + 1
myair2$Day[naindex] <- newnaday


library(Hmisc)
## 使用众数填补缺失值Type变量
## 找出众数
table(myair$Type)
myair2$Type <- impute(myair$Type,"C")


## 观察处理后新数据集的缺失值情况
aggr(myair2)



### 复杂的数据缺失值处理方法

## 复杂的缺失值处理方法
colnames(myair)

## 考虑"Ozone"   "Solar.R" "Wind"    "Temp"之间有关系对四个特征进行缺失值处理
## 提取数据
myair <- myair[,c(1:4)]

## 使用KNN方法来填补缺失值
library(DMwR)
myair2 <- knnImputation(myair,k=5,scale = TRUE,meth = "weighAvg")

## 使用随机森林的方式填补缺失值
library(missForest)
myair2 <- missForest(myair,ntree = 50)
## 填补缺失值后的数据
myair2$ximp
## OOB误差
myair2$OOBerror



## 缺失值多重插补
library(mice)
## 进行链式方程的多元插补
## m:多重插补的数量
## method : 指定插补方法
## norm.predict : 线性回归预测；pmm：均值插补方法，rf: 随机森林方法
## norm:贝叶斯线性回归
impdta <- mice(myair,m = 5,method=c("norm.predict","pmm","rf","norm"))
summary(impdta)











#### 2.3 数据操作 ####


### 长宽数据变换,数据标准化处理,数据集切分

### 长宽数据变换


### 长宽数据变换
library(tidyr)


Iris <- read.csv("data/chap2/Iris.csv",header = TRUE)
head(Iris,2)
str(Iris)

## 宽数据转化为长数据1
Irislong = gather(Iris,key="varname",value="value",SepalLengthCm:PetalWidthCm)
head(Irislong,2)
str(Irislong)

## 长数据转化为宽数据1
IrisWidth <- spread(Irislong,key="varname",value="value")
head(IrisWidth,2)
str(IrisWidth)

library(reshape2)
## 宽数据转化为长数据2
Irislong = melt(Iris,id = c("Id","Species"),variable.name = "varname",
                value.name="value")
head(Irislong,2)
str(Irislong)

## 长数据转化为宽数据2
IrisWidth <- dcast(Irislong,Id+Species~varname)
head(IrisWidth,2)
str(IrisWidth)




## 数据汇总


library(dplyr)

Irisgroup <- Iris%>%
  ## 根据一个或多个变量分组
  group_by(Species)%>%
  ## 将多个值减少到单个值
  summarise(meanSL = mean(SepalLengthCm),
            medianSW = median(SepalWidthCm),
            sdPL = sd(PetalLengthCm),
            IQRPW = IQR(PetalWidthCm),
            num = n()) %>%
  ## 按变量排列行
  arrange(desc(sdPL))%>%
  ## 返回具有匹配条件的行
  filter(num==50)%>%
  ## 添加新的变量
  mutate(varPL = sdPL^2)

Irisgroup





## 数据标准化


Iris <- read.csv("data/chap2/Iris.csv",header = TRUE)
Iris <- Iris[2:5]
head(Iris,2)
str(Iris)



## 数据中心化：是指变量减去它的均值；
Irisc <- scale(Iris,center = TRUE, scale = FALSE)
apply(Irisc,2,range)


## 数据标准化：是指数值减去均值，再除以标准差；
## 数据标准化处理
Iriss <- scale(Iris,center = TRUE, scale = TRUE)
apply(Iriss,2,range)

## min-max标准化方法是对原始数据进行线性变换。
## 设minA和maxA分别为属性A的最小值和最大值，
## 将A的一个原始值x通过min-max标准化映射成在区间[0,1]中的值
## 新数据=（原数据-最小值）/（最大值-最小值）
minmax <- function(x){
  x <- (x-min(x))/(max(x)-min(x))
}

Iris01 <- apply(Iris,2,minmax)
apply(Iris01,2,range)


## 使用caret包进行处理
library(caret)
## preProcess得到的结果可以使用predict函数作用于新的数据集
## 而且还包括其他方法，如标准化 "scale", "range", 等
## 1 中心化
center <- preProcess(Iris,method = "center")
Irisc <- predict(center,Iris)
head(Irisc,2)
apply(Irisc,2,range)

## 2 标准化
scal <- preProcess(Iris,method = c("center","scale"))
Iriss <- predict(scal,Iris)
head(Iriss,2)
apply(Iriss,2,range)

## [0-1]化
minmax01 <- preProcess(Iris,method = "range",rangeBounds = c(0,1))
Iris01 <- predict(minmax01,Iris)
apply(Iris01,2,range)





## 数据集切分
Iris <- read.csv("data/chap2/Iris.csv",header = TRUE)
Iris <- Iris[2:6]
head(Iris,2)


## 数据集切分 1
num <- round(nrow(Iris)*0.7)
index <- sample(nrow(Iris),size = num)
Iris_train <- Iris[index,]
Iris_test <- Iris[-index,]
dim(Iris_train)
dim(Iris_test)



## 数据集切分2 使用carte包中的函数
## carte包中切分数据的输出为训练数据集在所有数据中的行位置
## 使用createDataPartition获取数据切分的索引
index = createDataPartition(Iris$Species,p=0.7)
Iris_train <- Iris[index$Resample1,]
Iris_test <- Iris[-index$Resample1,]
dim(Iris_train)
dim(Iris_test)

## 获取数据k折的行位置
index2 <- createFolds(Iris$Species,k = 3)
index2










#### 2.4：数据描述 ####

## 集中趋势，离散程度、偏度和峰度


iris <- read.csv("data/chap2/Iris.csv")

## 数据的集中趋势
## 均值
apply(iris[,c(2:5)],2,mean)
## 中位数
apply(iris[,c(2:5)],2,median)


## 离散程度
## 方差
apply(iris[,c(2:5)],2,var)
## 标准差
apply(iris[,c(2:5)],2,sd)
## 中位数绝对偏差
apply(iris[,c(2:5)],2,mad)

## 变异系数 标准差／均值,越大说明数据越分散
apply(iris[,c(2:5)],2,sd) / apply(iris[,c(2:5)],2,mean)


## 四分位数 和 极值
apply(iris[,c(2:5)],2,quantile)
apply(iris[,c(2:5)],2,fivenum)
apply(iris[,c(2:5)],2,range)
## 四分位数范围 IQR(x) = quantile(x, 3/4) - quantile(x, 1/4).
apply(iris[,c(2:5)],2,IQR)

## 偏度和峰度,可以使用moments库
library(moments)
apply(iris[,c(2:5)],2,skewness)

apply(iris[,c(2:5)],2,kurtosis)

library(ggplot2)
library(tidyr)
## 宽数据转化为长数据
irislong = gather(iris[,c(2:5)],key="varname",
                  value="value",SepalLengthCm:PetalWidthCm)
## 可视化数据的分布
ggplot(irislong,aes(colour = varname,linetype = varname))+
  theme_bw()+geom_density(aes(value),bw = 0.5)

## 可视化数据的分布
ggplot(irislong,aes(colour = varname,fill = varname,linetype = varname))+
  theme_bw()+geom_density(aes(value),bw = 0.5,alpha = 0.4)

plot(density(iris$SepalWidthCm))
skewness(iris$SepalWidthCm)












#### 2.5：数据相似性度量 ####


## 相关系数
cor(iris[,c(2:5)])

## 数据之间的距离
## 计算3种花之间的4个特征均值，然后计算他们之间的距离
## 数据准备
library(dplyr)
newdata <- iris%>%group_by(Species)%>%
  summarise(SepalLengthMean = mean(SepalLengthCm),
            SepalWidthMean = mean(SepalWidthCm),
            PetalLengthMean = mean(PetalLengthCm),
            PetalWidthMean = mean(PetalWidthCm))
rownames(newdata) <- newdata$Species
newdata$Species <- NULL
newdata
##  欧式距离等
dist(newdata,method = "euclidean",upper = T,diag = T)

## 曼哈顿距离
dist(newdata,method = "manhattan",upper = T,diag = T)

## maximum
dist(newdata,method = "maximum",upper = T,diag = T)


## canberra
dist(newdata,method = "canberra",upper = T,diag = T)


## minkowski
dist(newdata,method = "minkowski",upper = T,diag = T,p = 0.5)











