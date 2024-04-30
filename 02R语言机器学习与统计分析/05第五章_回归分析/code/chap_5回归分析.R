
######## 第5章: 回归分析 ########








#### 5.1：一元回归模型 ####


library(ggplot2)
library(tidyr)

## 一元线型回归
onedata <- read.csv("data/chap5/simple linear regression.csv")
## 可视化数据
ggplot(onedata,aes(x = x,y = y))+
  theme_bw()+
  geom_point(colour = "red")+
  geom_smooth(method='lm',formula=y~x)

summary(lm(y~x,data = onedata))




## 多项式回归


# x = seq(-5,10,length.out = 100)
# y = 0.85*x^3-4.2*x^2-10.05*x+3.78
# y <- y + rnorm(100,mean = 20,sd = 20)
# index <- sample(1:100,100)
# datad <- data.frame(x = x[index],y = y[index])

# plot(datad$x,datad$y)

# write.csv(datad,"data/chap5/Polynomial regression.csv",quote = F,row.names = F)

## 读取数据
polydata <- read.csv("data/chap5/Polynomial regression.csv")

## 可视化数据，很显然数据不适合作线性回归
ggplot(polydata,aes(x=x,y = y))+geom_point()+theme_bw()

## 拟合3次多项式方程查看效果
lmp3 <- lm(y~poly(x,3),data = polydata)
summary(lmp3)
poly3 <- predict(lmp3,polydata)

polydata$poly3 <- poly3 
## 使用1，2，3，4次多项式回归拟合数据
lmp1 <- lm(y~poly(x,1),data = polydata)
poly1 <- predict(lmp1,polydata)
polydata$poly1 <- poly1
lmp2 <- lm(y~poly(x,2),data = polydata)
poly2 <- predict(lmp2,polydata)
polydata$poly2 <- poly2
lmp4 <- lm(y~poly(x,4),data = polydata)
poly4 <- predict(lmp4,polydata)
polydata$poly4 <- poly4
## 可视化模型各个多项式回归拟合的效果
polydatalong <- gather(polydata,key="model",value="value",
                       c("poly1","poly2","poly3","poly4"))
ggplot(polydatalong)+theme_bw()+geom_point(aes(x,y))+
  geom_line(aes(x = x,y = value,linetype = model,colour  = model),size = 0.8)+
  theme(legend.position = c(0.1,0.8))
  





## 贝叶斯估计多项式回归


## 生成随机数
set.seed(123)
n <- 100
x <- runif(n,-5,5)
beta <- c(5,-2,0,0.05,0,0.01)
lmdata <- data.frame(x = x,x2=x^2,x3=x^3,x4=x^4,x5=x^5)
X <- cbind(rep(1,n),lmdata)
y <- as.matrix(X) %*% beta + rnorm(n,0,5)
lmdata$y <- y
## 拟合3次多项式回归
lm_mod <- lm(y~.,data = lmdata)
coef(lm_mod)
## 该函数使用吉布斯采样从具有高斯误差的线性回归模型的后验分布生成回归系数的估计
library(BAS)
lm_bas <- bas.lm(y~.,data = lmdata,method = "MCMC",
                 prior = "BIC",modelprior = uniform(),
                 MCMC.iterations = 10000)

coef(lm_bas)

## 可视化两种模型的差异
lm_mod_pre <- predict(lm_mod,lmdata)
lm_bas_pre <- predict(lm_bas,lmdata)
poltdata <- data.frame(x=x,y=y)
poltdata$lm_mod_pre <- lm_mod_pre
poltdata$lm_bas_pre <- lm_bas_pre$fit
gather(poltdata,key="model",value="value",c("lm_mod_pre","lm_bas_pre"))%>%
  ggplot()+theme_bw()+geom_point(aes(x,y))+
  geom_line(aes(x = x,y = value,linetype = model,colour  = model),size = 0.8)+
  theme(legend.position = c(0.1,0.8))











#### 5.2：多元线性回归分析 ####


library(ggcorrplot)
library(tidyr)
library(GGally)

## 美国不同地区的平均房价预测数据集

## 读取数据
house <- read.csv("data/chap5/USA_Housing.csv")

head(house)

colnames(house)

## 数据探索
summary(house)

## 可视化数据的分布
## 可视化密度曲线
houselong <- gather(house,key="varname",value="value",1:6)
ggplot(houselong)+theme_bw()+
  geom_density(aes(value),fill = "red",alpha = 0.5)+
  facet_wrap(.~varname,scales = "free")+
  theme(axis.text.x = element_text(angle = 30))



## 数据可视化，相关系数热力图，分析变量之间的相关性
## 计算相关系数
house_cor <- cor(house)
ggcorrplot(house_cor,method = "square",lab = TRUE)+
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10))





## 多元线型回归


## 多元线型回归
lm1 <- lm(AvgPrice~.,data = house)

summary(lm1)

## 从输出的结果中可以发现AvgAreaNumberofBedrooms变量是不显著的
## 剔除该变量拟合新的回归模型
lm2 <- lm(AvgPrice~AvgAreaIncome+AvgAreaHouseAge+AvgAreaNumberRooms
          +AreaPopulation,data = house)

summary(lm2)

## 可视化回归模型的系数
ggcoef(lm2,exclude_intercept = T,vline_color = "red",
       errorbar_color = "blue",errorbar_height = 0.1)+
  theme_bw()

## 可视化回归模型的图像
par(mfrow = c(2,2))
plot(lm2)










#### 5.3：逐步回归进行变量选择 ####

## 如果回归模型中有多个自变量是不显著的，可以使用逐步回归


library(readxl)
library(GGally)
library(Metrics)
library(car)
ENB <- read_excel("data/chap5/ENB2012.xlsx")
head(ENB)

summary(ENB)

str(ENB)

## 数据探索

## 可视化矩阵散点图
ggscatmat(ENB)+theme(axis.text.x = element_text(angle = 60))




## 数据切分为训练集和测试集，训练集70%
set.seed(12)
index <- sample(nrow(ENB),round(nrow(ENB)*0.7))
trainEnb <- ENB[index,]
testENB <- ENB[-index,]

Enblm <- lm(Y1~.,data = trainEnb)
summary(Enblm)

## Coefficients: (1 not defined because of singularities)
## 因为奇异性问题，有一个变量没有计算系数


prelm <- predict(Enblm,testENB)
## Mean Squared Error
sprintf("均方根误差为: %f",mse(testENB$Y1,prelm))


## 判断模型的多重共线性问题
kappa(Enblm,exact=TRUE) #exact=TRUE表示精确计算条件数；
## 1.740667e+15 条件数很大，说明数据之间具有很强的多重共线性
## vif(Enblm)  会出错，提示模型中有aliased coefficients 
alias(Enblm)



## 逐步回归
Enbstep <- step(Enblm,direction = "both")
summary(Enbstep)

## 判断模型的多重共线性问题
kappa(Enbstep,exact=TRUE)
## 150955.4 条件减小了约10^10倍，说明数据之间的多重共线性问题得到了大大的缓解
vif(Enbstep)



## 计算在测试集上的预测误差
prestep <- predict(Enbstep,testENB)
## Mean Squared Error
sprintf("均方根误差为: %f",mse(testENB$Y1,prestep))

## 可视化测试集数据和原始数据

## 数据准备
index <- order(testENB$Y1)
X <- sort(index)
Y1 <- testENB$Y1[index]
lmpre <- prelm[index]
steppre <- prestep[index]

plotdata <- data.frame(X = X,Y1 = Y1,lmpre =lmpre,steppre = steppre)
head(plotdata)
plotdata <- gather(plotdata,key="model",value="value",c(-X,-Y1))

## 可视化
ggplot(plotdata,aes(x = X))+theme_bw()+
  geom_point(aes(y = Y1),colour = "red",alpha = 0.5)+
  geom_line(aes(y = value,linetype = model,colour = model),size = 0.6)+
  theme(legend.position = c(0.1,0.8))
  









#### 5.4：Logistic回归模型 ####

## 根据发出的声音判断性别

##数据来源：https://www.kaggle.com/primaryobjects/voicegender/home

## This database was created to identify a voice as male or female, based upon acoustic properties of the voice and speech. The dataset consists of 3,168 recorded voice samples, collected from male and female speakers. The voice samples are pre-processed by acoustic analysis in R using the seewave and tuneR packages, with an analyzed frequency range of 0hz-280hz (human vocal range).

## The following acoustic properties of each voice are measured and included within the CSV:

## meanfreq: mean frequency (in kHz)

## sd: standard deviation of frequency

## median: median frequency (in kHz)

## Q25: first quantile (in kHz)

## Q75: third quantile (in kHz)

## IQR: interquantile range (in kHz)

## skew: skewness (see note in specprop description)

## kurt: kurtosis (see note in specprop description)

## sp.ent: spectral entropy

## sfm: spectral flatness

## mode: mode frequency

## centroid: frequency centroid (see specprop)

## peakf: peak frequency (frequency with highest energy)

## meanfun: average of fundamental frequency measured across acoustic signal

## minfun: minimum fundamental frequency measured across acoustic signal

## maxfun: maximum fundamental frequency measured across acoustic signal

## meandom: average of dominant frequency measured across acoustic signal

## mindom: minimum of dominant frequency measured across acoustic signal

## maxdom: maximum of dominant frequency measured across acoustic signal

## dfrange: range of dominant frequency measured across acoustic signal

## modindx: modulation index. Calculated as the accumulated absolute difference
## between adjacent measurements of fundamental frequencies divided by the frequency range

## label: male or female


## 数据准备和探索
library(caret)
library(ROCR)
library(tidyr)
library(corrplot)
voice <- read.csv("data/chap5/voice.csv",stringsAsFactors = F)


head(voice)

summary(voice)

table(voice$label)
 
str(voice)

## 可视化相关系数
voice_cor <- cor(voice[,1:20])
ggcorrplot(voice_cor,method = "square")

corrplot.mixed(voice_cor,tl.col="black",tl.pos = "lt",
         tl.cex = 0.8,number.cex = 0.45)


## 可视化不同的特征在两种数据下的分布
plotdata <- gather(voice,key="variable",value="value",c(-label))
ggplot(plotdata,aes(fill = label))+
  theme_bw()+geom_density(aes(value),alpha = 0.5)+
  facet_wrap(~variable,scales = "free")






## 逻辑回归模型


voice$label <- factor(voice$label,levels = c("male","female"),labels = c(0,1))
##  数据集切分为70%训练集和30%测试集
index <- createDataPartition(voice$label,p = 0.7)
voicetrain <- voice[index$Resample1,]
voicetest <- voice[-index$Resample1,]

## 在训练集上训练模型
## 使用所有变量进行逻辑回归
voicelm <- glm(label~.,data = voicetrain,family = "binomial")

summary(voicelm)

## 对逻辑回归模型进行逐步回归，来筛选变量
voicelmstep <- step(voicelm,direction = "both")
summary(voicelmstep)

## 可视化在剔除变量过程中AIC的变化
stepanova <- voicelmstep$anova
stepanova$Step <- as.factor(stepanova$Step)
ggplot(stepanova,aes(x = reorder(Step,-AIC),y = AIC))+
  theme_bw(base_size = 12)+
  geom_point(colour = "red",size = 2)+
  geom_text(aes(y = AIC-1,label = round(AIC,2)))+
  theme(axis.text.x = element_text(angle = 30,size = 12))+
  labs(x = "删除的特征")



## 对比两个模型逐步回归前后的在测试集上预测的精度
voicelmpre <- predict(voicelm,voicetest,type = "response")
voicelmpre2 <- as.factor(ifelse(voicelmpre > 0.5,1,0))
voicesteppre <- predict(voicelmstep,voicetest,type = "response")
voicesteppre2 <- as.factor(ifelse(voicesteppre > 0.5,1,0))
sprintf("逻辑回归模型的精度为：%f",accuracy(voicetest$label,voicelmpre2))
sprintf("逐步逻辑回归模型的精度为：%f",accuracy(voicetest$label,voicesteppre2))

## 计算混淆矩阵
table(voicetest$label,voicelmpre2)
table(voicetest$label,voicesteppre2)

## 绘制出ROC曲线对比两种模型的效果
## 计算逻辑回归模型的ROC坐标
pr <- prediction(voicelmpre, voicetest$label)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
prfdf <- data.frame(x = prf@x.values[[1]],logitic = prf@y.values[[1]])
## 计算逐步逻辑回归模型的ROC坐标
pr <- prediction(voicesteppre, voicetest$label)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
## 添加新数据
prfdf$logiticstep <- prf@y.values[[1]]

prfdf2 <- gather(prfdf,key="model",value="y",2:3)

ggplot(prfdf2,aes(x= x,y = y,colour = model,linetype = model))+
  theme_bw()+geom_line(size = 1)+
  theme(aspect.ratio=1)+
  labs(x = "假正例率",y = "真正例率")











#### 5.5：泊松回归模型 ####

## 该数据一共有3个变量，


library(glmnet)
## 读取数据
poi_sim <- read.csv("data/chap5/poisson_sim.csv")
poi_sim <- poi_sim[,2:4]
poi_sim$prog <- factor(poi_sim$prog,levels=1:3, 
                       labels=c("一般", "学术", "职业"))
## 可视化获奖次数的直方图
hist(poi_sim$num_awards)

model <- glm(num_awards~.-1,data = poi_sim,family = poisson(link = "log"))
summary(model)

exp(coef(model))












## 5.6：Ridge和lasso回归 ####

## Ridge 回归


library(readxl)
library(caret)
library(glmnet)
library(corrplot)
library(Metrics)
library(ggplot2)

## 读取数据
diabete <- read.csv("data/chap5/diabetes.csv",sep = "\t")

## 可视化相关系数
diabete_cor <- cor(diabete)
corrplot.mixed(diabete_cor,tl.col="black",tl.pos = "d",number.cex = 0.8)


## 切分为训练集和测试集，70%训练，30%测试
set.seed(123)
d_index <- createDataPartition(diabete$Y,p = 0.7)
train_d <- diabete[d_index$Resample1,]
test_d <- diabete[-d_index$Resample1,]
## 数据标准化
scal <- preProcess(train_d,method = c("center","scale"))
train_ds <- predict(scal,train_d)
test_ds <- predict(scal,test_d)
## 查看标准化使用的均值和标准差
scal$mean

scal$std



## ridge 回归

## 在训练集上寻找合适的ridge参数
## 使用交叉验证来分析ridge回归合适的参数
lambdas <- seq(0,5, length.out = 200)
X <- as.matrix(train_ds[,1:10])
Y <- train_ds[,11]
set.seed(1245)
ridge_model <- cv.glmnet(X,Y,alpha = 0,lambda = lambdas,nfolds =3)
## 查看lambda对模型均方误差的影响
plot(ridge_model)

## 可视化ridge模型的回归系数的轨迹线
plot(ridge_model$glmnet.fit, "lambda", label = T)

## 找到使回归效果最好的lambda(均方误差最小)
ridge_min <- ridge_model$lambda.min
ridge_min



## 使用ridge_min 拟合ridge模型
ridge_best <- glmnet(X,Y,alpha = 0,lambda = ridge_min)

summary(ridge_best)

## 查看ridge回归的系数
coef(ridge_best)



## 预测在测试集上的效果
test_pre <- predict(ridge_best,as.matrix(test_ds[,1:10]))
sprintf("标准化后平均绝对误差为: %f",mae(test_ds$Y,test_pre))
## 将预测值逆标准化和原始数据进行比较
test_pre_o <- as.vector(test_pre[,1] * scal$std[11] + scal$mean[11])
sprintf("标准化前平均绝对误差为: %f",mae(test_d$Y,test_pre_o))





## lasso回归


## lasso线性回归

library(readxl)
library(caret)
library(glmnet)
library(corrplot)
library(Metrics)
library(ggplot2)


## 读取数据
diabete <- read.csv("data/chap5/diabetes.csv",sep = "\t")

## 可视化相关系数
diabete_cor <- cor(diabete)
corrplot.mixed(diabete_cor,tl.col="black",tl.pos = "d",number.cex = 0.8)


## 数据集切分和数据标准化

## 切分为训练集和测试集，70%训练，30%测试
set.seed(123)
d_index <- createDataPartition(diabete$Y,p = 0.7)
train_d <- diabete[d_index$Resample1,]
test_d <- diabete[-d_index$Resample1,]
## 数据标准化
scal <- preProcess(train_d,method = c("center","scale"))
train_ds <- predict(scal,train_d)
test_ds <- predict(scal,test_d)
## 查看标准化使用的均值和标准差
scal$mean

scal$std



## 寻找合适的lasso参数


## 在训练集上寻找合适的lasso参数
## 使用交叉验证来分析lasso回归合适的参数
lambdas <- seq(0,2, length.out = 100)
X <- as.matrix(train_ds[,1:10])
Y <- train_ds[,11]
set.seed(1245)
lasso_model <- cv.glmnet(X,Y,alpha = 1,lambda = lambdas,nfolds =3)
## 查看lambda对模型均方误差的影响
plot(lasso_model)

## 可视化lasso模型的回归系数的轨迹线
plot(lasso_model$glmnet.fit, "lambda", label = T)
## 找到使回归效果最好的lambda(均方误差最小)
lasso_min <- lasso_model$lambda.min
lasso_min

## 使得误差在最小值的1个标准误差范围内的lambda的最大值。
lasso_lse <- lasso_model$lambda.1se
lasso_lse


## 训练合适的模型并预测


## 使用lasso_min 拟合lasso模型
lasso_best <- glmnet(X,Y,alpha = 1,lambda = lasso_min)

summary(lasso_best)

## 查看lasso回归的系数
coef(lasso_best)


## 预测在测试集上的效果
test_pre <- predict(lasso_best,as.matrix(test_ds[,1:10]))
sprintf("标准化后平均绝对误差为: %f",mae(test_ds$Y,test_pre))
## 将预测值逆标准化和原始数据进行比较
test_pre_o <- as.vector(test_pre[,1] * scal$std[11] + scal$mean[11])
sprintf("标准化前平均绝对误差为: %f",mae(test_d$Y,test_pre_o))






## lasso广义回归


## 读取数据
voice <- read.csv("data/chap5/voice.csv",stringsAsFactors = F)
voice$label <- factor(voice$label,levels = c("male","female"),labels = c(0,1))

set.seed(123)
##  数据集切分为70%训练集和30%测试集
index <- createDataPartition(voice$label,p = 0.7)
voicetrain <- voice[index$Resample1,]
voicetest <- voice[-index$Resample1,]

## 寻找合适的参数
#lambdas <- seq(1,1000, length.out = 100)
lambdas <- c(0.000001,0.00001,0.0001,0.001,0.01,0.1,0.5,1,2)
X <- as.matrix(voicetrain[,1:20])
Y <- voicetrain$label
lasso_model <- cv.glmnet(X,Y,alpha = 1,lambda = lambdas,nfolds =3,
                         family = "binomial",type.measure = "class")
## 查看lambda对模型均方误差的影响
plot(lasso_model)


plot(lasso_model$glmnet.fit, "lambda", label = T)

## 找到使回归效果最好的lambda(均方误差最小)
lasso_min <- lasso_model$lambda.min
lasso_min


## lasso广义回归
## 使用lasso_min 拟合lasso模型
lasso_best <- glmnet(X,Y,alpha = 1,lambda = lasso_min,
                     family = "binomial")

summary(lasso_best)

## 查看lasso回归的系数
coef(lasso_best)

## 预测在测试集上的效果
test_pre <- predict(lasso_best,as.matrix(voicetest[,1:20]))
test_pre <- as.factor(ifelse(test_pre > 0.5,1,0))
sprintf("在测试集上的预测精度为: %f",accuracy(voicetest$label,test_pre))

## 通过调整分类阈值，分析模型的精度
thresh <- seq(0.05,0.95,by = 0.05)
acc <- thresh
for (ii in 1:length(thresh)){
  test_pre <- predict(lasso_best,as.matrix(voicetest[,1:20]))
  test_pre <- as.factor(ifelse(test_pre > thresh[ii],1,0))
  acc[ii] <- accuracy(voicetest$label,test_pre)
}
## 可视化变化曲线
plotdata <- data.frame(thresh = thresh,acc = acc)
ggplot(plotdata,aes(x = thresh,y = acc))+
  theme_bw()+
  geom_point()+geom_line()+ylim(c(0.95,1))+
  scale_x_continuous("分类阈值",thresh)+
  labs(y = "模型精度",title = "Lasso广义回归精度")+
  theme(plot.title = element_text(hjust = 0.5))
  





