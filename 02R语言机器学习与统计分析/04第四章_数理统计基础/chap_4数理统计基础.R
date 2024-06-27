
######## 第4章: 数理统计基础 ########






 

#### 4.1.随机数模拟 ####


library(ggplot2)
library(gridExtra)
## 模拟生成正态分布数据
nor1<- rnorm(300,mean = 0,sd = 1)
## Poisson Distribution泊松分布
pois <- rpois(300,lambda = 2)
## Uniform Distribution 均匀分布
unif1 <- runif(300,min = 0, max = 1)
## F分布
f1 <- rf(300,10,10)

## 数据可视化
p1<- ggplot()+theme_bw()+
  geom_density(aes(nor1),bw = 0.4,fill="red",alpha=0.4)+
  labs(x="",title = "正态分布")+
  theme(plot.title = element_text(hjust = 0.5))


p2 <- ggplot()+theme_bw()+
  geom_density(aes(pois),bw = 0.8,fill = "red",alpha = 0.4)+
  labs(x="",title = "泊松分布")+
  theme(plot.title = element_text(hjust = 0.5))

p3<- ggplot()+theme_bw()+
  geom_histogram(aes(unif1),bins = 15,fill = "red",alpha = 0.4)+
  labs(x="",title = "均匀分布")+
  theme(plot.title = element_text(hjust = 0.5))


p4<- ggplot()+theme_bw()+
  geom_density(aes(f1),bw = 0.8,fill = "red",alpha = 0.4)+
  labs(x="",title = "F分布")+
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(p1,p2,p3,p4,nrow=2)




## 根据随机数，估计数据分布的参数
library(MASS)
## Maximum-likelihood fitting of univariate distributions
fitdistr(nor1,densfun = "normal")


fitdistr(pois,densfun = "Poisson")



### 多元正态分布


library(MASS)
library(plotly)
## 生成Multivariate Normal Distribution多变量分布样本
set.seed(123)
sigma2 = matrix(c(10,3,3,4),2,2)
norm2d <- mvrnorm(n=800,mu=c(0,4),Sigma = sigma2)
norm2df <- as.data.frame(norm2d)
colnames(norm2df) <- c("x","y")
## 可视化2维正态分布数据
ggplot(norm2df,aes(x=x,y=y))+
  theme_bw()+
  geom_point(colour="red")+
  geom_density2d()+
  labs(title = "二元正态分布")+
  theme(plot.title = element_text(hjust = 0.5))

## 可视化2元正态分布的3维密度曲面图
## Two-Dimensional Kernel Density Estimation 二维核密度估计
kde <- kde2d(norm2df$x,norm2df$y,n = 50)
plot_ly(x = kde$x, y = kde$y, z = kde$z,type = "surface")

## 计算数据的方差矩阵
cov(norm2df)
## 计算均值
colMeans(norm2d)
apply(norm2d,2,mean)


## 估计二元正态分布的参数
library(tmvtnorm)
## 截断多元正态分布参数估计
mlefit1 <- mle.tmvnorm(norm2d,lower = c(-Inf,-Inf), upper = c(+Inf,+Inf))
summary(mlefit1)
mlefit1@coef











#### 4.2.假设检验 ####

## 数据分布检验
## 数据相关性检验
## t检验（均值检验等）
## 方差齐性检验
  


library(energy)
## 检验数据是否正态分布
## 模拟生成正态分布数据
set.seed(12)
nordata<- rnorm(500,mean = 2,sd = 5)
ggplot()+theme_bw()+
  geom_histogram(aes(nordata),stat = "density",bins = 40)+
  labs(x="",title = "正态分布")+
  theme(plot.title = element_text(hjust = 0.5))

## 1:QQ图经验法
#qqnorm(): 产生qq分布图
#qqline(): 添加一个参考线
par(pty="s")
qqnorm(nordata, pch = 1, frame = FALSE)
qqline(nordata, col = "steelblue", lwd = 2)
## 使用car包中的qqPlot函数
library(car)
par(pty="s")
qqPlot(nordata,distribution="norm")


## 2 ks检验
ks.test(x = nordata,"pnorm")
ks.test(x = nordata,"pnorm",mean = 2,sd=5)
## ks检验还可以检验两组数据是否具有相同的分布
## Poisson Distribution泊松分布
pois <- rpois(100,lambda = 2)
ks.test(x = nordata,y = pois)


## 2元正态分布数据的数据正态性检验
set.seed(1234)
sigma2 = matrix(c(1,0,0,1),2,2)
norm2d <- mvrnorm(n=100,mu=c(0,0),Sigma = sigma2)


## energy 包里的mvnorm.etest()基于E统计量做多元正态性检验
library(energy)
## H0:是多元正态分布
mvnorm.etest(norm2d,R = 199)


library(MVN)
##  For multivariate normality, both p-values of skewness and kurtosis statistics should be greater than 0.05.
par(pty="s")
result <- mvn(norm2d, mvnTest = "mardia",multivariatePlot = "qq")
## 多个变量正态性检验
result$multivariateNormality
## 每个变量的一维正态性检验
## Shapiro–Wilk test tests the null hypothesis that a sample x1, ..., xn 
## came from a normally distributed population.
result$univariateNormality
##   "mardia" 根据偏度和峰度的显著性来确定时否维多元正态分布，
## 如果是多元正态分布，则偏度核峰度的p值都应大于0.05


## 检验鸢尾花数据的4个特征是否为4元正态分布数据
Iris <- read.csv("data/chap4/Iris.csv",header = TRUE)
Iris <- scale(Iris[,2:5],center = T,scale = T)
par(pty="s")
irismvn <- mvn(Iris, mvnTest = "mardia",multivariatePlot = "qq")
irismvn$multivariateNormality
irismvn$univariateNormality

## 结果显示，4个变量不是4元正态分布



###  t检验和方差齐性检验



## t检验,数据的均值检验
## 单样本t检验，检验样本的均值是否为指定值
t1<- rnorm(100,mean = 0,sd = 4)
t.test(t1,mu = 0)

## 双样本t检验，检验样本的均值是否相等
t2<- rnorm(100,mean = 4,sd = 4)
t.test(t1,t2,mu = 0)
## 可以通过mu = num来检验两样本的均值差
t.test(t2,t1,mu = 4)

## 方差齐性检验
## F对于两个总体；数据服从正态分布。
## F Test to Compare Two Variances
var.test(t1,t2)

## Bartlett检验条件：对于多个总体；数据服从正态分布。
t3 <- rnorm(100,mean = 4,sd = 8)
bartlett.test(list(t1,t2,t3))

## Levene检验这一方法更为稳健，且不依赖总体分布，是方差齐性检验的首选方法。
## 它既可用于对两个总体方差进行齐性检验，也可用于对多个总体方差进行齐性检验
## 在car包中可以使用
library(car)
library(ggplot2)
testdata <- data.frame(x = c(t1,t2,t3),
                   group1 = c(rep(c("A","B","C"),c(100,100,100))))
leveneTest(x~group1,data = testdata)

ggplot(testdata,aes(x = group1,y = x))+theme_bw()+
  geom_violin(aes(fill = group1),alpha = 0.2)+
  geom_jitter(aes(colour = group1))+
  theme(legend.position = "none")+
  labs(y = "values")





### 数据相关性检验


Iris <- read.csv("data/chap4/Iris.csv",header = TRUE)
Iris <- Iris[,2:5]
## 检验鸢尾花数据的两个特征
cor.test(Iris$SepalLengthCm,Iris$SepalWidthCm)

## 如何快速检验4个变量两两之间的相关性是否显著
library(psych)
result <- corr.test(Iris,method="pearson")
## 相关系数
result$r
## 相关性检验的P值
result$p
## P值小于0.05，说明可以拒绝原假设，说明相关性不显著。


## Hmisc包的rcorr()也能完成相关性检验
library(Hmisc)
sper <- rcorr(as.matrix(Iris),type = "pearson")
## 相关系数
sper$r
## 相关性检验的P值
sper$P

## 为了方便观察相关性检验的结果，编写如下函数

comcor <- function(data,minre1 = 0.8,minre2 = -0.8,maxp = 0.05,type = "pearson"){
  ## 该函数用来计算相关系数，并且的高符合要求的组合
  ## data : 数据矩阵，每列为一个特征
  ## minre = 0.8 最小的相关系数
  ## maxp = 0.05 最大的相关系数显著性值
  library(Hmisc)
  n <- ncol(data)
  ## 计算相关系数
  sper <- rcorr(as.matrix(data),type = type)
  ## 生成相应的变量组合
  hang <- matrix(rownames(sper$r), nrow = n, ncol = n,byrow = FALSE)
  lie <- matrix(colnames(sper$r),nrow = n,ncol = n,byrow = TRUE)
  zuhe <- matrix(paste(hang,lie,sep = "~"),nrow = n)
  ## 对相应的数据取下三角形的数值
  lowrel <- sper$r[lower.tri(sper$r)]
  lowp <- sper$P[lower.tri(sper$P)]
  lowzuhe <- zuhe[lower.tri(zuhe)]
  result<-data.frame(zuhe=lowzuhe,r = lowrel,p = lowp)
  ## 找到相关系数中 r>=0.8,p<=0.05) 的组合
  index <- which((lowrel >=minre1 | lowrel <=minre2) & lowzuhe !=1 & lowp <= maxp)
  return(result[index,])
}
## 找到相关性显著的变量
comcor(Iris,minre1 = 0.8,minre2 = -0.8,maxp = 0.05,type = "pearson")











#### 4.3.方差分析 ####


##  单因素方差分析
  
  
##  双因素方差分析




## 单因素方差分析
Iris <- read.csv("data/chap4/Iris.csv",header = TRUE)
colnames(Iris)
## 比较鸢尾花数据集在SepalWidthCm上的均值差异
boxplot(SepalWidthCm~Species,Iris)
library(gplots)
## 可视化不同类数据的均值

plotmeans(SepalWidthCm~Species,Iris,col = "red",
          main = "")
## 进行单因素方差分析
# bartlett.test(SepalWidthCm~Species,Iris)
irisaov <- aov(SepalWidthCm~Species,Iris)
summary(irisaov)
## 检验结果说明了各组均值不等，可以进一步进行两两比较

## TukeyHSD()函数提供了对各组均值差异的成对检验
tky <- TukeyHSD(irisaov)
tky = as.data.frame(tky$Species)
tky$pair = rownames(tky)


# Plot pairwise TukeyHSD comparisons and color by significance level
ggplot(tky, aes(colour=cut(`p adj`, c(0, 0.01, 0.05, 1), 
                           label=c("p<0.01","p<0.05","Non-Sig")))) +
  theme_bw(base_size = 16)+
  geom_hline(yintercept=0, lty="11", colour="grey30",size = 1) +
  geom_errorbar(aes(pair, ymin=lwr, ymax=upr), width=0.2,size = 1) +
  geom_point(aes(pair, diff),size = 2) +
  labs(colour="")+
  theme(axis.text.x = element_text(size = 14))

## 双因素方差分析，不考虑交互作用

##  examines the effects of Vitamin C on tooth growth in Guinea Pigs (ToothGrowth)
## 每种动物通过两种递送方法之一，橙汁或抗坏血酸（一种维生素C形式，编码为VC）
## 接受三种剂量水平的维生素C（0.5,1和2mg /天）中的一种。
data(ToothGrowth)
## 将浓度数据转化为因子变量
ToothGrowth$dose <- factor(ToothGrowth$dose,levels = c(0.5, 1, 2),
                           labels = c("D0.5", "D1", "D2"))
str(ToothGrowth)

## 可视化数据
# 绘制多个分组的小提琴图
# Add error bars: mean_se
# (other values include: mean_sd, mean_ci, median_iqr, ....)
library("ggpubr")

ggviolin(ToothGrowth, x = "dose", y = "len", color = "supp",
         add = "dotplot",palette = c("red", "blue"))

## 双因素方差分析,不考虑交互作用
aov1 <- aov(len~dose+supp,data = ToothGrowth)
summary(aov1)

## 可视化方差分析交互图
ggline(ToothGrowth, x = "dose", y = "len", color = "supp",
       shape = "supp",add = "mean",palette = c("red", "blue"))


## 双因素方差分析,考虑交互作用
aov1 <- aov(len~dose*supp,data = ToothGrowth)
summary(aov1)











#### 4.4 列联表分析 ####



testdata<- read.csv("data/chap4/原料.csv")
rownames(testdata) <- testdata$X
testdata$X <- NULL
testdata

(result <- chisq.test(testdata))


## 计算出期望值
result$expected

## 使用马赛克图进行可视化

mosaicplot(testdata,main = "",color = TRUE)

library(vcd)
assocstats(as.matrix(testdata))




## 高维列联表分析


library(vcd)
library(MASS)

## 1973年按入学和性别划分的伯克利大学研究生院申请人数最多的6个学院的汇总数据。
data("UCBAdmissions")

UCBAdmissions
## A,B,C,D,E,F为6个学院
## 性别为男女
## Admit  为入学和被拒绝

## 使用马赛克图可视化数据
mosaic(~Admit+Dept+Gender,data=UCBAdmissions,shade = TRUE, legend = TRUE)

## 相互独立：判断Admit,Dept,Gender是否成对独立。
loglm(~Admit+Dept+Gender, data=UCBAdmissions)
## P值小于0说明，三者之间不独立


## 在给定Dept的情况下判断是否入学和性别是否独立
## 条件独立：Admit与Gender无关，给定Dept.
loglm(~Admit+Dept+Gender+Admit*Dept+Gender*Dept, data=UCBAdmissions)
## P值小于0说明，两者之间不独立

mosaic(~Admit+Gender,data=UCBAdmissions,shade = TRUE, legend = TRUE)
## 从马赛克图上可以看出入学的男性数量更高，而被拒绝的那女数量相当






