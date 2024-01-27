
######## 第6章: 多元统计分析 ########









#### 6.1：主成分分析 ####

### 主成分分析在图像数据上的应用

## 使用主成分分析对图像数据降维，可视化图像的分布位置，

## 对图像的样本量可视化降维，可视化图像的特征图像


## 主成分分析

library(R.matlab)

ETHdata <- readMat("data/chap6/ETH_8class_object_8_big_classes_32_32_1024D.mat")

ETHims <- ETHdata$A / 255.0
dim(ETHims)

## 可视化部分样本的图像
set.seed(123)
index <- sample(ncol(ETHims),40)
par(mfrow = c(5,8),mai=c(0.05,0.05,0.05,0.05))
for(ii in seq_along(index)){
  im <- matrix(ETHims[,index[ii]],nrow=32,ncol = 32,byrow = TRUE)
  image(im,col = gray(seq(0, 1, length = 256)),xaxt= "n", yaxt= "n")
}



## 对每个特征进行主成分分析，找到样本在二维空间的位置


## 对数据主成分分析，可视化部分图像的位置
ETHlocal <- t(ETHims)
dim(ETHlocal)
ETHpca1 <- princomp(ETHlocal)

## 找到每个样本的前两个主成分的坐标
local <- ETHpca1$scores[,1:2]
dim(local)
## 为了防止遮挡，随机的挑选1000张图片进行可视化
set.seed(123)
index <- sample(nrow(local),1000)
localind <- local[index,]
x <- localind[,1]
y <- localind[,2]
ETHimsindex <- ETHims[,index]

##  设置图像的宽和高
width = 0.015*diff(range(x))
height = 0.03*diff(range(y))
## 可视化图像
plot(x,y, t="n",xlab = "PCA 1",ylab = "PCA 2")
for (ii in seq_along(localind[,1])){
  imii <- matrix(ETHimsindex[,ii],nrow=32,ncol = 32,byrow = TRUE)
  rasterImage(imii, xleft=x[ii] - 0.5*width,
              ybottom= y[ii] - 0.5*height,
              xright=x[ii] + 0.5*width, 
              ytop= y[ii] + 0.5*height, interpolate=FALSE)
  }





## 局部放大图
plot(x,y, t="n",xlim = c(-8,0),ylim = c(-2,4),
     xlab = "PCA 1",ylab = "PCA 2")
for (ii in seq_along(localind[,1])){
  imii <- matrix(ETHimsindex[,ii],nrow=32,ncol = 32,byrow = TRUE)
  rasterImage(imii, xleft=x[ii] - 0.5*width,
              ybottom= y[ii] - 0.5*height,
              xright=x[ii] + 0.5*width, 
              ytop= y[ii] + 0.5*height, interpolate=FALSE)
  }





## 对每个样本进行主成分分析，找到所有的样本的特征图像



library(psych)
library(ggplot2)
dim(ETHims)

## 随机挑选800张图像，为了使feature<样本数
set.seed(1234)
index <- sample(ncol(ETHims),800)
##  每类约有100张图像
table(ETHdata$labels[index])
ETHimsample<- ETHims[,index]
dim(ETHimsample)


## 可视化碎石图，选择合适的主成分数
parpca <- fa.parallel(ETHimsample,fa = "pc")
## 可视化碎石图的部分图像
pcanum <- 50
plotdata <- data.frame(x = 1:pcanum,pc.values = parpca$pc.values[1:pcanum])
ggplot(plotdata,aes(x = x,y = pc.values))+
  theme_bw()+
  geom_point(colour = "red")+geom_line(colour = "blue")+
  labs(x = "主成分个数")

## 主成分分析,提取前30个主成分
ETHcor <- cor(ETHimsample,ETHimsample)
dim(ETHcor)
ETHpca2 <- principal(ETHcor,nfactors = 30)
## 使用pca模型获取数据集的30个主成分
ETHpca_im<- predict.psych(ETHpca2,ETHimsample)
## 可视化这些主成分
par(mfrow = c(5,6),mai=c(0.05,0.05,0.05,0.05))
for(ii in seq_along(1:30)){
  im <- matrix(ETHpca_im[,ii],nrow=32,ncol = 32,byrow = TRUE)
  image(im,col = gray(seq(0, 1, length = 128)),xaxt= "n", yaxt= "n")
}










#### 6.2：聚类分析 ####


### 系统聚类


library(ggplot2)
library(gridExtra)
library(ggdendro)
library(cluster)
library(ggfortify)


## 系统聚类,鸢尾花数据集
iris <- read.csv("data/chap6/Iris.csv")
## 调整数据的类别标签
iris$Species <- stringr::str_replace(iris$Species,"Iris-","")
iris4 <- iris[,2:5]
str(iris4)
iris_scale <- scale(iris4)

## 系统聚类及可视化
hc1 <- hclust(dist(iris_scale),method = "ward.D2")
hc1$labels <- paste(iris$Species,1:150,sep = "-")
##  可视化结果
par(cex = 0.45)
plot(hc1,hang = -1)
rect.hclust(hc1, k=3, border="red") 

ggdendrogram(hc1, segments = T,rotate = F, theme_dendro = FALSE,size = 4)+
  theme_bw()+theme(axis.text.x = element_text(size = 5,angle = 90))





### k-means聚类


## k-means聚类,鸢尾花数据集
## 计算组内平方和  组间平方和
tot_withinss <- vector()
betweenss <- vector()
for(ii in 1:15){
  k1 <- kmeans(iris_scale,ii)
  tot_withinss[ii] <- k1$tot.withinss
  betweenss[ii] <- k1$betweenss
}

kmeanvalue <- data.frame(kk = 1:15,
                         tot_withinss = tot_withinss,
                         betweenss = betweenss)


p1 <- ggplot(kmeanvalue,aes(x = kk,y = tot_withinss))+
  theme_bw()+
  geom_point() + geom_line() +labs(y = "value") +
  ggtitle("Total within-cluster sum of squares")+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_x_continuous("kmean 聚类个数",kmeanvalue$kk)

p2 <- ggplot(kmeanvalue,aes(x = kk,y = betweenss))+
  theme_bw()+
  geom_point() +geom_line() +labs(y = "value") +
  ggtitle("The between-cluster sum of squares") +
  theme(plot.title = element_text(hjust = 0.5))+
  scale_x_continuous("kmean 聚类个数",kmeanvalue$kk)

grid.arrange(p1,p2,nrow=2)


## 可以发现选择聚类数目为3比较合适
## 随着聚类个数的增加，组内平方和在减少，组间平方和在增加，根据图和数据的背景可将数据聚类为三类。
set.seed(245)
k3 <- kmeans(iris_scale,3)
summary(k3)

k3
table(k3$cluster)

## 对聚类结果可视化
clusplot(iris_scale,k3$cluster,main = "kmean cluster number=3")

## 可视化轮廓图，表示聚类效果
sis1 <- silhouette(k3$cluster,dist(iris_scale,method = "euclidean"))


plot(sis1,main = "Iris kmean silhouette",
     col = c("red", "green", "blue"))




### 密度聚类


##  使用双月数据集和圆形数据集

library(fpc)
## 使用DBSCAN算法进行数据聚类
# 读取数据
moondata <- read.csv("data/chap6/moonsdatas.csv")
moondata$Y <- as.factor(moondata$Y)
str(moondata)
## 可视化数据的情况
ggplot(moondata,aes(x = X1,y = X2,shape = Y))+
  theme_bw()+geom_point()


# 用fpc包中的dbscan函数进行密度聚类

model1 <- dbscan(moondata[,1:2],eps=0.05,MinPts=5)
## 聚类结果
model1

table(model1$cluster)
## 可视化
moondata$clu <- model1$cluster
ggplot(moondata,aes(x = X1,y = X2,shape = as.factor(clu)))+
    theme_bw()+geom_point()+theme(legend.position = c(0.8,0.8))

## 可视化在不同的eps情况下，聚类的情况
eps <- c(0.05,0.06,0.25,0.3)
name <- c("one","two","three","four")
dbdata <- moondata[,1:2]
for (ii in 1:length(eps)) {
  modeli <- dbscan(dbdata[,1:2],eps=eps[ii],MinPts=5)
  dbdata[[name[ii]]] <- as.factor(modeli$cluster)

}

head(dbdata)

p1<- ggplot(dbdata,aes(x = X1,y = X2,shape = one,colour = one))+
  theme_bw(base_size = 8)+geom_point()+
  theme(legend.position = c(0.8,0.8))+ggtitle("eps=0.05,MinPts=5")
p2<- ggplot(dbdata,aes(x = X1,y = X2,shape = two,colour = two))+
  theme_bw(base_size = 8)+geom_point()+
  theme(legend.position = c(0.8,0.8))+ggtitle("eps=0.06,MinPts=5")
p3<- ggplot(dbdata,aes(x = X1,y = X2,shape = three,colour = three))+
  theme_bw(base_size = 8)+geom_point()+
  theme(legend.position = c(0.8,0.8))+ggtitle("eps=0.2,MinPts=5")
p4<- ggplot(dbdata,aes(x = X1,y = X2,shape = four,colour = four))+
  theme_bw(base_size = 8)+geom_point()+
  theme(legend.position = c(0.8,0.8))+ggtitle("eps=0.3,MinPts=5")

grid.arrange(p1,p2,p3,p4,nrow = 2)











#### 6.3：对应分析 ####


## 对应分析研究两个分类变量之间详细的依赖关系
library(ca)

## smoke数据集包含虚构公司中员工组（高级经理，初级经理，高级员工，初级员工和秘书）(senior managers, junior managers, senior employees, junior employees and secretaries)的吸烟习惯（无，轻，中，重）(none, light, medium and heavy)的频率。

data("smoke")

## 卡方检验判断两个变量是否独立
(result <- chisq.test(smoke))

 ## p-value = 0.1718 > 0.05, 说明不独立


## 使用马赛克图进行可视化数据

mosaicplot(smoke,main = "",color = c("red","blue","green","orange"))
## 数据中 JM，jE的中度吸烟者最多，SE中更多的none不吸烟，事实是否是这样，可以使用对应分析
 

## 对应分析分析
smca <- ca(smoke)
summary(smca)
plot(smca,main = "smoke data")




## 三维列联表的对应分析


library(factoextra)
library(FactoMineR)
## 1973年按入学和性别划分的伯克利大学研究生院申请人数最多的6个学院的汇总数据。
data("UCBAdmissions")
mca <- mjca(UCBAdmissions)
summary(mca)

plot(mca, mass = c(TRUE, TRUE),col = c("black","red","green","blue"),
     main = "三维列联表对应分析")











#### 6.4：典型相关分析 ####


library(ade4)
library(CCA)
library(candisc)

## 分析两组变量之间的相关性
## This data set gives the performances of 33 men's decathlon at the Olympic Games (1988).
data(olympic)
## is a data frame with 33 rows and 10 columns events of the decathlon: 100 meters (100), long jump (long), shotput (poid), high jump (haut), 400 meters (400), 110-meter hurdles (110), discus throw (disq), pole vault (perc), javelin (jave) and 1500 meters (1500).
olytab <- olympic$tab
summary(olytab)
head(olytab)
## 数据标准化

olytab_s <- as.data.frame(scale(olytab))

## 切分数据，分别为上肢相关和下肢相关的项目
## X: shotput (poid), discus throw (disq), javelin (jave), pole vault (perc) ;; arm
## Y: run100, run400, run1500, hurdle, long.jump, high.jump  ;; leg
xname <- c("poid","disq","jave","perc")
yname <- c("100","long","haut","400","110","1500")
olytab_sX <- olytab_s[,xname]
olytab_sY <- olytab_s[,yname]

##  典型相关分析
olycca <- candisc::cancor(olytab_sX, olytab_sY)
summary(olycca)

par(mfrow = c(2,2))
plot(olycca,which = 1)
plot(olycca,which = 2)
plot(olycca,which = 3)
plot(olycca,which = 4)

## 可视化典型相关分析的相关系数
olycca <- matcor(olytab_sX, olytab_sY)
img.matcor(olycca,type = 2)












#### 6.5：判别分析 ####


library(MASS)
library(klaR)
data("iris")
head(iris)
## 选择100个样本作为训练集，其余的作为测试集
set.seed(223)
index <- sample(nrow(iris),100)
iris_train <- iris[index,]
iris_test <- iris[-index,]

table(iris_train$Species)
table(iris_test$Species)

## 使用线性判别
irislda <- lda(Species~.,data = iris_train)
irislda
## 预测测试集
irisldapre <- predict(irislda,iris_test)
table(iris_test$Species,irisldapre$class)
## 可以发现只有两个样本预测错误
plot(irislda,abbrev = 1)

## 使用klaR包中的partimat函数探索可视化判别分析的效果
## 线性判别分析

partimat(Species~.,data = iris_train,method="lda",
         main = "线性判别")
## 二次判别分析

partimat(Species~.,data = iris_train,method="qda",
         main = "二次判别")


## 使用二次判别
irisqda <- qda(Species~.,data = iris_train)
irisqda

## 预测测试集
irisqdapre <- predict(irisqda,iris_test)
table(iris_test$Species,irisqdapre$class)

## 可以发现识别精度和线性判别分析的结果一样











#### 6.6：关联规则分析 #####


library(readr)
library(ggplot2)
library(dplyr)
library(arules)
library(arulesViz)
## 分析购物篮数据等

## 读取数据
groupdata <- read_csv("data/chap6/dataset_group.csv",col_names = F)
summary(groupdata)
## 数据一共包含约38种商品,共有1139条购买记录
unique(groupdata$X2)
length(unique(groupdata$X1))



## 查看每样商品在数据中出现的次数
items_unm <- groupdata %>%
  group_by(X2)%>%
  summarise(num = n())

ggplot(items_unm,aes(x = reorder(X2,num),y = num)) +
  theme_bw(base_size = 10) +
  geom_bar(stat = "identity",fill = "lightblue") +
  labs(x = "商品",y = "商品出现次数") + 
  coord_flip() +
  geom_text(aes(x = reorder(X2,num),y = num + 50,label = num),size = 3)




# 数据表转化为list
buy_data<- split(x=groupdata$X2,f=as.factor(groupdata$X1))
# 查看一共有多少个实例
sum(sapply(buy_data,length))

# 过滤掉每个购买记录中相同的实例
buy_data <- lapply(buy_data,unique)
sum(sapply(buy_data,length))
## 转化为（"transactions"）交易数据集
buy_data <- as(buy_data,"transactions")


## 1:可视化频繁项集
## 出现的频率大于0.25的项目
par(cex = 0.7)
itemFrequencyPlot(buy_data,support = 0.25,col = "lightblue",
                  xlab = "频繁项目",ylab = "项目频率",
                  main = "频率>0.25的项目")

## 可视化top20的项目
par(cex = 0.75)
itemFrequencyPlot(buy_data,top = 20,col = "lightblue",
                  xlab = "频繁项目",ylab = "项目频率",
                  main = "Top20的项目")


## 找到规则
myrule <- apriori(data = buy_data,
                  parameter = list(support = 0.25,
                                   confidence = 0.4,
                                   minlen = 1))
## 找到了57个规则
summary(myrule)


## 关联分析2，指定项目的右侧出现的项目，找到频繁的规则
## 这次主要挖掘指定右项集为"ice cream"
myrule2 <- apriori(buy_data,   #数据集
                   parameter = list(minlen =3,  # 频数项集长度
                                    maxlen = 8,# 项集的最大长度
                                    supp = 0.1, ## 支持度阈值
                                    conf = 0.45,  ## 置信度阈值
                                    target = "rules"),
                   ## 设定右边项集只能出现"ice creame","fruits"，
                   ## 左项集默认参数
                   appearance = list(rhs=c("ice cream"),
                                     default="lhs"))

summary(myrule2)
## 探索更详细的规则信息,将得到的规则按照提升度进行排序
myrule2_sortl <- sort(myrule2,by = "lift")
inspect(myrule2_sortl)



## 可视化获取得到的规则
plot(myrule2, method="graph")

plot(myrule2, method="grouped")








