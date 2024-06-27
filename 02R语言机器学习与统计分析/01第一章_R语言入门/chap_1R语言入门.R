
########  第1章：R语言入门  ########









#### 1.1：R与Rstudo ####

## 使用R的帮助

?t.test()

??t.test()










#### 1.2：R的数据类型 ####


### 向量，矩阵，数组，数据框，Factor，列表，字符串，

### 向量

### 在R中向量可以通过多种方式生成，并且向量也可以包含多种模式，如一个向量的每个元素都可以是字符串，向量也可以是一个因子。



## 向量得生成
## 1:通过":"产生
A <- 1:5
A
## 2:通过c()函数生产
A <- c(1,3,5,7,9)
A
## 3:通过seq()函数指定步长生成
B = seq(from=2,to=10,by=2)
B
##  通过seq()函数指定数目生成
B = seq(from=2,to=10,length.out = 5)
B
## 4: 通过rep()函数生成具有重复元素得向量
C <- rep(1:2,5)
C
## 也可以分别指定每个元素得重复次数
C <- rep(1:2,c(2,3))
C


## 向量里面所包含得内容，不止可以为数字，向量中得元素也可以是字符串,也可以是TRUE或者FALISE等，同时向量也可以是一个因子向量


## 字符串向量
v_char <- c("A","B","C","D","E")
class(v_char)

## 逻辑向量
v_log <- rep(c(T,F),c(2,3))
v_log
class(v_log)

## 因子形式的向量
v_fac <- factor(x=c("A","B","C","A","C"),levels = c("A","B","C"),
                labels = c("A","B","C"))
v_fac
levels(v_fac)
## 因子向量重新排序
v_fac <- ordered(v_fac,c("C","B","A"))
v_fac
levels(v_fac)




## 向量的简单计算和获取指定位置的元素



## 向量如果进行四则运算，则会使用整个向量进行运算
vec <- seq(1,7)
## 进行除法运算
vec / 2
## 如果两个向量长度相等，则对应位置的元素进行运算
vec / (2*vec)
## 计算向量的累乘
cumprod(1:5)
## 计算向量的累加
cumsum(vec)


## 计算向量的长度
length(vec)

## 从向量中获取需要的元素,可以使用在中括号指定位置
vec[c(1,3,5,7,9)]

## 从向量中获取需要的元素,可以使用等长的逻辑向量
## 获取vec中能被3整除的元素
vec %% 3 == 0
vec[vec %% 3 == 0]


## 在［］中使用－号可以删除指定位置的元素
vec[c(-1:-5)]
## 给出向量的倒序
rev(vec)
## 给出符合条件元素所在的位置
which(vec %% 2 ==1)

## 数字向量转化为字符串向量
vec_num <- seq(from=2,to=10,by=2)
str(vec_num)
vec_char <- as.character(vec_num)
str(vec_char)
## 字符串向量转化为numeric
vec_num <- as.numeric(vec_char)
is.numeric(vec_num)

## 因子向量转化为字符串向量
vec_fac <- factor(c("A","B","C","A","C"))
str(vec_fac)
vec_fac2char <- as.character(vec_fac)
str(vec_fac2char)


## 查看向量的取值
unique(vec_fac2char)

## 查看每种取值的个数
table(vec_fac2char)


## 计算两个向量的并集
union(c(1:5),seq(2,10,2))

## 计算两个向量的差集
setdiff(c(1:5),seq(2,10,2))

## 计算两个向量的交集
intersect(c(1:5),seq(2,10,2))

##  序列1是否是序列2中的元素
is.element(c(1:5),seq(2,10,2))

## 也可以使用 %in% 
c(1:5) %in% seq(2,10,2)



### 矩阵

## 向量属于一维数组，在R中矩阵二维数组可以使用matrix()函数生成


## 1 矩阵的生成
## 使用向量生成矩阵
vec <- seq(1,12)
mat <- matrix(vec,nrow = 2)
mat

mat <- matrix(vec,ncol = 4)
mat

## 生成矩阵时优先排列行
mat <- matrix(vec,nrow = 2,ncol = 4,byrow = TRUE)
mat

## 2 使用cbind()按照列连接多个向量
mat <- cbind(c(1,3,5,7),c(2,4,6,8),c(1:4))
mat

##  使用rbind()按照行连接多个向量
mat <- rbind(c(1,3,5,7),c(2,4,6,8),c(1:4))
mat

## 使用diag生成单位矩阵
diag(4)

## 也可以指定对角元素的内容
diag(c(1:4))


## 为矩阵添加列名和行名
colnames(mat) <- c("A","B","C","D")
rownames(mat) <- c("a","b","c")
mat
## 查看矩阵的维度
dim(mat)

## 计算矩阵有多少行
nrow(mat)

## 计算矩阵有多少列
ncol(mat)

## 计算矩阵的长度，即所有元素的个数
length(mat)



### 获取矩阵中的元素



## 可以使用［行，列］来获取元素
mat <- rbind(c(1,3,5,7),c(2,4,6,8),c(1:4))
colnames(mat) <- c("A","B","C","D")
rownames(mat) <- c("a","b","c")
mat

## 获取矩阵第2行第3列位置的元素
mat[2,3]

## 获取矩阵第2列的元素
mat[,2]

## 获取矩阵第1行的元素
mat[1,]
## 获取矩阵第"A","C"列的元素
mat[,c("A","C")]

## 通过逻辑值获取需要的元素,获取矩阵中的偶数
mat %% 2 == 0
mat[mat %% 2 == 0]




### 矩阵的运算


mat <- matrix(c(1:12),nrow = 3)
mat

## 矩阵的转置
t(mat)

## 矩阵的行和
rowSums(mat)
apply(mat, 1, sum)

## 矩阵的列和
colSums(mat)
apply(mat,2,sum)
## 矩阵的行均值
rowMeans(mat)
## 矩阵的列均值
colMeans(mat)



## 矩阵与矩阵相乘
## 1 : 对应位置相乘
mat * mat

## 2 : 矩阵乘法
mat %*% t(mat)
mat2 <- mat %*% t(mat)
mat2

## 得到上三角矩阵
mat2[lower.tri(mat2)] <- 0
mat2

##  计算矩阵的行列式
mat3 <- cbind(1, 2:4, c(2,4,1))
mat3
det(mat3)

## 计算矩阵的对角线元素
diag(mat3)

## 矩阵的逆矩阵,求解ax=b,默认b=I(单位矩阵)
set.seed(123)
solve(matrix(runif(16),4,4))




### 高维数组


## 使用array生成3维数组
arr <- array(1:24,dim = c(3,4,2))
arr

## 获取数组中的元素
## 第2层数据中的第二行的内容
arr[2,,2]

arr[which(arr %% 5 == 0)]


dim(arr)
## 对数据的每层计算均值
apply(arr,3,mean)
## 对数据的第二维度，列数据求和
apply(arr, 2,sum)




### 数据框



## 生成数据框
df <- data.frame(id = c("A","B","C","D"),
                 age = c(10,15,9,12),
                 sex = c("F","M","M","F"),
                 score = c(17:20),
                 stringsAsFactors = FALSE)

head(df)

## 查看数据的汇总
summary(df)

## 将sex转化为因子
df$sex <- factor(df$sex)
## 查看数据的汇总
str(df)

## 通过矩阵生成数据框
mat <- rbind(c(1,3,5,7),c(2,4,6,8),c(1:4))
mat2df <- as.data.frame(mat)
colnames(mat2df) <- c("A","B","C","D")
mat2df




### 选取数据框中的元素

## 通过［］选择
df[,2]
## 通过$选择
df$id
## 获取id下得第3个元素
df$id[3]
## 通过变量的名称选择
df[c("id","age")]

## 通过行索引来选择指定的行
df[df$age > 10,]

## 可以通过with函数取消$的使用
with(df,age > 10)

## 使用逻辑值进行索引
df[df$id %in% c("B","D","F"),1:3]

## 为数据框添加新的变量
df$newvar <- df$score * 2




### 列表

### 列表可以容纳任何类型和结构的数据。


## 生成list
A <- factor(c("A","B","C","C","B"))
B <- matrix(seq(1:8),nrow = 2)
C <- "Type"
D <- data.frame(id = c("A","B","C","D"),
                 age = c(10,15,9,12))
## 使用A,B,C,D生成一个列表
mylist <- list(A,B,C,D)

mylist

str(mylist)
## 获取列表中的内容

## 1 使用［］
mylist[1]
mylist[[1]]
mylist[[2]][2,1:3]
mylist[[4]]$age[1:3]

## 给列表中的内容添加名字
names(mylist) <- c("one","two","three","four")
names(mylist) 
## 通过$来提取数据
mylist$one









#### 1.3：控制和函数 ####


### 条件执行


## 判断数值能否被3整除
num <- 9
if(num %% 3 == 0) print("数值可以被3整除") else print("数值不能被3整除")

## 使用 ifelse(test, yes, no)
num <- 10
ifelse(num %% 3 == 0,num,NA)


## switch 精确匹配
id = c("A","B","C","D")
switch(id[2],
       A = 10,
       B = 15,
       C = 9,
       D = 12)




### 循环语句

### for 循环和while循环



## 找出向量中的偶数和奇数
vec <- seq(1:20)
result1 <- result2 <- vector()
for (ii in 1:length(vec)) {
  ## 偶数
  if(vec[ii] %% 2 == 0){
    result1 <- c(result1,vec[ii])
  }else{
    result2 <- c(result2,vec[ii])
  }
  
}
result1
result2

## 通过break来跳出循环
## 从向量中找出5个偶数
set.seed(12)
vec <- sample(seq(1:100),40)
ii <- 1
result1 <- vector()
while(ii){
  ## 保存偶数
  if(vec[ii] %% 2 == 0) result1 <- c(result1,vec[ii])
  ## 满足条件，跳出循环
  if (length(result1) == 5){
    break
  }
  ii <- ii + 1
}
result1




### 函数

### R中的函数使用function来定义



##  1 二分法求方程根
##  编写所要求解单变量非线性方程的函数
solvefunction <- function(x){
  x^3-2*x^2-1
}

# 编写二分法求解方程
twosol <- function(a,b,ee=10^(-5)){
  #a：左边界，b：右边界，ee=10^(-5)：精度
  if (solvefunction(a)*solvefunction(b) > 0 | a > b)
    print("请更改边界")
  else
    while(abs(a-b)>=ee) {
      c <- (a+b)/2
      if (solvefunction(c) == 0)
        return(c)
      if (solvefunction(a)*solvefunction(c)<0)
        b <- c
      if (solvefunction(c)*solvefunction(b)<0)
        a <- c
      }
  return(c)
}

## 求解方程的根
answ <- twosol(0,3,ee=10^(-5))
answ









#### 1.4：R中的常用包 ####

### R中加载新的包的方法library()


# install.packages("dplyr")

library(dplyr)
library(readr)
library(readxl)
library(ggplot2)
library(VIM)
library(tidyr)
library(d3heatmap)
library(treemap)
library(GGally)
library(gridExtra)
library(gganimate)
library(stringr)
library(plotly)










