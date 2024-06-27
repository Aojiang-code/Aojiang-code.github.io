
######## 第7章: 时间序列分析 ########


## 统一设置ggplot2的绘图风格
library(ggplot2)
theme_set(theme_bw())












#### 7.1：时间序列的相关检验 ####


### 白噪声检验
## 如果时间序列数据没有通过白噪声检验，则说明该序列为随机数序列，则没有建立时间序列模型进行分析的必要。


### 单位根检验
## 用来判断时间序列是否为平稳序列


## 协整检验和Granger因果检验



library(dplyr)
library(tidyr)
library(zoo)
library(tseries)
## 模拟一组随机数据和非随机数据进行白噪声检验
AirPas <- read.csv("data/chap7/AirPassengers.csv",stringsAsFactors=FALSE)
AirPas$Month <- as.yearmon(AirPas$Month)
set.seed(123) ## 生成一组随机数据
AirPas$randdata <- round(rnorm(nrow(AirPas),mean = mean(AirPas$Passengers),
                               sd=50))
head(AirPas)




# ## 转化为时间序列数据
# Air_ts <- ts(AirPas$Passengers,start = c(1949,1),deltat = 1/12)
# plot.ts(Air_ts)
# ## 可视化两组时间序列数据
# Air_tsdf <- tsdf(Air_ts,colname = "month")
# head(Air_tsdf)
# set.seed(123)
# Air_tsdf$randdata <- round(rnorm(nrow(AirPas),mean = mean(AirPas$Passengers), 
#                                  sd=20))
# head(Air_tsdf)


## 可视化两组数据集
AirPas%>%gather(key = "dataclass",value = "Number",-Month)%>%
  ggplot(aes(x=Month,y=Number))+
  geom_line(aes(colour=dataclass,linetype = dataclass))+
  theme(legend.position = c(0.15,0.8))
## Ljung-Box 检验
Box.test(AirPas$Passengers,type ="Ljung-Box")
## p-value < 2.2e-16 说明该序列为非随机数据

Box.test(AirPas$randdata,type ="Ljung-Box")
## p-value = 0.6978 说明该序列为随机数据,即为白噪声


## 单位根检验，检验时间序列的平稳性
## ADF检验的零假设为有单位根

## 如果一个时间序列是平稳的，则通常只有随机成分，常用ARMA模型来预测未来的取值，
## 如果将一组数据经过d次差分后可以将不平稳的序列准化为平稳的序列，则称序列为d阶单整。

## 生成ARIMA(2,2,2)的时间序列数据，进行单位根检验演示
adfdata <- arima.sim(list(order = c(2,2,2),ar = c(0.8897, -0.4858),
                          d=2,ma = c(-0.2279, 0.2488)),n = 200)
diff1 <- diff(adfdata)
diff2 <- diff(diff1)
diff3 <- diff(diff2)
## 可视化4种曲线
par(mfrow=c(2,2))
plot(adfdata,main="ARIMA(2,2,2)")
plot(diff1,main="差分1次")
plot(diff2,main="差分2次")
plot(diff3,main="差分3次")
## 进行单位根检验
adf.test(adfdata)
##  p-value = 0.9684 说明数据不是平稳的
adf.test(diff1)
## p-value = 0.6465  说明一阶差分后数据不是平稳的
adf.test(diff2)
Box.test(diff2,type ="Ljung-Box")
## p-value = 0.01, 说明2阶差分后数据是平稳的,而且不是白噪声数据
## 可以对原始数据建立ARIMA(p,2,q)模型
Box.test(diff3,type ="Ljung-Box")
## 3阶差分后的数据已经是白噪声数据，没有分析的价值










#### 7.2：自回归移动平均模型 ####

## 如果一个时间序列数据是平稳的，那么可以使用ARMA模型来预测未来的数据


library(ggfortify)
library(gridExtra)
library(forecast)
# library(TSA)
## 读取数据
ARMAdata <- read.csv("data/chap7/ARMAdata.csv")
ARMAdata <- ts(ARMAdata$x)
plot.ts(ARMAdata)
autoplot(ARMAdata)+ggtitle("序列变化趋势")
## 白噪声检验
Box.test(ARMAdata,type ="Ljung-Box")
## p-value = 4.552e-15 ,说明不是白噪声

## 平稳性检验，单位根检验
adf.test(ARMAdata)
## p-value = 0.01,说明数据是平稳的

## 分析序列的自相关系数和偏自相关系数确定参数p和q
p1 <- autoplot(acf(ARMAdata,lag.max = 30,plot = F))+
  ggtitle("序列自相关图")
p2 <- autoplot(pacf(ARMAdata,lag.max = 30,plot = F))+
  ggtitle("序列偏自相关图")
gridExtra::grid.arrange(p1,p2,nrow=2)

## 偏自相关图3阶后截尾，可以认为p的取值为3左右，
## 自相关图5阶后截尾，可以认为q的取值为5左右，

## 通过观察自相关系数和偏自相关系数虽然可以确定p和q，但是这不是最好的方法，
## R提供了自动寻找序列合适的参数的函数
auto.arima(ARMAdata)
## 可以发现较好的ARMA模型为ARMA(2,1)

## 对数据建立ARMA(2,1)模型，并预测后面的数据
ARMAmod <- arima(ARMAdata,order = c(2,0,1))
summary(ARMAmod)
## 对拟合残差进行白噪声检验
Box.test(ARMAmod$residuals,type ="Ljung-Box")
## p-value = 0.7853 ,说明是白噪声

## 可视化模型未来的预测值

plot(forecast(ARMAmod,h=20))











#### 7.3：季节ARIMA模型 ####


## 对飞机乘客数据建立季节趋势的ARIMA模型进行预测未来的数据
AirPas <- read.csv("data/chap7/AirPassengers.csv",stringsAsFactors=FALSE)
## 处理为时间序列数据
AirPas$Month <- as.yearmon(AirPas$Month)
AirPas <- ts(AirPas$Passengers,start = AirPas$Month[1],frequency = 12)
head(AirPas)
## 可视化序列
autoplot(AirPas)+ggtitle("飞机乘客数量变化趋势")
## 将数据即切分位两个部分，一部分用于训练模型，一部分用于查看预测效果
AirPas_train <- window(AirPas,end=c(1958,12))
AirPas_test <- window(AirPas,star=c(1959,1))
adf.test(AirPas_train,k=12)
adf.test(diff(AirPas_train),k=12)
adf.test(diff(diff(AirPas_train)),k=12)
## 说明数据延迟12阶，原始数据和差分一次数据都有单位根，而差分两次后数据是平稳的

AirPasdiff2 <- diff(diff(AirPas_train))
## 分析序列的自相关系数和偏自相关系数分析参数p和q
p1 <- autoplot(acf(AirPasdiff2,lag.max = 40,plot = F))+
  ggtitle("序列自相关图")
p2 <- autoplot(pacf(AirPasdiff2,lag.max = 40,plot = F))+
  ggtitle("序列偏自相关图")
gridExtra::grid.arrange(p1,p2,nrow=2)

## 从自相关图和偏自相关图可以很明显的发现数据可能具有周期性，
## 不能很好的确定参数p和q的取值,根据图可知，序列可能具有年周期性
## 使用auto.arima()函数确定模型的参数
auto.arima(AirPas_train)
## 最好的模型为ARIMA(1,1,0)(0,1,0)[12] 

ARIMA <- arima(AirPas_train, c(1, 1, 0),
              seasonal = list(order = c(0, 1, 0),period = 12))
summary(ARIMA)
Box.test(ARIMA$residuals,type ="Ljung-Box")
## p-value = 0.9274,此时，模型的残差已经是白噪声数据，数据中的信息已经充分的提取出来了

## 可视化模型的预测值和这是值之间的差距

plot(forecast(ARIMA,h=24),shadecols="oldstyle")
points(AirPas_test,col = "red")
lines(AirPas_test,col = "red")











#### 7.4：多元时间序列ARIMAX模型 ####


library(readxl)
# library(TSA)
library(forecast)
## 读取数据
gasco2 <- read_excel("data/chap7/gas furnace data.xlsx")
head(gasco2)
GasRate <- ts(gasco2$GasRate)
CO2 <- ts(gasco2$`C02%`)

p1 <- autoplot(GasRate)
p2 <- autoplot(CO2)
gridExtra::grid.arrange(p1,p2,nrow=2)
##  切分位训练集和测试集
trainnum <- round(nrow(gasco2)*0.8)
GasRate_train <- window(GasRate,end = trainnum)
GasRate_test <- window(GasRate,start = trainnum+1)
CO2_train <- window(CO2,end = trainnum)
CO2_test <- window(CO2,start = trainnum+1)


## 自动寻找合适的p，q
auto.arima(y=CO2_train,xreg = GasRate_train)

# ARIMAXmod <- arimax(CO2_train,order = c(2,1,2),xreg = GasRate_train)
# ARIMAXmod <- arima(CO2_train,order = c(2,1,2),xreg = GasRate_train)
## 为了解决后面预测图可能会出现错误，将建立时间序列模型的函数arima()更换为forecast包中的Arima()函数
ARIMAXmod <- Arima(CO2_train,order = c(2,1,2),xreg = GasRate_train)
summary(ARIMAXmod)

## 可视化模型的预测值和真实值之间的差距
plot(forecast(ARIMAXmod,h=length(GasRate_test),xreg = GasRate_test))
lines(CO2_test,col="black")













#### 7.5：prophet预测时间序列 ####


library(prophet)
library(zoo)
## 读取数据并对数据重新命名
AirPas <- read.csv("data/chap7/AirPassengers.csv",stringsAsFactors=FALSE)
colnames(AirPas) <- c("ds","y")
AirPas$ds <- as.yearmon(AirPas$ds)
head(AirPas)
## 建立具有季节趋势的模型
model <- prophet(AirPas,growth = "linear",
                 yearly.seasonality = TRUE,weekly.seasonality = FALSE,
                 daily.seasonality = FALSE,seasonality.mode = "multiplicative")
## 预测后面两年的数据,并将预测结果可视化
future <- make_future_dataframe(model, periods = 24,freq = "month")
forecast <- predict(model, future)
plot(model, forecast)

## 可视化预测的组成部分，主要有线性趋势和季节趋势
prophet_plot_components(model, forecast)








