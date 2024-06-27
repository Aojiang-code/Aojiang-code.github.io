
######## 第3章: 数据可视化 ########









#### 3.1：R基础的数据可视化 ####


##  读取数据
iris <- read.csv("data/chap3/Iris.csv")
## 可视化散点图
par(pch = 17)
plot(iris$SepalLengthCm,iris$SepalWidthCm,
     type = "p",col = "red",main = "散点图",
     xlab = "SepalLengthCm",ylab = "SepalWidthCm")




generateRPointShapes<-function(){
  oldPar<-par()
  par(font=2, mar=c(0.5,0,0,0))
  y=rev(c(rep(0.5,9),rep(1,9), rep(1.5,9)))
  x=c(rep(1:9,3))
  x = x[1:26]
  y = y[1:26]
  plot(x, y, pch = 0:25, cex=1.5, ylim=c(0,3), xlim=c(1,9.5), 
       axes=FALSE, xlab="", ylab="", bg="blue")
  text(x, y, labels=0:25, pos=3)
  par(mar=oldPar$mar,font=oldPar$font )
}
generateRPointShapes()


cl <- colors()
length(cl); cl[1:20]




## 可视化多个图像窗口
par(mfrow=c(2,2))
layout(matrix(c(1,2,3,3),2,2,byrow = TRUE))
hist(iris$SepalLengthCm,breaks = 20,col = "lightblue",main = "直方图",xlab = "SepalLengthCm")
smoothScatter(iris$PetalLengthCm,iris$PetalWidthCm, nbin = 64,main = "散点图",xlab = "PetalLengthCm",ylab = "PetalWidthCm")
## 添加第3个图像
boxplot(SepalLengthCm~Species,data = iris,main = "箱线图",ylab = "SepalLengthCm")









#### 3.2：ggplot2系列包的可视化 ####

## ggplot2系列的包可以使用＋来绘制图像


library(ggplot2)
library(GGally)
library(gridExtra)
library(dplyr)

##  可视化简单的图像
## 散点图
p1 <- ggplot(iris,aes(x = PetalLengthCm,y = PetalWidthCm))+
  theme_bw(base_size = 9)+
  geom_point(aes(colour = Species))+
  labs(title = "散点图")

p1

## 小提琴图 
p2 <- ggplot(iris,aes(x = Species,y = SepalLengthCm))+
  theme_gray(base_size = 9)+
  geom_violin(aes(fill = Species),show.legend = F)+
  labs(title = "小提琴图")+
  theme(plot.title = element_text(hjust = 0.5))

p2

p3 <- ggplot(iris,aes(SepalWidthCm))+
  theme_minimal(base_size = 9)+
  geom_density(aes(colour = Species,fill = Species),alpha = 0.5)+
  labs(title = "密度曲线")+
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(0.8,0.8))
p3

p4 <- ggplot(iris,aes(x = SepalLengthCm,y = SepalWidthCm))+
  theme_classic(base_size = 9)+
  geom_point(shape = 17)+
  geom_density_2d(linemitre = 5)+
  theme(plot.title = element_text(hjust = 0.5))+
  ggtitle("二维密度曲线")

p4

## 将4副图放进一个图像中
grid.arrange(p1,p2,p3,p4,nrow = 2)




### 使用矩阵散点图分析变量两两之间的关系


ggscatmat(data = iris[,2:6],columns = 1:4,color = "Species",alpha = 0.8)+
  theme_bw(base_size = 10)+
  theme(plot.title = element_text(hjust = 0.5))+
  ggtitle("矩阵散点图")




### 使用平行坐标图分析每个样例在各个特征上的变化情况


ggparcoord(data = iris[,2:6],columns = 1:4,
           groupColumn = "Species",scale = "center")+
  theme_bw(base_size = 10)+
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "bottom")+
  ggtitle("平行坐标图")+labs(x = "")

## 平滑的平行坐标图
ggparcoord(data = iris[,2:6],columns = 1:4,
           groupColumn = "Species",scale = "globalminmax",
           splineFactor = 50,order = c(4,1,2,3))+
  theme_bw(base_size = 10)+
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "bottom")+
  ggtitle("平滑的平行坐标图")+labs(x = "")




### 分析奥利匹克120年的数据

## 热力图，直方图等


## 读取数据，数据融合
library(readr)
athlete_events <- read_csv("data/chap3/athlete_events.csv")
noc_regions <- read_csv("data/chap3/noc_regions.csv")
## 数据连接
athletedata <- inner_join(athlete_events,noc_regions[,1:2],by=c("NOC"="NOC"))

## 查看数据
summary(athletedata)
head(athletedata)

str(athletedata)

## 查看每个国家参与奥运会运动员人数
plotdata <- athletedata%>%group_by(region)%>%
  summarise(number=n())%>%
   arrange(desc(number))

 ## 可视化前40个人数多的国家的参与人数
 
ggplot(plotdata[1:30,],aes(x=reorder(region,number),y=number))+
  theme_bw()+
  geom_bar(aes(fill=number),stat = "identity",show.legend = F)+
  coord_flip()+
  scale_fill_gradient(low = "#56B1F7", high = "#132B43")+
  labs(x="地区",y="运动员人数",title="不同地区奥运会运动员人数")+
  theme(axis.text.x = element_text(vjust = 0.5),
        plot.title = element_text(hjust = 0.5))



### 热力图可视化数据


library(RColorBrewer)
## 人数最多的30个地区，不同年份运动员人数变化
region30 <- athletedata%>%group_by(region)%>%
  summarise(number=n())%>%
   arrange(desc(number))
region30 <- region30$region[1:30]

## 不同性别下的，可视化人数最多的15个地区，不同年份运动员人数变化
plotdata <- athletedata[athletedata$region %in%region30[1:15],]%>%
  group_by(region,Year,Sex)%>%
  summarise(number=n())

ggplot(data=plotdata, aes(x=Year,y=region)) + 
  theme_bw() +
  geom_tile(aes(fill = number),colour = "white")+
  scale_fill_gradientn(colours=rev(brewer.pal(10,"RdYlGn")))+
  scale_x_continuous(breaks=unique( plotdata$Year)) +
  theme(axis.text.x = element_text(angle = 90,vjust = 0.5))+
  facet_wrap(~Sex,nrow = 2)




### 使用表情包绘图


library(ggChernoff)
## 查看不同季节举办的的奥运会运动员人数变化

region6 <- c("USA","Germany","France" ,"UK","Russia","China")
index <- ((athletedata$region %in% region6)&(!is.na(athletedata$Medal))&(athletedata$Season=="Summer"))
plotdata <- athletedata[index,]
plotdata2 <- plotdata%>%group_by(Year,region)%>%
  summarise(Medalnum=n())

ggplot(plotdata2,aes(x=Year,y=Medalnum))+
  theme_bw()+
  geom_line()+
  geom_chernoff(fill = 'goldenrod1')+
  facet_wrap(~region,ncol = 2)+
  labs(x="举办时间",y="奖牌数")



## 可视化动画


library(gganimate)

## 可视化每个地区每年奖牌的获取情况
index <- (athletedata$region %in% region30[1:20]&(!is.na(athletedata$Medal)))
plotdata <- athletedata[index,]

plotdata2 <- plotdata%>%group_by(Year,region,Medal)%>%
  summarise(Medalnum = n())
head(plotdata2)

plotdata2$Year <- as.integer(plotdata2$Year)
ggplot(plotdata2,aes(x=region,y=Medalnum,fill=Medal))+
  theme_bw()+
  geom_bar(stat = "identity",position = "stack")+
  theme(axis.text.x = element_text(angle = 90,vjust = 0.5))+
  scale_fill_brewer(palette="RdYlGn")+
  transition_time(Year) +
  labs(title = 'Year: {frame_time}')


##截取其中两个图片
p2 <- ggplot(plotdata2[plotdata2$Year == 2000,],aes(x=region,y=Medalnum,fill=Medal))+
  theme_bw()+
  geom_bar(stat = "identity",position = "stack")+
  theme(axis.text.x = element_text(angle = 90,vjust = 0.5))+
  scale_fill_brewer(palette="RdYlGn")+
  labs(title = 'Year: 2000')

p2

p3 <- ggplot(plotdata2[plotdata2$Year == 1996,],aes(x=region,y=Medalnum,fill=Medal))+
  theme_bw()+
  geom_bar(stat = "identity",position = "stack")+
  theme(axis.text.x = element_text(angle = 90,vjust = 0.5))+
  scale_fill_brewer(palette="RdYlGn")+
  labs(title = 'Year: 1996')

p3












#### 3.3：其它数据可视化包 ####

### 树图可视化数据


library(treemap)
## 使用treemap 可视化数据
plotdata <- athletedata%>%
  group_by(region,Sex)%>%
  summarise(number=n())
##  计算奖牌数量
plotdata2 <- athletedata[!is.na(athletedata$Medal),]%>%
  group_by(region,Sex)%>%
  summarise(Medalnum=n())
## 合并数据
plotdata3 <- inner_join(plotdata2,plotdata,by=c("region", "Sex"))

treemap(plotdata3,index = c("Sex","region"),vSize = "number",
        vColor = "Medalnum",type="value",palette="RdYlGn",
        title = "不同性别下每个国家的运动员人数",
        title.legend = "奖牌数量")



### 绘制地图

## 可视化美国机场之间的联系


library(maps)
library(geosphere)
## 读取飞机航线的数据
usaairline <- read.csv("data/chap3/usaairline.csv")
airportusa <- read.csv("data/chap3/airportusa.csv")

head(airportusa)
head(usaairline)

map("state",col="palegreen", fill=TRUE, bg="lightblue", lwd=0.1)
# 添加起点的位置
points(x=airportusa$Longitude, y=airportusa$Latitude, pch=19, cex=0.4,col="tomato")

col.1 <- adjustcolor("orange", alpha=0.4)
## 添加边
for(i in 1:nrow(usaairline)) {
  node1 <- usaairline[i,c("Latitude.x","Longitude.x")]
  node2 <- usaairline[i,c("Latitude.y","Longitude.y")]
  arc <- gcIntermediate( c(node1$Longitude.x, node1$Latitude.x),
                         c(node2$Longitude.y, node2$Latitude.y),
                         n=1000, addStartEnd=TRUE )
  lines(arc, col=col.1, lwd=0.2)
}




## igraph包可视化社交网络图

library(igraph)
## 读取顶点和边的数据
vertexdata <- read.csv("data/chap3/vertex.csv")
edgedata <- read.csv("data/chap3/edge.csv")
## Country:国家，airportnumber：机场数量，vtype：节点的类型
head(vertexdata)
## Country.x,Country.y :连线的两个点，connectnumber：连接的数量， etype：边的类型
head(edgedata)

## 定义网络图
g <- graph_from_data_frame(edgedata,vertices = vertexdata,directed = TRUE)
## 添加边的宽度
E(g)$width <- log10(E(g)$connectnumber)

# 生成节点和边的颜色
colrs <- c("gray50", "tomato", "gold")
V(g)$color <- colrs[V(g)$vtype]
E(g)$color <- colrs[E(g)$etype]


# plot 4个图，2 rows, 2 columns，每个图使用不同的图像样式
par(mfrow=c(2,2), mar=c(0,0,0,0),cex = 1) 
plot(g, layout =  layout_in_circle(g),
     edge.arrow.size=0.4,
     vertex.size = 10*log10(V(g)$airportnumber), 
     vertex.label.cex = 0.6)

plot(g, layout =  layout_with_fr(g),
     edge.arrow.size=0.4,
     vertex.size = 10*log10(V(g)$airportnumber), 
     vertex.label.cex = 0.6)

plot(g, layout =  layout_on_sphere(g),
     edge.arrow.size=0.4,
     vertex.size = 10*log10(V(g)$airportnumber), 
     vertex.label.cex = 0.6)
plot(g, layout =  layout_randomly(g),
     edge.arrow.size=0.4,
     vertex.size = 10*log10(V(g)$airportnumber), 
     vertex.label.cex = 0.6)




## 可视化集合之间的关系——韦恩图

library(VennDiagram)
## VennDiagram包最多可以绘制5个集合的韦恩图，这里绘制4个数组的韦恩图
vcol <- c("red","blue","green","DeepPink")
T<-venn.diagram(list(First =c(1:30),
                     Second=seq(1,50,by = 2),
                     Third =seq(2,50,by = 2),
                     Four = c(20,70)),
                filename = NULL,lwd = 0.5,
                fill = vcol,alpha = 0.5,margin = 0.1)
grid.draw(T)





## 使用UpSetR包可视化多个集合的交集情况

library(UpSetR)
## 数据准备
one  <- 1:100
two <- seq(1,200,by = 2)
three <- seq(10,300,by = 5)
four <- seq(2,400,by = 4)
five <- seq(10,500,by = 10)
six <- seq(3,400,by = 3)
## 1: 将6个集合的并集计算出来，
all <- unique(c(one,two,three,four,five,six))
## 建立一个数据表格
plotdata <- data.frame(matrix(nrow = length(all),ncol = 7))
colnames(plotdata) <-c("element","one","two","three","four",
                       "five","six")
## 2:数据表第一列是6个集合的并集的所有元素
plotdata[,1] <- all
## 3:其它列中的对应行，如果包含那一行的元素，则取值为1，否则取值为0
for (i in 1:length(all)) {
  plotdata[i,2] <- ifelse(all[i] %in% one,1,0)
  plotdata[i,3] <- ifelse(all[i] %in% two,1,0)
  plotdata[i,4] <- ifelse(all[i] %in% three,1,0)
  plotdata[i,5] <- ifelse(all[i] %in% four,1,0)
  plotdata[i,6] <- ifelse(all[i] %in% five,1,0)
  plotdata[i,7] <- ifelse(all[i] %in% six,1,0)
}
## 查看数据
head(plotdata)


## 使用该数据表进行可视化
## 可视化6个集合的交并情况
upset(plotdata,
      sets = c("one","two","three","four","five","six"),
      nintersects = 40, ## 默认显示前40个交集
      order.by = "freq", ## 根据频数排序
      ## 设置主条形图
      matrix.color  = "black", ## 数据矩阵的颜色
      main.bar.color = "red",# 主要条形图的颜色
      ##设置集合条形图
      sets.bar.color = "tomato",
      ## 设置矩阵点图
      point.size  = 2.5,line.size = 0.5,
      ## 矩阵点图和条形图的比例
      mb.ratio = c(0.65, 0.35))












#### 3.4：R可视化3D图像 ####


library(plotly)
library(plot3D)

## 使用plot3D包绘制3D图像
x <- y <- seq(0,10,by = 0.5)
## 生成网格数据并计算Z
xy <- mesh(x,y)

z <- sin(xy$x) + cos(xy$y) + sin(xy$x) * cos(xy$y)
par(mfrow = c(1,2))
hist3D(x,y,z,phi = 45, theta = 45,space = 0.1,colkey = F,bty = "g")
surf3D(xy$x,xy$y,z,colkey = F,border = "black",bty = "b2")


## 使用plotly包绘制3D图像
plot_ly(x = xy$x, y = xy$y, z = z,showscale = FALSE)%>% 
  add_surface()






