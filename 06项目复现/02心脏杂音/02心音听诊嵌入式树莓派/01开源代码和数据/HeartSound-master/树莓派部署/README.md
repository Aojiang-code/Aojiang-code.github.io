# README

树莓派部署文件夹中：

Full.py是主程序，能够实现读取数据->LCD提醒读取完毕->LCD显示测试结果 的流程。

LCDnew.py是操作LCD屏的Python代码，在主程序Full.py中调用

PCF8591.py是操作数模转换器PCF8591的Python代码，在主程序Full.py中调用

ReadLite.py是读取tflite模型文件的实例程序

convert_lit.py是将h5模型转换为tflite模型的代码