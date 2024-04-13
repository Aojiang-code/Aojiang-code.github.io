# README

改进后199*39输入模型文件夹中：（如要调试，请务必先运行getAllData_2s.py）

Augmentation.py中引入几个数据增强函数

getAllData_2s.py获取所有2s长数据并存为npy文件

CRNN_2s.py中搭建输入为199*39的模型

trainAllData_2s.py读取npy数据训练模型

testAllData.py测试模型

cdds.h5是输入为199*39的模型

