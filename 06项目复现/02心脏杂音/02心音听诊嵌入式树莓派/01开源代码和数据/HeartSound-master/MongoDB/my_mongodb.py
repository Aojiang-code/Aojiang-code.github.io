from pymongo import MongoClient
import bson.binary
from io import StringIO,BytesIO
import os
import csv
client=MongoClient('127.0.0.1',27017)


PCGdatabase=client['heart_sound_DB']
col_test=PCGdatabase['test']

for i in os.listdir('/home/yinaihua/Desktop/深度学习/数据汇总/kaggle/set_a'):
    if i[0:6]=='normal':
        label1=i[0:6]
    elif i[0:8]=='extrahls':
        label1=i[0:8]
    elif i[0:6]=='murmur':
        label1=i[0:6]
    elif i[0:8]=='artifact':
        label1=i[0:8]
    else:
        label1='Aunlabelledtest'

    with open ('/home/yinaihua/Desktop/深度学习/数据汇总/kaggle/set_a/'+i,'rb') as audio:
        content = BytesIO(audio.read())
        col_test.save(dict(
        content= bson.binary.Binary(content.getvalue()),
        filename = i,
        label1=label1,##PASCAL中分类：normal,murmur,extrahs,artifact,Aunlabelledtest,extrasystole
        label2=False,##normal or abnormal
        posture=False,#姿势
        area=False,#听诊器放置区域
        posion=False,#听诊器听诊的位置
        period=False,#音频的时期（只针对密歇根大学）
        original_dataset='Kaggle',#原始数据集
        col_method='phone'#采集方式:stethoscope(听诊器),phone(手机)
      ))

for i in os.listdir('/home/yinaihua/Desktop/深度学习/数据汇总/kaggle/set_b'):
    if i[0:6]=='normal':
        label1=i[0:6]
    elif i[0:12]=='extrasystole':
        label1=i[0:12]
    elif i[0:6]=='murmur':
        label1=i[0:6]
    else:
        label1='Bunlabelledtest'

    with open ('/home/yinaihua/Desktop/深度学习/数据汇总/kaggle/set_b/'+i,'rb') as audio:
        content = BytesIO(audio.read())
        col_test.save(dict(
        content= bson.binary.Binary(content.getvalue()),
        filename = i,
        label1=label1,
        label2=False,
        posture=False,#姿势
        area=False,#听诊器放置区域
        posion=False,#听诊器听诊的位置
        period=False,#音频的时期（只针对密歇根大学）
        original_dataset='Kaggle',
        col_method='stethoscope'#采集方式:stethoscope(听诊器),phone(手机)
      ))

with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-a/RECORDS-normal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    normal_list = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#normal_list[i][0]
with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-a/RECORDS-abnormal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    abnormal_list = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#abnormal_list[i][0]
print(abnormal_list[1][0])
print(normal_list[0][0])
for i in os.listdir('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-a'):
        index=False
        if i[-3:]=='wav':
            for j in range(len(normal_list)):
                if i[0:5]==normal_list[j][0]:
                    label2='normal'
                    index=True
                    continue
            if index==False:
                for p in range(len(abnormal_list)):
                    if i[0:5] == abnormal_list[p][0]:
                        label2 = 'abnormal'
                            #continue
            # print(i)
            # print(label2)
            with open ('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-a/'+i,'rb') as audio:
                content = BytesIO(audio.read())
                col_test.save(dict(
                content= bson.binary.Binary(content.getvalue()),
                filename = i,
                label1=False,
                label2=label2,
                posture=False,#姿势
                area=False,#听诊器放置区域
                posion=False,#听诊器听诊的位置
                period=False,#音频的时期（只针对密歇根大学）
                original_dataset='pysionet',
                col_method=False#采集方式:stethoscope(听诊器),phone(手机)
              ))



with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-b/RECORDS-normal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    normal_list_b = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#normal_list[i][0]
with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-b/RECORDS-abnormal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    abnormal_list_b = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#abnormal_list[i][0]

for i in os.listdir('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-b'):
        index=False
        if i[-3:]=='wav':
            for j in range(len(normal_list_b)):
                if i[0:5]==normal_list_b[j][0]:
                    label2='normal'
                    index=True
                    continue
            if index==False:
                for p in range(len(abnormal_list_b)):
                    if i[0:5] == abnormal_list_b[p][0]:
                        label2 = 'abnormal'

            with open ('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-b/'+i,'rb') as audio:
                content = BytesIO(audio.read())
                col_test.save(dict(
                content= bson.binary.Binary(content.getvalue()),
                filename = i,
                label1=False,
                label2=label2,
                posture=False,#姿势
                area=False,#听诊器放置区域
                posion=False,#听诊器听诊的位置
                period=False,#音频的时期（只针对密歇根大学）
                original_dataset='pysionet',
                col_method=False#采集方式:stethoscope(听诊器),phone(手机)
              ))



with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-c/RECORDS-normal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    normal_list_c = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#normal_list[i][0]
with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-c/RECORDS-abnormal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    abnormal_list_c = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#abnormal_list[i][0]

for i in os.listdir('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-c'):
        index=False
        if i[-3:]=='wav':
            for j in range(len(normal_list_c)):
                if i[0:5]==normal_list_c[j][0]:
                    label2='normal'
                    index=True
                    continue
            if index==False:
                for p in range(len(abnormal_list_c)):
                    if i[0:5] == abnormal_list_c[p][0]:
                        label2 = 'abnormal'

            with open ('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-c/'+i,'rb') as audio:
                content = BytesIO(audio.read())
                col_test.save(dict(
                content= bson.binary.Binary(content.getvalue()),
                filename = i,
                label1=False,
                label2=label2,
                posture=False,#姿势
                area=False,#听诊器放置区域
                posion=False,#听诊器听诊的位置
                period=False,#音频的时期（只针对密歇根大学）
                original_dataset='pysionet',
                col_method=False#采集方式:stethoscope(听诊器),phone(手机)
              ))


with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-d/RECORDS-normal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    normal_list_d = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#normal_list[i][0]
with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-d/RECORDS-abnormal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    abnormal_list_d = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#abnormal_list[i][0]

for i in os.listdir('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-d'):
        index=False
        if i[-3:]=='wav':
            for j in range(len(normal_list_d)):
                if i[0:5]==normal_list_d[j][0]:
                    label2='normal'
                    index=True
                    continue
            if index==False:
                for p in range(len(abnormal_list_d)):
                    if i[0:5] == abnormal_list_d[p][0]:
                        label2 = 'abnormal'

            with open ('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-d/'+i,'rb') as audio:
                content = BytesIO(audio.read())
                col_test.save(dict(
                content= bson.binary.Binary(content.getvalue()),
                filename = i,
                label1=False,
                label2=label2,
                posture=False,#姿势
                area=False,#听诊器放置区域
                posion=False,#听诊器听诊的位置
                period=False,#音频的时期（只针对密歇根大学）
                original_dataset='pysionet',
                col_method=False#采集方式:stethoscope(听诊器),phone(手机)
              ))


with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-e/RECORDS-normal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    normal_list_e = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#normal_list[i][0]
with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-e/RECORDS-abnormal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    abnormal_list_e = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#abnormal_list[i][0]

for i in os.listdir('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-e'):
        index=False
        if i[-3:]=='wav':
            for j in range(len(normal_list_e)):
                if i[0:6]==normal_list_e[j][0]:
                    label2='normal'
                    index=True
                    continue
            if index==False:
                for p in range(len(abnormal_list_e)):
                    if i[0:6] == abnormal_list_e[p][0]:
                        label2 = 'abnormal'

            with open ('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-e/'+i,'rb') as audio:
                content = BytesIO(audio.read())
                col_test.save(dict(
                content= bson.binary.Binary(content.getvalue()),
                filename = i,
                label1=False,
                label2=label2,
                posture=False,#姿势
                area=False,#听诊器放置区域
                posion=False,#听诊器听诊的位置
                period=False,#音频的时期（只针对密歇根大学）
                original_dataset='pysionet',
                col_method=False#采集方式:stethoscope(听诊器),phone(手机)
              ))

with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-f/RECORDS-normal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    normal_list_f = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#normal_list[i][0]
with open('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-f/RECORDS-abnormal','r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    abnormal_list_f = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型#abnormal_list[i][0]

for i in os.listdir('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-f'):
        index=False
        if i[-3:]=='wav':
            for j in range(len(normal_list_f)):
                if i[0:5]==normal_list_f[j][0]:
                    label2='normal'
                    index=True
                    continue
            if index==False:
                for p in range(len(abnormal_list_f)):
                    if i[0:5] == abnormal_list_f[p][0]:
                        label2 = 'abnormal'

            with open ('/home/yinaihua/Desktop/深度学习/数据汇总/pysionet/training-f/'+i,'rb') as audio:
                content = BytesIO(audio.read())
                col_test.save(dict(
                content= bson.binary.Binary(content.getvalue()),
                filename = i,
                label1=False,
                label2=label2,
                posture=False,#姿势
                area=False,#听诊器放置区域
                posion=False,#听诊器听诊的位置
                period=False,#音频的时期（只针对密歇根大学）
                original_dataset='pysionet',
                col_method=False#采集方式:stethoscope(听诊器),phone(手机)
              ))

##注：第20个音频的命名修改成‘20 Pulm, Spilt S2 Transient, Supine, Diaph.mp3’
for i in os.listdir('/home/yinaihua/Desktop/深度学习/数据汇总/Michigan_Heart_Sounds'):
    if i!='.DS_Store':
        #print(i.split(','))
        area=i.split(',')[0].split(' ')[1]
        period=i.split(',')[1]
        posture=i.split(',')[2]
        posion=i.split(',')[3].split('.')[0]
        with open ('/home/yinaihua/Desktop/深度学习/数据汇总/Michigan_Heart_Sounds/'+i,'rb') as audio:
                        content = BytesIO(audio.read())
                        col_test.save(dict(
                        content= bson.binary.Binary(content.getvalue()),
                        filename = i,
                        label1=False,
                        label2=False,
                        posture=posture,#姿势
                        area=area,#听诊器放置区域
                        posion=posion,#听诊器听诊的位置
                        period=period,#音频的时期（只针对密歇根大学）
                        original_dataset='MHSDB',
                        col_method=False#采集方式:stethoscope(听诊器),phone(手机)
                      ))



for i in os.listdir('/home/yinaihua/Desktop/深度学习/数据汇总/shiraz-university-fetal-heart-sounds-database-1.0.0'):
    if i[-3:]=='wav':
        with open ('/home/yinaihua/Desktop/深度学习/数据汇总/shiraz-university-fetal-heart-sounds-database-1.0.0/'+i,'rb') as audio:
                        content = BytesIO(audio.read())
                        col_test.save(dict(
                        content= bson.binary.Binary(content.getvalue()),
                        filename = i,
                        label1=False,
                        label2=False,
                        posture=False,#姿势
                        area=False,#听诊器放置区域
                        posion=False,#听诊器听诊的位置
                        period=False,#音频的时期（只针对密歇根大学）
                        original_dataset='SUFHSDB',
                        col_method=False#采集方式:stethoscope(听诊器),phone(手机)
                      ))



# #####读取数据库音频数据，还原成mp4
# data = col_test.find_one({'filename':'a0014.wav'})
# out = open('/home/yinaihua/Desktop/深度学习/数据汇总/my.mp3','wb')
# out.write(data['content'])


