# [01Chronic_KIdney_Disease_dataset](06项目复现\04kaggle\02数据集\02肾脏疾病数据集\01Chronic_KIdney_Disease_dataset\README.md)


> 网址： [Chronic KIdney Disease dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease)

* 2017年创建
* 315人投票
* 36,427 downloads

## 相关代码(按vote排序)

### [01Chronic_Kidney_Disease_Prediction_(98%_Accuracy)](06项目复现\04kaggle\02数据集\02肾脏疾病数据集\01Chronic_KIdney_Disease_dataset\01Chronic_Kidney_Disease_Prediction_0.98_Accuracy\README.md)


> 网址：[Chronic Kidney Disease Prediction (98% Accuracy)](https://www.kaggle.com/code/niteshyadav3103/chronic-kidney-disease-prediction-98-accuracy)

* 2021年发布
* 37,139次浏览
* 162人认可
* 937次复现



### [06Chronic Kidney Disease Prediction + EDA](06项目复现\04kaggle\02数据集\02肾脏疾病数据集\01Chronic_KIdney_Disease_dataset\06Chronic_Kidney_Disease_Prediction_EDA\README.md)


> 网址：[Chronic Kidney Disease Prediction + EDA](https://www.kaggle.com/code/equinxx/chronic-kidney-disease-prediction-eda)

* 2022年发布
* 6,021次浏览
* 42人认可
* 103次复现


### 








### 




## About Dataset

> Data has 25 feattures which may predict a patient with chronic kidney disease

### Context

First, I am new to ML, and just in case I slip up, apologies in advance!!
So, I am doing an online ML course and this is an assignment where we are supposed to practice scikit-learn's PCA routine. Since the course has been ARCHIVED - which means the discussion posts are not answered!! - hence my posting of the problem here.

What better way to learn than to get so many experts giving me feedback … right?

### Content

The data was taken over a 2-month period in India with 25 features ( eg, red blood cell count, white blood cell count, etc). The target is the 'classification', which is either 'ckd' or 'notckd' - ckd=chronic kidney disease. There are 400 rows

The data needs cleaning: in that it has NaNs and the numeric features need to be forced to floats. Basically, we were instructed to get rid of ALL ROWS with Nans, with no threshold - meaning, any row that has even one NaN, gets deleted.

Part 1: We are asked to choose 3 features (bgr, rc, wc), visualize them, then run the PCA with n_components=2.
the PCA is to be run twice: one with no scaling and the second run WITH scaling. And this is where my issue starts … in that after scaling I can hardly see any difference!

I will stop here for now till I get feedback and then move to Part 2.

## Acknowledgements
The dataset is available at: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease

### Dataset Information
#### Additional Information

| Variable | English Description | Chinese Translation |
|----------|---------------------|---------------------|
| age      | age                 | 年龄                |
| bp       | blood pressure      | 血压                |
| sg       | specific gravity    | 比重                |
| al       | albumin             | 白蛋白              |
| su       | sugar               | 糖                  |
| rbc      | red blood cells     | 红细胞              |
| pc       | pus cell            | 脓细胞              |
| pcc      | pus cell clumps     | 脓细胞团           |
| ba       | bacteria            | 细菌                |
| bgr      | blood glucose random| 随机血糖            |
| bu       | blood urea          | 血尿素              |
| sc       | serum creatinine    | 血清肌酐            |
| sod      | sodium              | 钠                  |
| pot      | potassium           | 钾                  |
| hemo     | hemoglobin          | 血红蛋白            |
| pcv      | packed cell volume  | 红细胞压积          |
| wc       | white blood cell count | 白细胞计数          |
| rc       | red blood cell count| 红细胞计数          |
| htn      | hypertension        | 高血压              |
| dm       | diabetes mellitus   | 糖尿病              |
| cad      | coronary artery disease | 冠状动脉疾病      |
| appet    | appetite            | 食欲                |
| pe       | pedal edema         | 足部水肿            |
| ane      | anemia              | 贫血                |
| class    | class               | 类别（慢性肾病与否） |

### Variables Table

| Variable Name | Role   | Type       | Demographic | Description | Units | Missing Values |
|---------------|--------|------------|-------------|------------|-------|----------------|
| age           | Feature| Integer    | Age         |             | year  | yes            |
| bp            | Feature| Integer    | Blood Pressure | Blood pressure | mm/Hg | yes            |
| sg            | Feature| Categorical|             | Specific gravity |      | yes            |
| al            | Feature| Categorical|             | Albumin      |      | yes            |
| su            | Feature| Categorical|             | Sugar       |      | yes            |
| rbc           | Feature| Binary     | Red Blood Cells | Red blood cells |      | yes            |
| pc            | Feature| Binary     | Pus Cell    | Pus cell    |      | yes            |
| pcc           | Feature| Binary     | Pus Cell Clumps | Pus cell clumps |      | yes            |
| ba            | Feature| Binary     | Bacteria    | Bacteria    |      | yes            |
| bgr           | Feature| Integer    | Blood Glucose Random | Blood glucose random | mgs/dl | yes            |
| bu            | Feature| Integer    | Blood Urea | Blood urea | mgs/dl | yes            |
| sc            | Feature| Continuous | Serum Creatinine | Serum creatinine | mgs/dl | yes            |
| sod           | Feature| Integer    | Sodium      | Sodium      | mEq/L | yes            |
| pot           | Feature| Continuous | Potassium   | Potassium   | mEq/L | yes            |
| hemo          | Feature| Continuous | Hemoglobin  | Hemoglobin  | gms   | yes            |
| pcv           | Feature| Integer    | Packed Cell Volume | Packed cell volume |      | yes            |
| wbcc          | Feature| Integer    | White Blood Cell Count | White blood cell count | cells/cmm | yes            |
| rbcc          | Feature| Continuous | Red Blood Cell Count | Red blood cell count | millions/cmm | yes            |
| htn           | Feature| Binary     | Hypertension | Hypertension |      | yes            |
| dm            | Feature| Binary     | Diabetes Mellitus | Diabetes mellitus |      | yes            |
| cad           | Feature| Binary     | Coronary Artery Disease | Coronary artery disease |      | yes            |
| appet         | Feature| Binary     | Appetite    | Appetite    |      | yes            |
| pe            | Feature| Binary     | Pedal Edema | Pedal edema |      | yes            |
| ane           | Feature| Binary     | Anemia      | Anemia      |      | yes            |
| class         | Target | Binary     | CKD or not CKD | CKD or not CKD |      | no             |




| Variable Name | Role   | Type       | Demographic | Description                | Units  | Missing Values |
|---------------|--------|------------|-------------|----------------------------|--------|----------------|
| age           | 特征   | 整数       | 年龄         |                             | 年     | 是             |
| bp            | 特征   | 整数       | 血压         | 血压                        | 毫米汞柱 | 是             |
| sg            | 特征   | 分类       |             | 比重                        |        | 是             |
| al            | 特征   | 分类       |             | 白蛋白                      |        | 是             |
| su            | 特征   | 分类       |             | 糖                          |        | 是             |
| rbc           | 特征   | 二元       | 红细胞计数   | 红细胞                      |        | 是             |
| pc            | 特征   | 二元       | 脓细胞      | 脓细胞                      |        | 是             |
| pcc           | 特征   | 二元       | 脓细胞团    | 脓细胞团                    |        | 是             |
| ba            | 特征   | 二元       | 细菌        | 细菌                        |        | 是             |
| bgr           | 特征   | 整数       | 随机血糖     | 随机血糖                    | 毫克/分升 | 是             |
| bu            | 特征   | 整数       | 血尿素       | 血尿素                      | 毫克/分升 | 是             |
| sc            | 特征   | 连续       | 血清肌酐     | 血清肌酐                    | 毫克/分升 | 是             |
| sod           | 特征   | 整数       | 钠          | 钠                          | 毫摩尔/升 | 是             |
| pot           | 特征   | 连续       | 钾          | 钾                          | 毫摩尔/升 | 是             |
| hemo          | 特征   | 连续       | 血红蛋白     | 血红蛋白                    | 克     | 是             |
| pcv           | 特征   | 整数       | 红细胞压积   | 红细胞压积                  |        | 是             |
| wbcc          | 特征   | 整数       | 白细胞计数   | 白细胞计数                  | 细胞/立方毫米 | 是             |
| rbcc          | 特征   | 连续       | 红细胞计数   | 红细胞计数                  | 百万/立方毫米 | 是             |
| htn           | 特征   | 二元       | 高血压      | 高血压                      |        | 是             |
| dm            | 特征   | 二元       | 糖尿病      | 糖尿病                      |        | 是             |
| cad           | 特征   | 二元       | 冠状动脉疾病 | 冠状动脉疾病                |        | 是             |
| appet         | 特征   | 二元       | 食欲        | 食欲                        |        | 是             |
| pe            | 特征   | 二元       | 足部水肿    | 足部水肿                    |        | 是             |
| ane           | 特征   | 二元       | 贫血        | 贫血                        |        | 是             |
| class         | 目标   | 二元       | 慢性肾病或非慢性肾病 | 慢性肾病或非慢性肾病        |        | 否             |



#### Additional Variable Information

We use 24 + class = 25 ( 11  numeric ,14  nominal)


Additional Variable Information
We use 24 + class = 25 (11 numerical, 14 nominal)

1. Age (numerical)
   - Age in years (年龄，以年为单位)
2. Blood Pressure (numerical)
   - BP in mm/Hg (血压，以毫米汞柱为单位)
3. Specific Gravity (nominal)
   - SG - (1.005, 1.010, 1.015, 1.020, 1.025) (比重 - 1.005至1.025)
4. Albumin (nominal)
   - AL - (0, 1, 2, 3, 4, 5) (白蛋白 - 0至5级)
5. Sugar (nominal)
   - SU - (0, 1, 2, 3, 4, 5) (糖 - 0至5级)
6. Red Blood Cells (nominal)
   - RBC - (normal, abnormal) (红细胞 - 正常、异常)
7. Pus Cell (nominal)
   - PC - (normal, abnormal) (脓细胞 - 正常、异常)
8. Pus Cell clumps (nominal)
   - PCC - (present, not present) (脓细胞团 - 存在、不存在)
9. Bacteria (nominal)
   - BA - (present, not present) (细菌 - 存在、不存在)
10. Blood Glucose Random (numerical)
    - BGR in mgs/dl (随机血糖 - 以毫克/分升为单位)
11. Blood Urea (numerical)
    - BU in mgs/dl (血尿素 - 以毫克/分升为单位)
12. Serum Creatinine (numerical)
    - SC in mgs/dl (血清肌酐 - 以毫克/分升为单位)
13. Sodium (numerical)
    - SOD in mEq/L (钠 - 以毫当量/升为单位)
14. Potassium (numerical)
    - POT in mEq/L (钾 - 以毫当量/升为单位)
15. Hemoglobin (numerical)
    - HEMO in gms (血红蛋白 - 以克为单位)
16. Packed Cell Volume (numerical)
17. White Blood Cell Count (numerical)
    - WC in cells/cumm (白细胞计数 - 以细胞/立方毫米为单位)
18. Red Blood Cell Count (numerical)
    - RC in millions/cmm (红细胞计数 - 以百万/立方毫米为单位)
19. Hypertension (nominal)
    - HTN - (yes, no) (高血压 - 是、否)
20. Diabetes Mellitus (nominal)
    - DM - (yes, no) (糖尿病 - 是、否)
21. Coronary Artery Disease (nominal)
    - CAD - (yes, no) (冠状动脉疾病 - 是、否)
22. Appetite (nominal)
    - Appet - (good, poor) (食欲 - 好、差)
23. Pedal Edema (nominal)
    - PE - (yes, no) (足部水肿 - 是、否)
24. Anemia (nominal)
    - ANE - (yes, no) (贫血 - 是、否)
25. Class (nominal)
    - Class - (CKD, not CKD) (类别 - 慢性肾病、非慢性肾病)


| Variable | English Description | Chinese Translation |
|----------|---------------------|---------------------|
| age      | age                 | 年龄                |
| bp       | blood pressure      | 血压                |
| sg       | specific gravity    | 比重                |
| al       | albumin             | 白蛋白              |
| su       | sugar               | 糖                  |
| rbc      | red blood cells     | 红细胞              |
| pc       | pus cell            | 脓细胞              |
| pcc      | pus cell clumps     | 脓细胞团           |
| ba       | bacteria            | 细菌                |
| bgr      | blood glucose random| 随机血糖            |
| bu       | blood urea          | 血尿素              |
| sc       | serum creatinine    | 血清肌酐            |
| sod      | sodium              | 钠                  |
| pot      | potassium           | 钾                  |
| hemo     | hemoglobin          | 血红蛋白            |
| pcv      | packed cell volume  | 红细胞压积          |
| wc       | white blood cell count | 白细胞计数          |
| rc       | red blood cell count| 红细胞计数          |
| htn      | hypertension        | 高血压              |
| dm       | diabetes mellitus   | 糖尿病              |
| cad      | coronary artery disease | 冠状动脉疾病      |
| appet    | appetite            | 食欲                |
| pe       | pedal edema         | 足部水肿            |
| ane      | anemia              | 贫血                |
| class    | class               | 类别（慢性肾病与否） |


---


| # | Variable (变量) | Type (类型) | Values (取值) | Description (描述) | Chinese Translation (中文翻译) |
|---|----------------|-----------|--------------|------------------|------------------------------|
| 1 | Age             | numerical |              | Age in years     | 年龄，以年为单位            |
| 2 | Blood Pressure  | numerical |              | BP in mm/Hg      | 血压，以毫米汞柱为单位    |
| 3 | Specific Gravity| nominal   | 1.005,1.010,1.015,1.020,1.025 | Specific Gravity | 比重 - 1.005至1.025        |
| 4 | Albumin         | nominal   | 0,1,2,3,4,5   | Albumin          | 白蛋白 - 0至5级            |
| 5 | Sugar           | nominal   | 0,1,2,3,4,5   | Sugar            | 糖 - 0至5级                |
| 6 | Red Blood Cells | nominal   | normal,abnormal | Red Blood Cells  | 红细胞 - 正常、异常      |
| 7 | Pus Cell        | nominal   | normal,abnormal | Pus Cell         | 脓细胞 - 正常、异常      |
| 8 | Pus Cell clumps | nominal   | present,not present | Pus Cell clumps | 脓细胞团 - 存在、不存在  |
| 9 | Bacteria        | nominal   | present,not present | Bacteria | 细菌 - 存在、不存在      |
|10 | Blood Glucose Random | numerical |              | BGR in mgs/dl   | 随机血糖 - 以毫克/分升为单位|
|11 | Blood Urea     | numerical |              | BU in mgs/dl    | 血尿素 - 以毫克/分升为单位|
|12 | Serum Creatinine| numerical |              | SC in mgs/dl     | 血清肌酐 - 以毫克/分升为单位|
|13 | Sodium          | numerical |              | SOD in mEq/L     | 钠 - 以毫当量/升为单位    |
|14 | Potassium       | numerical |              | POT in mEq/L     | 钾 - 以毫当量/升为单位    |
|15 | Hemoglobin      | numerical |              | HEMO in gms      | 血红蛋白 - 以克为单位    |
|16 | Packed Cell Volume | numerical |              |                  | 红细胞压积                |
|17 | White Blood Cell Count | numerical |              | WC in cells/cumm | 白细胞计数 - 以细胞/立方毫米为单位 |
|18 | Red Blood Cell Count | numerical |              | RC in millions/cmm | 红细胞计数 - 以百万/立方毫米为单位 |
|19 | Hypertension    | nominal   | yes,no       | Hypertension    | 高血压 - 是、否          |
|20 | Diabetes Mellitus | nominal | yes,no       | Diabetes Mellitus| 糖尿病 - 是、否          |
|21 | Coronary Artery Disease | nominal | yes,no       | Coronary Artery Disease | 冠状动脉疾病 - 是、否    |
|22 | Appetite        | nominal   | good,poor    | Appetite        | 食欲 - 好、差            |
|23 | Pedal Edema     | nominal   | yes,no       | Pedal Edema     | 足部水肿 - 是、否        |
|24 | Anemia          | nominal   | yes,no       | Anemia          | 贫血 - 是、否            |
|25 | Class           | nominal   | CKD,not CKD  | Class           | 类别 - 慢性肾病、非慢性肾病 |


---


## Inspiration
I would like to get an intuitive and a practical understanding of PCA.








