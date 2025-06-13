以下是关于 **`pyhealth.tasks` 模块** 的中文系统学习笔记，聚焦其核心任务定义逻辑、支持任务类型与用法演示，适合系统掌握 PyHealth 中的任务建模机制。

---

# 🎯 PyHealth 学习笔记：`pyhealth.tasks` 模块详解

---

## ✅ 一、模块功能概述

在 PyHealth 中，`pyhealth.tasks` 模块用于**定义医疗预测任务的处理逻辑**，包括：

* 如何从患者的原始多模态数据中抽取样本
* 如何标注标签（output schema）
* 如何组织特征（input schema）
* 如何与下游模型对接训练数据结构

> 📌 你可以使用官方提供的任务函数，也可以自定义任务（继承 `BaseTask`）。

---

## 🧱 二、任务构建的结构化流程

PyHealth 任务构建遵循以下标准流程：

```python
# Step 1: 导入数据集（如 MIMIC3）
from pyhealth.datasets import MIMIC3Dataset
dataset = MIMIC3Dataset(root=..., tables=[...])

# Step 2: 定义任务（如：院内死亡预测）
from pyhealth.tasks import MortalityPredictionMIMIC3
task_fn = MortalityPredictionMIMIC3()

# Step 3: 设置任务结构
samples = dataset.set_task(task_fn=task_fn)
```

---

## 📦 三、支持的常见任务类型

PyHealth 中内置了多个主流医疗任务，涵盖 EHR、ECG、EEG、影像等模态：

### 📌 1. Mortality Prediction 死亡预测

| 类别        | 调用函数                                                       |
| --------- | ---------------------------------------------------------- |
| MIMIC-III | `MortalityPredictionMIMIC3()`                              |
| MIMIC-IV  | `MortalityPredictionMIMIC4()`                              |
| eICU      | `MortalityPredictionEICU()` / `MortalityPredictionEICU2()` |
| OMOP      | `MortalityPredictionOMOP()`                                |
| 多模态 MIMIC | `MultimodalMortalityPredictionMIMIC3/4()`                  |

---

### 📌 2. Readmission Prediction 再入院预测

| 类别        | 函数                                   |
| --------- | ------------------------------------ |
| MIMIC-III | `readmission_prediction_mimic3_fn()` |
| MIMIC-IV  | `readmission_prediction_mimic4_fn()` |
| eICU      | `readmission_prediction_eicu_fn()`   |
| OMOP      | `readmission_prediction_omop_fn()`   |

---

### 📌 3. Length of Stay Prediction 住院时间预测

| 类别        | 函数                                      |
| --------- | --------------------------------------- |
| MIMIC-III | `length_of_stay_prediction_mimic3_fn()` |
| MIMIC-IV  | `length_of_stay_prediction_mimic4_fn()` |
| eICU      | `length_of_stay_prediction_eicu_fn()`   |

> 🎯 支持分类 / 回归方式，按住院时长分段或直接预测数值

---

### 📌 4. Drug Recommendation 药物推荐

| 数据源       | 函数                                |
| --------- | --------------------------------- |
| MIMIC-III | `drug_recommendation_mimic3_fn()` |
| MIMIC-IV  | `drug_recommendation_mimic4_fn()` |
| eICU      | `drug_recommendation_eicu_fn()`   |
| OMOP      | `drug_recommendation_omop_fn()`   |

---

### 📌 5. ICD9编码预测任务

| 任务名称             | 函数                                |
| ---------------- | --------------------------------- |
| MIMIC3 ICD 多标签任务 | `pyhealth.tasks.MIMIC3ICD9Coding` |

---

### 📌 6. Sleep Staging 睡眠阶段划分

| 数据集       | 函数                            |
| --------- | ----------------------------- |
| ISRUC     | `sleep_staging_isruc_fn()`    |
| Sleep-EDF | `sleep_staging_sleepedf_fn()` |
| SHHS      | `sleep_staging_shhs_fn()`     |

---

### 📌 7. EEG 任务：异常检测与事件识别

| 任务   | 函数                    |
| ---- | --------------------- |
| 异常识别 | `EEG_isAbnormal_fn()` |
| 事件检测 | `EEG_events_fn()`     |

---

### 📌 8. 心电图（ECG）检测任务

使用 `cardiology_isAR_fn()`、`cardiology_isCD_fn()` 等函数对心律失常、传导阻滞等进行识别。

---

### 📌 9. COVID-19 X光分类任务

| 类别   | 函数                         |
| ---- | -------------------------- |
| 图像分类 | `COVID19CXRClassification` |

---

## 🧩 四、任务结构详解（以 `MortalityPredictionMIMIC3` 为例）

```python
from pyhealth.tasks import MortalityPredictionMIMIC3

task = MortalityPredictionMIMIC3()

print(task.task_name)       # 任务名，如 "mortality_prediction_mimic3"
print(task.input_schema)    # 输入特征结构（如诊断、检验、用药）
print(task.output_schema)   # 输出标签结构（如0/1）
```

每个任务函数封装了输入输出规则，并能与 PyHealth 数据集无缝对接。

---

## 🧪 五、任务设置 + 数据划分 + 模型训练 示例

```python
from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.models import RNN
from pyhealth.trainer import Trainer

# 1. 加载数据
dataset = MIMIC3Dataset(...)

# 2. 设定任务
task = MortalityPredictionMIMIC3()
samples = dataset.set_task(task)

# 3. 数据划分
train_set, val_set, test_set = split_by_patient(samples, [0.8, 0.1, 0.1])

# 4. 模型训练
model = RNN(dataset=samples)
trainer = Trainer(model)
trainer.train(get_dataloader(train_set), get_dataloader(val_set))
```

---

## ✅ 六、小结

| 内容     | 说明                              |
| ------ | ------------------------------- |
| 模块名称   | `pyhealth.tasks`                |
| 功能     | 定义各类医疗预测任务                      |
| 支持任务类型 | 死亡预测、再入院、住院时间、药物推荐、ICD 编码、睡眠分期等 |
| 使用方法   | 导入任务 → 调用 `.set_task()` 接入数据集   |
| 可自定义   | ✅（可继承 `BaseTask` 编写自己的标签生成规则）   |

---

如果你希望我**详细讲解如何自定义一个任务**，或继续进入模型模块（如 RETAIN、GRU-D），我可以为你继续整理学习笔记并附上运行示例。是否继续？

以下是关于 `pyhealth.tasks.BaseTask` 的中文学习笔记，适合希望**自定义任务**的用户深入理解其继承机制与实现方式。

---

# 🔧 PyHealth 学习笔记：`BaseTask` 基类详解

---

## ✅ 一、作用说明

`pyhealth.tasks.BaseTask` 是 PyHealth 中所有任务（如死亡预测、再入院预测等）的**抽象基类（ABC）**，用于统一不同任务的输入输出结构与任务逻辑。

如果你想自定义一个任务（例如预测某项特殊检验值是否异常），就需要继承该类，并实现其关键接口。

---

## 🧱 二、设计核心

该类通过 Python 的 `abc` 模块实现抽象类，强制子类实现以下方法和属性：

### 🌟 必须实现的接口

| 属性/方法           | 类型          | 功能                                           |
| --------------- | ----------- | -------------------------------------------- |
| `task_name`     | `str`       | 任务名称（必须唯一）                                   |
| `input_schema`  | `List[str]` | 输入数据类型（如 `conditions`, `procedures`, `labs`） |
| `output_schema` | `List[str]` | 输出标签结构（如 `mortality`、`los`）                  |
| `__call__()`    | 方法          | 传入 `Patient` 实例，返回样本列表（dict 结构）              |

---

## 🧪 三、自定义任务的完整示例

以下是一个自定义任务的简单模板，预测每位病人在首次诊断后是否会住院超过 7 天：

```python
from pyhealth.tasks import BaseTask
from typing import List, Dict

class MyStayPredictionTask(BaseTask):
    def __init__(self):
        self.task_name = "my_los_prediction"
        self.input_schema = ["conditions", "procedures"]
        self.output_schema = ["label"]

    def __call__(self, patient) -> List[Dict]:
        samples = []
        visits = patient.get_events(event_type="visit")
        if not visits:
            return []

        for visit in visits:
            length_of_stay = visit.attr_dict.get("length_of_stay", 0)
            label = int(length_of_stay > 7)
            sample = {
                "patient_id": patient.patient_id,
                "visit_id": visit.attr_dict.get("visit_id", None),
                "input": {
                    "conditions": patient.get_events(event_type="diagnosis"),
                    "procedures": patient.get_events(event_type="procedure"),
                },
                "output": {
                    "label": label
                }
            }
            samples.append(sample)
        return samples
```

---

## 🚀 四、接入 PyHealth 数据集流程

一旦你定义了自己的任务类，就可以像官方任务一样接入：

```python
from pyhealth.datasets import MIMIC3Dataset
dataset = MIMIC3Dataset(...)

from my_tasks import MyStayPredictionTask
task = MyStayPredictionTask()
samples = dataset.set_task(task_fn=task)
```

---

## 🔚 五、小结

| 项目   | 内容                               |
| ---- | -------------------------------- |
| 模块   | `pyhealth.tasks.BaseTask`        |
| 用途   | 自定义任务基类                          |
| 特点   | 使用抽象类机制，强制实现 `__call__`、schema 等 |
| 典型用法 | 自定义医疗预测任务结构                      |
| 接入方式 | 与官方任务方式一致，可用于 `.set_task()`      |

---

需要我继续讲解 `pyhealth.models` 模块（如 RNN、RETAIN）或整理“如何批量注册多个任务”？欢迎告诉我你的下一步需求！


以下是关于 `pyhealth.tasks.Readmission30DaysMIMIC4` 的中文学习笔记，适合你理解 **30天再入院预测任务** 的背景与实际操作方式。

---

# 🔁 PyHealth 学习笔记：`Readmission30DaysMIMIC4`

---

## ✅ 一、任务背景简介

在临床研究与管理中，**再入院率（Readmission Rate）** 是衡量住院服务质量的重要指标。该任务模拟了：

> 🏥 *基于住院病人的既往诊疗信息，预测其在出院后 30 天内是否会再次住院。*

该任务基于 **MIMIC-IV** 数据库。

---

## 🔧 二、任务定义类

```python
from pyhealth.tasks import Readmission30DaysMIMIC4

task = Readmission30DaysMIMIC4()
```

这是一个继承自 `BaseTask` 的标准任务类，适用于 MIMIC-IV 数据集结构。

---

## 📥 三、输入定义：`input_schema`

### 📌 输入包含三种类型的数据（即特征）：

```python
{
  "conditions": "List of condition codes (如 ICD10)",
  "procedures": "List of procedure codes (如 CPT)",
  "drugs": "List of drug codes (如 NDC)"
}
```

* **conditions**：疾病诊断代码序列
* **procedures**：医疗操作代码序列
* **drugs**：处方药品代码序列

这些是按时间排序的患者就诊历史，可以用于构建时间序列模型（如 RNN、GRU-D、RETAIN 等）。

---

## 📤 四、输出定义：`output_schema`

```python
{
  "readmission": "Binary (0 or 1)"
}
```

* `readmission=1`：表示在出院 30 天内再次入院
* `readmission=0`：无再入院

这个是标准的**二分类任务（binary classification）**。

---

## 🧪 五、调用与数据生成流程

```python
from pyhealth.datasets import MIMIC4Dataset

# 加载 MIMIC-IV 数据（你也可以用自己的数据集）
dataset = MIMIC4Dataset(
    root="路径",
    tables=["diagnoses_icd", "procedures_icd", "prescriptions"]
)

# 设置任务函数
task = Readmission30DaysMIMIC4()
sample_dataset = dataset.set_task(task_fn=task)

# 拆分训练/验证/测试
from pyhealth.datasets import split_by_patient, get_dataloader
train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])
train_loader = get_dataloader(train_ds, batch_size=32)
```

---

## 🧠 六、建模建议

你可以直接套用：

```python
from pyhealth.models import RETAIN, GRU, LSTM

model = RETAIN(dataset=sample_dataset)  # 或 GRU/LSTM
```

训练方式：

```python
from pyhealth.trainer import Trainer

trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    monitor="roc_auc",
    epochs=30
)
```

---

## ✅ 七、小结

| 项目   | 内容                                  |
| ---- | ----------------------------------- |
| 任务名  | `Readmission30DaysMIMIC4`           |
| 数据集  | `MIMIC-IV`                          |
| 输入   | `conditions`, `procedures`, `drugs` |
| 输出   | `readmission`（是否30天内再入院）            |
| 模型类型 | 二分类                                 |
| 适配模型 | LSTM / GRU / RETAIN 等时序网络           |
| 使用方式 | `set_task()` 接入 PyHealth pipeline   |

---

是否继续讲解另一个任务（如 `MortalityPredictionMIMIC3`）或进入 `pyhealth.models` 的模型说明？欢迎告诉我你的下一个需求方向。

以下是关于 `pyhealth.tasks.InHospitalMortalityMIMIC4`（住院期间死亡预测任务）的中文学习笔记，适用于你基于 MIMIC-IV 数据集构建死亡预测模型。

---

# 🏥 PyHealth 学习笔记：`InHospitalMortalityMIMIC4`

> **任务目标：预测患者在本次住院过程中是否死亡**

---

## ✅ 一、任务背景简介

住院期间死亡率（In-Hospital Mortality）是衡量重症医疗质量的重要指标。本任务模拟如下临床问题：

> 基于患者入院后的**实验室检查结果（lab results）**，预测该患者是否会在住院期间死亡。

适用于 ICU 数据，尤其来自 MIMIC-IV 数据集。

---

## 🔧 二、任务定义类

```python
from pyhealth.tasks import InHospitalMortalityMIMIC4

task = InHospitalMortalityMIMIC4()
```

该任务类继承自 `BaseTask`，可与 `MIMIC4Dataset` 配合使用。

---

## 📥 三、输入定义：`input_schema`

```python
{
  "labs": "A timeseries of lab results"
}
```

说明：

* `labs`：多次住院过程中，不同时间点的实验室检查数据。

  * 例如：血红蛋白、白细胞计数、血糖、电解质等
  * 是一个 **时间序列结构**，可用于 RNN、RETAIN 等模型

---

## 📤 四、输出定义：`output_schema`

```python
{
  "mortality": "Binary (0 or 1)"
}
```

说明：

* `mortality=1`：表示患者在此次住院中死亡
* `mortality=0`：表示患者在此次住院中生还出院
* 属于标准的 **二分类任务**

---

## 🧪 五、任务数据构建流程（完整示例）

```python
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import InHospitalMortalityMIMIC4
from pyhealth.datasets import split_by_patient, get_dataloader

# 步骤 1：加载 MIMIC-IV 数据
dataset = MIMIC4Dataset(
    root="路径",
    tables=["labevents"]  # 只需实验室检查表
)

# 步骤 2：设定任务
task = InHospitalMortalityMIMIC4()
sample_dataset = dataset.set_task(task_fn=task)

# 步骤 3：划分数据集
train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])
train_loader = get_dataloader(train_ds, batch_size=32)
val_loader = get_dataloader(val_ds, batch_size=32)
test_loader = get_dataloader(test_ds, batch_size=32)
```

---

## 🤖 六、适配模型建议

该任务基于时间序列（实验室数据）进行预测，推荐使用：

* `GRU`
* `RETAIN`
* `LSTM`
* `Transformer`

```python
from pyhealth.models import GRU

model = GRU(dataset=sample_dataset)
```

---

## 🚀 七、训练流程（示例）

```python
from pyhealth.trainer import Trainer

trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    monitor="roc_auc",
    epochs=20
)

# 评估模型
trainer.evaluate(test_loader)
```

---

## ✅ 八、小结

| 项目   | 内容                          |
| ---- | --------------------------- |
| 任务名  | `InHospitalMortalityMIMIC4` |
| 数据集  | MIMIC-IV                    |
| 输入   | `labs`（实验室检查数据）             |
| 输出   | `mortality`（是否死亡）           |
| 任务类型 | 二分类                         |
| 推荐模型 | GRU、RETAIN、LSTM、Transformer |
| 应用场景 | ICU 病人风险评估、重症监控建模等          |

---

是否继续讲解下一个任务（如 `drug_recommendation_mimic4_fn` 或 `MortalityPredictionMIMIC3`）？也可以进入模型模块 `pyhealth.models` 继续深入学习。请告诉我你下一步想学的内容。

以下是关于 PyHealth 中 `pyhealth.tasks.MIMIC3ICD9Coding` 的中文学习笔记，适用于基于 MIMIC-III 数据集的 **ICD-9 医疗编码任务**。

---

# 🧾 PyHealth 学习笔记：`MIMIC3ICD9Coding`

> **任务目标：根据临床文本记录预测对应的 ICD-9 诊断编码**

---

## ✅ 一、任务简介

ICD 编码任务（Medical Coding）是指根据病人的病历、出院记录、医生笔记等**临床文本资料**，预测其应被标注的疾病诊断代码。

在 MIMIC-III 中，每个住院病人可能会有多个 ICD-9 编码。该任务是一个 **多标签分类任务**（multi-label classification），每位病人可能同时拥有多个标签（编码）。

---

## 🧠 二、任务应用场景

| 场景     | 说明                |
| ------ | ----------------- |
| 医疗自动编码 | 减轻病历归档人员工作负担      |
| 文本建模   | 自然语言处理与医学的交叉      |
| 数据增强   | 作为辅助标签任务改进诊断预测准确性 |

---

## 🛠️ 三、任务定义结构

```python
from pyhealth.tasks import MIMIC3ICD9Coding

task = MIMIC3ICD9Coding()
```

---

## 📥 四、输入定义 `input_schema`

```python
{
  "notes": "clinical notes in text format"
}
```

* `notes`：患者住院过程中的文本数据，可能包含出院记录、医生诊断意见等。
* 类型：文本序列（String）

---

## 📤 五、输出定义 `output_schema`

```python
{
  "icd9_codes": "List of ICD-9 codes"
}
```

* `icd9_codes`：一个病人可能被标注的 ICD-9 诊断编码集合（多标签）
* 类型：`List[str]`

---

## 🧪 六、任务流程简要示例

```python
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import MIMIC3ICD9Coding
from pyhealth.datasets import split_by_patient, get_dataloader

# 步骤 1：加载文本数据
dataset = MIMIC3Dataset(
    root="路径",
    tables=["NOTEEVENTS"]  # 仅需文本记录
)

# 步骤 2：设置任务
task = MIMIC3ICD9Coding()
sample_dataset = dataset.set_task(task_fn=task)

# 步骤 3：划分数据集
train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])
train_loader = get_dataloader(train_ds, batch_size=32)
val_loader = get_dataloader(val_ds, batch_size=32)
test_loader = get_dataloader(test_ds, batch_size=32)
```

---

## 🤖 七、适配模型建议

由于输入为**文本数据**，该任务适合使用 NLP 模型：

* `Transformer`
* `CNN`（用于文本卷积）
* `MLP`（对 BOW 或 TF-IDF 特征）

如使用 Transformer 模型：

```python
from pyhealth.models import Transformer

model = Transformer(dataset=sample_dataset)
```

---

## 🚀 八、训练示例

```python
from pyhealth.trainer import Trainer

trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    monitor="f1",  # 多标签任务可选 F1-score
    epochs=20
)

# 测试模型
trainer.evaluate(test_loader)
```

---

## ✅ 九、小结

| 项目   | 内容                               |
| ---- | -------------------------------- |
| 任务名  | `MIMIC3ICD9Coding`               |
| 数据集  | MIMIC-III                        |
| 输入   | `notes`（临床文本记录）                  |
| 输出   | `icd9_codes`（ICD-9 编码，多标签）       |
| 任务类型 | 多标签文本分类                          |
| 推荐模型 | Transformer, CNN, Text-based MLP |
| 应用场景 | 自动ICD编码、临床NLP建模                  |

---

是否继续下一个任务（如 `drug_recommendation_mimic3_fn()` 或 `length_of_stay_prediction_mimic3_fn()`）？也可以深入学习 `pyhealth.models` 中 Transformer 的具体结构。请告诉我你的下一步学习方向。

以下是针对 **PyHealth 中心电图（心电）疾病分类任务** `pyhealth.tasks.cardiology_detect` 模块的**中文学习笔记**，适用于使用 CardiologyDataset 构建多个二分类任务（如心律失常、传导阻滞等）。

---

# ❤️ PyHealth 学习笔记：`cardiology_detect` 心电疾病分类任务模块

---

## 🧩 一、任务模块概览

`pyhealth.tasks.cardiology_detect` 中定义了**六类**基于心电图（ECG）信号的**二分类任务函数**，分别对应心脏病的六种异常：

| 分类函数                    | 预测病症                            | 中文含义     |
| ----------------------- | ------------------------------- | -------- |
| `cardiology_isAR_fn`    | Arrhythmias                     | 心律失常     |
| `cardiology_isBBBFB_fn` | Bundle branch/fascicular blocks | 传导束/分支阻滞 |
| `cardiology_isAD_fn`    | Axis deviations                 | 心电轴偏移    |
| `cardiology_isCD_fn`    | Conduction delays               | 心电传导延迟   |
| `cardiology_isWA_fn`    | Wave abnormalities              | 波形异常     |

---

## ⚙️ 二、每类任务的共通结构

```python
def cardiology_isXXX_fn(record, epoch_sec=10, shift=5):
    ...
```

* **输入参数：**

  * `record`：一个病人的信号数据（字典结构，含路径、性别、年龄等）
  * `epoch_sec`：每个采样窗口的持续秒数，默认 10 秒
  * `shift`：窗口滑动步长（单位：秒），默认 5 秒

* **返回：**

  * 样本列表，每个样本为一个字典，包含：

    * `patient_id`：病人编号
    * `record_id`：记录编号
    * `epoch_path`：分割后的信号片段存储路径（`.pkl` 格式）
    * `Sex`：性别
    * `Age`：年龄
    * `label`：该段信号是否出现指定疾病（0 或 1）

---

## 📚 三、使用流程（以心律失常为例）

### Step 1：加载心电数据集

```python
from pyhealth.datasets import CardiologyDataset

# chosen_dataset 六个参数控制是否载入对应六个子数据库（如 PTB、CPSC 等）
dataset = CardiologyDataset(
    root="physionet.org/files/challenge-2020/1.0.2/training",
    chosen_dataset=[1, 1, 1, 1, 1, 1]
)
```

### Step 2：选择任务函数（以心律失常为例）

```python
from pyhealth.tasks import cardiology_isAR_fn

# 设置任务并构建样本
cardiology_ds = dataset.set_task(task_fn=cardiology_isAR_fn)
```

### Step 3：查看样本结构

```python
print(cardiology_ds.samples[0])
```

输出示例：

```python
{
    'patient_id': '0_0',
    'visit_id': 'A0033',
    'record_id': 1,
    'Sex': ['Female'],
    'Age': ['34'],
    'epoch_path': '/xxx/0_0-0.pkl',
    'label': '0'
}
```

---

## 🤖 四、推荐模型：用于信号分析的模型

| 模型名             | 简介                     |
| --------------- | ---------------------- |
| `ResNet1D`      | 适合处理 1D 信号（如 ECG）      |
| `Transformer1D` | 适合长序列建模                |
| `CNN1D`         | 轻量、适配边缘部署              |
| 自定义网络           | 基于信号片段 `.pkl` 自行加载构建特征 |

---

## 📂 五、pkl 文件结构（epoch\_path）

每一个分段信号文件（`.pkl`）中包含以下信息：

```python
{
    "X": ndarray ECG信号矩阵（通道×长度）,
    "Sex": 性别,
    "Age": 年龄,
    "Y": 标签（0/1）
}
```

你可以使用 `pickle.load(open(epoch_path, "rb"))` 来加载信号和标签。

---

## 🧪 六、完整任务训练框架（简略版）

```python
from pyhealth.datasets import get_dataloader
from pyhealth.models import ResNet1D
from pyhealth.trainer import Trainer

train_ds, val_ds, test_ds = split_by_patient(cardiology_ds, [0.8, 0.1, 0.1])

train_loader = get_dataloader(train_ds, batch_size=32)
val_loader = get_dataloader(val_ds, batch_size=32)
test_loader = get_dataloader(test_ds, batch_size=32)

model = ResNet1D(dataset=cardiology_ds)

trainer = Trainer(model=model)
trainer.train(train_loader, val_loader, epochs=20)
trainer.evaluate(test_loader)
```

---

## ✅ 七、小结

| 项目   | 内容                                 |
| ---- | ---------------------------------- |
| 任务集合 | `pyhealth.tasks.cardiology_detect` |
| 支持分类 | 心律失常、传导阻滞、轴偏移等                     |
| 输入数据 | ECG 信号（分段）                         |
| 输出标签 | 二分类（是否有该异常）                        |
| 推荐模型 | ResNet1D、CNN1D、Transformer1D       |
| 数据集  | `CardiologyDataset`（PhysioNet ECG） |
| 输出结构 | 含 `epoch_path` 指向 `.pkl` 信号段       |

---

如果你希望进一步学习如何自定义心电模型、如何处理 `.pkl` 数据并构建数据加载器，或者希望我们从其他心电任务展开，请告诉我。是否继续学习下一个任务模块？

以下是 PyHealth 中用于处理胸部X光图像疾病分类任务模块 `pyhealth.tasks.COVID19CXRClassification` 的**中文学习笔记**，适合希望构建基于医学影像的多分类模型的学习者：

---

# 🩻 PyHealth 学习笔记：`COVID19CXRClassification` 胸部X光疾病分类任务

---

## 🧩 一、任务概述

`COVID19CXRClassification` 是 PyHealth 中专门针对 **胸部X光图像**（Chest X-ray, CXR）设计的多分类任务，用于自动识别肺部疾病，例如 COVID-19、肺炎等。

* 每位病人对应一张胸部X光图像；
* 模型学习从图像中提取特征，并预测其对应的疾病类别；
* 支持**多分类**任务。

---

## 🔧 二、任务定义

```python
from pyhealth.tasks import COVID19CXRClassification
```

### ✅ 1. 任务名

```python
COVID19CXRClassification.task_name  # "COVID19CXRClassification"
```

* 类型：`str`
* 用于标识任务类型

---

### ✅ 2. 输入模式（input\_schema）

```python
COVID19CXRClassification.input_schema
# {'image': 'image'}
```

说明：

| 键       | 说明                |
| ------- | ----------------- |
| `image` | 图像类型，要求传入单张胸部X光图像 |

图像可为 `.jpg`、`.png`、`.dcm` 等常见格式，通常在 Dataset 类中会有转换。

---

### ✅ 3. 输出模式（output\_schema）

```python
COVID19CXRClassification.output_schema
# {'disease': 'multiclass'}
```

说明：

| 键         | 类型         | 说明                             |
| --------- | ---------- | ------------------------------ |
| `disease` | multiclass | 图像对应的肺部疾病类别，例如 COVID-19、正常、肺炎等 |

---

## 🖼️ 三、数据样本结构

任务运行前，需要先有一个图像类数据集（你可以自定义或使用 PyHealth 提供的影像数据集）。

每个样本需包含：

* `patient_id`
* `image_path`：本地图像文件路径
* `label`：疾病标签（如 "COVID-19"、"normal"、"bacterial pneumonia"）

PyHealth 内部会自动将图像加载成张量（tensor）。

---

## 🧠 四、模型建议

| 模型             | 说明                             |
| -------------- | ------------------------------ |
| `ResNet18/50`  | 可作为默认图像分类模型（需加载 torchvision）   |
| `EfficientNet` | 更深更强的图像分类网络                    |
| 自定义CNN         | 支持自定义 PyTorch 模型结构，只要输入为图像张量即可 |

---

## 🧪 五、使用流程简略示例

```python
from pyhealth.datasets import YourCXRXrayDataset  # 自定义或已有图像数据集
from pyhealth.tasks import COVID19CXRClassification
from pyhealth.models import ResNet18
from pyhealth.trainer import Trainer

# 加载数据集并设置任务
dataset = YourCXRXrayDataset(root="data/CXR")
dataset = dataset.set_task(COVID19CXRClassification)

# 切分数据
train_ds, val_ds, test_ds = dataset.split([0.7, 0.1, 0.2])

# 模型构建（输入图像张量）
model = ResNet18(dataset=dataset)

# 训练器设置与训练
trainer = Trainer(model=model)
trainer.train(train_ds, val_ds, epochs=10)
trainer.evaluate(test_ds)
```

---

## ⚠️ 六、注意事项

* 图像处理需依赖 `torchvision`，确保已安装；
* 数据预处理建议统一图像大小、标准化；
* 数据集中每条样本需指向**单个图像路径**；
* 多分类标签需为整数编码或类别名映射。

---

## ✅ 七、小结表

| 项目   | 内容                                                 |
| ---- | -------------------------------------------------- |
| 任务名  | `COVID19CXRClassification`                         |
| 输入   | 单张胸部 X-ray 图像                                      |
| 输出   | 疾病分类（多分类）                                          |
| 模型建议 | ResNet18/50、EfficientNet、自定义CNN                    |
| 使用场景 | 医学影像辅助诊断，如 COVID-19 检测                             |
| 数据结构 | `{'image': 'image'}` → `{'disease': 'multiclass'}` |

---

是否需要我为你构建一个完整的 CXR 任务演示项目模板，或者继续讲解下一个任务模块？

以下是 PyHealth 中 “**药物推荐任务（drug recommendation）**” 模块的 **中文学习笔记整理**，涵盖了 `MIMIC-III`、`MIMIC-IV`、`eICU` 和 `OMOP` 等多种常用数据库，适合用于多标签药物预测任务建模学习：

---

# 💊 PyHealth 学习笔记：药物推荐任务 `drug_recommendation_*_fn`

---

## 🧩 一、任务简介

**药物推荐任务**旨在根据患者的**历史病情（诊断/手术/既往用药）**，预测其在当前住院就诊中的用药情况。

* 是一个 **多标签分类任务**（multi-label classification）
* 每条样本对应一次就诊，预测的是该次就诊所使用的药物集合

---

## 🔁 二、任务通用输入输出结构

每个 `drug_recommendation_*_fn(patient)` 函数会接收一个 `Patient` 对象，返回包含多条 `sample` 的列表，结构如下：

### ✅ 输入：

```python
{
    "conditions": [list of 诊断代码序列（每次就诊一组）],
    "procedures": [list of 手术代码序列（每次就诊一组）],
    "drugs_hist": [list of 药品历史记录（不含当前就诊）]
}
```

### ✅ 输出（预测目标）：

```python
"drugs": 当前住院就诊中使用的药物列表（多标签）
```

---

## 🏥 三、任务函数总览

| 函数名称                            | 对应数据库     | 常用表格                                             | 特点             |
| ------------------------------- | --------- | ------------------------------------------------ | -------------- |
| `drug_recommendation_mimic3_fn` | MIMIC-III | `DIAGNOSES_ICD`、`PROCEDURES_ICD`、`PRESCRIPTIONS` | 使用 ICD9 + 药品历史 |
| `drug_recommendation_mimic4_fn` | MIMIC-IV  | `diagnoses_icd`、`procedures_icd`                 | 使用 ICD10 及药品   |
| `drug_recommendation_eicu_fn`   | eICU      | `diagnosis`、`medication`                         | 精简病情和药物记录      |
| `drug_recommendation_omop_fn`   | OMOP CDM  | `condition_occurrence`、`procedure_occurrence`    | 通用 CDM 结构      |

---

## 🔍 四、各任务详细结构说明

### 📌 1. `drug_recommendation_mimic3_fn`

```python
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import drug_recommendation_mimic3_fn

mimic3_base = MIMIC3Dataset(
    root="mimiciii/path",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    code_mapping={"ICD9CM": "CCSCM"},
)

mimic3_sample = mimic3_base.set_task(drug_recommendation_mimic3_fn)
mimic3_sample.samples[0]
```

返回格式如下：

```python
{
  "patient_id": "107",
  "visit_id": "174162",
  "conditions": [["139", "158", ...]],        # 多次就诊，每次为一组诊断代码
  "procedures": [["4443", "4513"]],           # 多次就诊，每次为一组手术代码
  "drugs_hist": [[]],                         # 历史药物记录
  "drugs": ["0033", "5817", ...]              # 当前要预测的药物标签（目标）
}
```

---

### 📌 2. `drug_recommendation_mimic4_fn`

```python
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import drug_recommendation_mimic4_fn
```

输入和输出结构基本一致，只是使用 ICD10 代码和新版数据表：

```python
{
  "patient_id": "103",
  "visit_id": "130744",
  "conditions": [["42", "109", "98"]],
  "procedures": [["1"]],
  "label": [["2", "3", "4"]]  # drugs 的别名（即 prediction target）
}
```

---

### 📌 3. `drug_recommendation_eicu_fn`

```python
from pyhealth.datasets import eICUDataset
from pyhealth.tasks import drug_recommendation_eicu_fn
```

简化格式如下：

```python
{
  "patient_id": "103",
  "visit_id": "130744",
  "conditions": [["42", "109"]],
  "procedures": [["1"]],
  "label": [["2", "3", "4"]]
}
```

---

### 📌 4. `drug_recommendation_omop_fn`

```python
from pyhealth.datasets import OMOPDataset
from pyhealth.tasks import drug_recommendation_omop_fn
```

支持 CDM 结构，适配标准结构的多次记录：

```python
{
  "patient_id": "103",
  "visit_id": "130744",
  "conditions": [["42", "109"], ["98"]],
  "procedures": [["1"], ["2"]],
  "label": [["2", "3", "4"], ["0", "1"]]
}
```

---

## 🧠 五、典型模型建议

| 模型名称                     | 说明                          |
| ------------------------ | --------------------------- |
| `MLP`                    | 简单全连接模型，适合稀疏输入（如药物编码）       |
| `RNN/LSTM/GRU`           | 序列建模，考虑历史用药轨迹               |
| `Transformer`            | 捕捉长期依赖，适用于 visit-based 多步预测 |
| `Multi-label classifier` | 多标签预测器，通常输出 sigmoid 激活      |

---

## ✅ 六、小结表

| 内容       | 描述                                     |
| -------- | -------------------------------------- |
| 📌 任务类型  | 多标签药物推荐                                |
| 📚 输入特征  | `conditions`，`procedures`，`drugs_hist` |
| 🎯 输出标签  | `drugs` / `label`（预测该次就诊所开的药物）         |
| 🏥 数据库支持 | MIMIC-III、MIMIC-IV、eICU、OMOP           |
| 🔧 应用方向  | 医疗决策支持系统（CDSS）、智能药物推荐、辅助诊疗建议           |

---

需要我为你\*\*构建一个 drug recommendation 任务的完整复现代码模板（含模型+训练）\*\*吗？或者继续学习下一个任务模块？

以下是 **PyHealth 中 EEG 异常检测任务 `EEG_isAbnormal_fn`** 的中文学习笔记整理，便于你构建复现代码或加入工程模板：

---

# 🧠 PyHealth 学习笔记：EEG 异常检测任务 `EEG_isAbnormal_fn`

---

## 📌 一、任务目标

**EEG 异常检测**任务旨在自动识别脑电图（EEG）是否异常，即：

> 🔍 输入：一段 EEG 信号
> 🎯 输出：是否为异常信号（0：正常，1：异常）

该任务是一个 **二分类任务**（Binary Classification）。

---

## 🗂️ 二、任务数据来源：TUAB 数据集

* 名称：**TUABDataset**（Temple University Hospital EEG Abnormal Corpus）
* 数据内容：临床 EEG 原始数据及标注标签（正常/异常）
* 下载地址：[https://isip.piconepress.com/projects/nedc/html/tuh\_eeg/#c\_tuab](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tuab)

---

## 🧩 三、任务函数：`EEG_isAbnormal_fn(record)`

### ✅ 输入参数：

* `record`：**TUABDataset 中的单个 patient 记录**

  ```python
  {
    "load_from_path": "...",
    "signal_file": "...edf",
    "label_file": "...",
    "patient_id": "...",
    "visit_id": "...",
    "save_to_path": "..."
  }
  ```

---

### ✅ 输出结果：

返回一个样本列表（每个样本为一个 epoch）：

```python
{
  "patient_id": "aaaaamye",
  "visit_id": "s001",
  "record_id": "1",
  "epoch_path": "/path/to/0.pkl",  # 保存了 epoch 的路径（pkl文件中含 signal + label）
  "label": 1   # 二分类标签：0=正常，1=异常
}
```

---

## 🧪 四、完整调用流程

```python
from pyhealth.datasets import TUABDataset
from pyhealth.tasks import EEG_isAbnormal_fn

# 1. 加载 TUAB 数据集（脑电图）
tuab_ds = TUABDataset(
    root="/path/to/tuh_eeg_abnormal/v3.0.0/edf/",
    download=True  # 自动下载
)

# 2. 设置任务
eeg_abnormal_ds = tuab_ds.set_task(EEG_isAbnormal_fn)

# 3. 查看样本结构
print(eeg_abnormal_ds.samples[0])
```

---

## 🗃️ 五、数据文件结构说明（每条样本）

每条样本会生成一个 `.pkl` 文件，内容如下：

```python
{
  "signal": np.ndarray of shape (n_channels, time_steps),
  "label": 0 or 1
}
```

你可以使用 `pickle.load(open(epoch_path, 'rb'))` 加载查看数据内容。

---

## 📦 六、模型设计建议

由于 EEG 是时序信号，可采用如下模型：

| 模型结构                 | 简要说明                     |
| -------------------- | ------------------------ |
| CNN                  | 对 EEG 信号进行特征提取（如 ResNet） |
| RNN / LSTM           | 适合处理时序 EEG 特征演变          |
| Transformer          | 长序列建模，适合长时间 EEG 分析       |
| EEGNet / DeepConvNet | 专为脑电图设计的轻量模型结构           |

---

## ✅ 七、任务小结

| 项目      | 内容                          |
| ------- | --------------------------- |
| 📄 任务名称 | EEG 异常检测                    |
| 🏥 数据集  | TUABDataset                 |
| 🎯 预测目标 | 是否为异常脑电（label=0/1）          |
| ⚙️ 输出字段 | `epoch_path`（信号+标签）、`label` |
| 🧪 输入数据 | `.edf` 格式脑电文件               |
| 📊 任务类型 | 二分类 Binary Classification   |

---

需要我为你整理成一个**完整可复现模板代码 + EEG 分类模型训练脚本 + 导出 .zip 工程结构**吗？我可以为你一键生成。你也可以告诉我：是否想结合 PyTorch 模型一起生成。

以下是对 `pyhealth.tasks.EEG_events` 模块的**详细学习笔记**整理，适用于 EEG 信号事件标注任务的理解、复现与工程整合（含模型开发的思路）：

---

# 🧠 PyHealth 学习笔记：EEG 多分类事件标注任务 `EEG_events_fn`

---

## 🎯 一、任务目标

本任务用于对 **TUEV 数据集**中的 EEG 信号进行**事件类型标注**（Multiclass classification）：

> ✅ 每段 EEG 信号被划分为一小段（epoch），并预测其对应事件类别。

---

## 🧾 二、目标分类（6 类）

| 类别编码 | 类别含义                                                |
| ---- | --------------------------------------------------- |
| 1    | SPSW - spike and sharp wave                         |
| 2    | GPED - generalized periodic epileptiform discharges |
| 3    | PLED - periodic lateralized epileptiform discharges |
| 4    | EYEM - eye movement                                 |
| 5    | ARTF - artifact                                     |
| 6    | BCKG - background                                   |

---

## 🗂️ 三、数据来源：TUEV 数据集

* 名称：`TUEVDataset`
* 官网：[https://isip.piconepress.com/projects/tuh\_eeg/html/downloads.shtml](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)
* 数据格式：EDF（脑电原始信号），包含人工标注事件

---

## 🧩 四、任务函数：`EEG_events_fn(record)`

### ✅ 输入参数：

* `record`: 来自 `TUEVDataset` 的一个病人记录（dict 类型），包含：

  ```python
  {
      "load_from_path": "...",
      "signal_file": "...edf",
      "label_file": "...tse_bi",
      "patient_id": "...",
      "visit_id": "...",
      "save_to_path": "..."
  }
  ```

---

### ✅ 输出结果（每个 epoch 为一个样本）：

```python
{
    'patient_id': '0_00002265',
    'visit_id': '00000001',
    'record_id': 0,
    'epoch_path': '/path/to/0_00002265_00000001_0.pkl',
    'label': 6,                        # 1~6 类别之一
    'offending_channel': array([4.])  # 出现该事件的 EEG 通道
}
```

* `epoch_path` 指向一个 `.pkl` 文件，包含 `signal` 和 `label` 字段（即 EEG 信号和事件类型标签）

---

## 🔁 五、辅助函数（仅供底层调用）：

| 函数                                       | 说明               |
| ---------------------------------------- | ---------------- |
| `readEDF(fileName)`                      | 读取 EDF 脑电文件      |
| `convert_signals(signals, Rawdata)`      | 将 EDF 信号转换为标准格式  |
| `BuildEvents(signals, times, EventData)` | 构建每个 epoch 的事件标注 |

---

## 🧪 六、完整使用流程示例

```python
from pyhealth.datasets import TUEVDataset
from pyhealth.tasks import EEG_events_fn

# 1. 加载数据集
tuev_ds = TUEVDataset(
    root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf/",
    download=True
)

# 2. 设置事件标注任务
eeg_event_ds = tuev_ds.set_task(EEG_events_fn)

# 3. 查看样本格式
print(eeg_event_ds.samples[0])
```

---

## 📦 七、模型开发建议

由于 EEG 是时间序列数据，且为多分类问题，可以采用如下模型方案：

| 模型结构              | 适用说明                |
| ----------------- | ------------------- |
| 1D CNN            | 提取时域特征，适合短段 EEG     |
| LSTM / GRU        | 学习 EEG 的时序依赖结构      |
| Transformer       | 长时间 EEG 的建模         |
| TSCNet / SleepNet | 专为 EEG / 睡眠建模设计的结构  |
| EEGNet            | 高效建模 EEG 的通道 + 时序结构 |

---

## ✅ 八、任务总结表格

| 项目      | 内容                                         |
| ------- | ------------------------------------------ |
| 📄 任务名称 | EEG 事件类型识别                                 |
| 🧠 数据集  | TUEVDataset                                |
| 🔢 输出标签 | 1-6 类 EEG 事件                               |
| 📂 输出字段 | `epoch_path`, `label`, `offending_channel` |
| 📊 任务类型 | 多分类 Multiclass Classification              |

---

如你希望我生成完整的工程模板（含模型训练脚本 + zip 工程结构），或集成多任务脑电框架，我可以一步生成。

是否现在帮你创建？可以选择：
✅ 模型结构（CNN/LSTM/Transformer）
✅ 是否带有预测脚本 + 可视化代码

以下是对 `pyhealth.tasks.length_of_stay_prediction` 模块的完整中文学习笔记，整理了**住院时长预测任务**的目标、标签设计、数据来源、输出格式与典型用法：

---

# 🏥 PyHealth 学习笔记：住院时长预测任务 `length_of_stay_prediction`

---

## 🎯 一、任务简介

**住院时长预测（Length of Stay Prediction）**旨在预测病人本次住院时间的长度。
任务形式为**多分类任务**（Multi-class Classification），输入包括诊断信息和治疗操作。

---

## 🗂️ 二、标签设计：共 10 类分类标签

使用 `categorize_los(days)` 方法将“天数”划分为如下十类：

| 分类编号（label） | 天数范围     | 类别说明               |
| ----------- | -------- | ------------------ |
| 0           | <1 天     | 住院不足 1 天           |
| 1           | 第 1 天    | 1 ≤ LOS < 2        |
| 2           | 第 2 天    | 2 ≤ LOS < 3        |
| 3           | 第 3 天    | 3 ≤ LOS < 4        |
| 4           | 第 4 天    | 4 ≤ LOS < 5        |
| 5           | 第 5 天    | 5 ≤ LOS < 6        |
| 6           | 第 6 天    | 6 ≤ LOS < 7        |
| 7           | 第 7 天    | 7 ≤ LOS < 14（第二周内） |
| 8           | ≥14 天    | 住院超过 2 周           |
| 9           | \[可能为保留] | 未明确用途或保留的标签        |

> 📌 注意：标签为 int 类型，用于分类模型中作为 `y` 值。

---

## 📚 三、支持的数据集与接口函数

| 数据集类型     | 函数接口                                  | 示例代码模块                                        |
| --------- | ------------------------------------- | --------------------------------------------- |
| MIMIC-III | `length_of_stay_prediction_mimic3_fn` | `from pyhealth.datasets import MIMIC3Dataset` |
| MIMIC-IV  | `length_of_stay_prediction_mimic4_fn` | `from pyhealth.datasets import MIMIC4Dataset` |
| eICU      | `length_of_stay_prediction_eicu_fn`   | `from pyhealth.datasets import eICUDataset`   |
| OMOP      | `length_of_stay_prediction_omop_fn`   | `from pyhealth.datasets import OMOPDataset`   |

---

## 🧩 四、每个样本输出字段（dict）

输出是多个样本的列表，每个样本为如下格式：

```python
{
    'visit_id': '130744',
    'patient_id': '103',
    'conditions': [['42', '109', '98', ...]],    # 每次就诊的诊断（多次）
    'procedures': [['1']],                       # 每次就诊的操作
    'label': 4                                   # 住院天数分类标签（第4天）
}
```

* `conditions`: 多次就诊对应的诊断编码（可多次）
* `procedures`: 对应操作编码（如 ICD9PROC / ICD10PROC）
* `label`: 分类标签（0\~8 之间）

---

## 🧪 五、典型代码示例（以 MIMIC-IV 为例）

```python
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import length_of_stay_prediction_mimic4_fn

# 加载数据集
mimic4_base = MIMIC4Dataset(
    root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
    tables=["diagnoses_icd", "procedures_icd"],
    code_mapping={"ICD10PROC": "CCSPROC"}
)

# 设置任务
mimic4_sample = mimic4_base.set_task(length_of_stay_prediction_mimic4_fn)

# 取一个样本查看格式
print(mimic4_sample.samples[0])
```

---

## 📦 六、任务配置总结

| 项目      | 内容                              |
| ------- | ------------------------------- |
| 🎯 任务   | 住院时长预测                          |
| 🧠 输入   | 条件（diagnoses）+ 操作（procedures）   |
| 🔢 标签   | 10 类 LOS 分类标签                   |
| 📊 任务类型 | 多分类                             |
| ✅ 支持数据集 | MIMIC-III, MIMIC-IV, eICU, OMOP |
| 🧰 应用场景 | 医院床位优化、资源调度、保险成本预测              |

---

## 📌 七、模型设计建议

* **输入编码器：** 多次就诊序列 → 诊断 + 操作 → 多模态编码（Embedding + RNN / Transformer）
* **输出：** 多分类 Softmax，预测 0-8 类标签
* **评估指标：**

  * Accuracy
  * Macro-F1
  * Confusion Matrix（查看错误集中在哪些时间段）

---

是否需要我继续帮助你：

1. 使用某一数据集构建住院时长预测模型的完整工程？
2. 用 PyTorch Lightning/Sklearn 快速搭建训练 + 验证脚本？
3. 可视化不同类别 LOS 的模型性能（如雷达图）？

可直接说出你想要的场景！

以下是 `pyhealth.tasks.MedicalTranscriptionsClassification` 的中文学习笔记，适合用作掌握此任务的参考材料或工程开发说明。

---

# 📝 PyHealth 学习笔记：医疗转录分类任务 `MedicalTranscriptionsClassification`

---

## 🎯 一、任务简介

**任务目标**：将**医疗转录文本（medical transcription）**分类为对应的**医学专科（medical specialty）**。

这是一个典型的**自然语言处理（NLP）多分类任务**，输入是医疗文本，输出是该文本所属的专科类别，如“心内科”、“耳鼻喉科”、“放射科”等。

---

## 📦 二、任务属性

| 属性     | 内容                                                   |
| ------ | ---------------------------------------------------- |
| 任务类型   | 多分类                                                  |
| 输入数据类型 | 医疗转录文本（string）                                       |
| 输出标签类型 | 医学专科（string / category）                              |
| 任务模块   | `pyhealth.tasks.MedicalTranscriptionsClassification` |

---

## 🗂️ 三、Schema 定义

### 🔹 `input_schema`

```python
{
  "transcription": "text"
}
```

* `transcription`: 医疗转录内容，通常为英文自由文本。

### 🔹 `output_schema`

```python
{
  "medical_specialty": "multiclass"
}
```

* `medical_specialty`: 医学专科标签，如 cardiology、radiology 等，标签数量依赖于具体数据集（如 mtsamples）。

---

## 💾 四、数据来源说明

该任务依赖于包含医疗转录的**患者记录数据**，如 `mtsamples` 数据集，通常包含如下字段：

* `transcription`: 转录文本
* `medical_specialty`: 专科标签

PyHealth 任务函数会自动从数据集中抽取这些信息构建训练样本。

---

## 🧪 五、典型示例

```python
from pyhealth.datasets import MTSamplesDataset
from pyhealth.tasks import MedicalTranscriptionsClassification

# 初始化数据集
mtsamples_base = MTSamplesDataset(
    root="/path/to/mtsamples",  # 自定义路径
    tables=["mtsamples"],
)

# 设置分类任务
mtsamples_task = mtsamples_base.set_task(MedicalTranscriptionsClassification)

# 查看第一个样本
print(mtsamples_task.samples[0])
```

输出示例（伪造）：

```python
{
    'patient_id': '12345',
    'visit_id': 'A001',
    'transcription': 'The patient is a 67-year-old male with history of ...',
    'label': 'cardiology'
}
```

---

## 📊 六、模型设计建议

| 模块    | 建议方法                                        |
| ----- | ------------------------------------------- |
| 文本编码器 | TF-IDF、Word2Vec、BERT（推荐）                    |
| 分类器   | LogisticRegression、XGBoost、Transformer Head |
| 输出层   | 多分类 Softmax                                 |
| 评估指标  | Accuracy、Macro F1、Confusion Matrix          |

---

## 🧰 七、任务应用场景

* 医疗文档自动分发到对应科室；
* 医学文本检索与知识图谱构建；
* 医疗NLP基础任务构建与微调（Pretraining / Finetune）。

---

是否需要我帮你：

1. 构建用于此任务的 BERT 文本分类模型？
2. 可视化分类性能（如每类的准确率柱状图）？
3. 用 PyHealth 配套构建完整数据集加载 + 模型训练 + 推理脚本？

你只需告诉我目标即可！

以下是 PyHealth 框架中关于\*\*死亡预测任务（Mortality Prediction）\*\*的完整中文笔记，覆盖了多个数据集（MIMIC-III / MIMIC-IV / eICU / OMOP）及模态（结构化、多模态）版本：

---

# 💀 PyHealth 死亡预测任务笔记（`pyhealth.tasks.mortality_prediction`）

---

## 📌 一、任务目标概述

> **任务目标**：基于当前一次住院或 ICU 就诊信息，预测患者在下一次住院前是否会死亡。

* **任务类型**：二分类（`mortality: 0/1`）
* **输入数据**：结构化数据（诊断、药物、操作）+ 可选的文本与图像（临床笔记、胸片等）
* **输出标签**：死亡标记（`mortality`），`1` 表示死亡，`0` 表示未死亡

---

## 🧩 二、各数据集任务类概览

| 类名                                    | 数据源       | 模态  | 输入类型                                                   | 输出     |
| ------------------------------------- | --------- | --- | ------------------------------------------------------ | ------ |
| `MortalityPredictionMIMIC3`           | MIMIC-III | 结构化 | conditions, drugs, procedures                          | binary |
| `MultimodalMortalityPredictionMIMIC3` | MIMIC-III | 多模态 | 上述 + clinical\_notes                                   | binary |
| `MortalityPredictionMIMIC4`           | MIMIC-IV  | 结构化 | conditions, drugs, procedures                          | binary |
| `MultimodalMortalityPredictionMIMIC4` | MIMIC-IV  | 多模态 | 上述 + image\_paths, discharge, radiology, xrays\_negbio | binary |
| `MortalityPredictionEICU`             | eICU      | 结构化 | conditions, drugs, procedures（ICD表）                    | binary |
| `MortalityPredictionEICU2`            | eICU      | 结构化 | conditions, procedures（替代编码）                           | binary |
| `MortalityPredictionOMOP`             | OMOP      | 结构化 | conditions, drugs, procedures                          | binary |

---

## 🔍 三、输入输出 Schema 对比

| 类别           | 输入字段（`input_schema`）                                         | 输出字段（`output_schema`） |
| ------------ | ------------------------------------------------------------ | --------------------- |
| MIMIC3       | `conditions`, `drugs`, `procedures`                          | `mortality`           |
| MIMIC3 (多模态) | 上述 + `clinical_notes`                                        | `mortality`           |
| MIMIC4       | `conditions`, `drugs`, `procedures`                          | `mortality`           |
| MIMIC4 (多模态) | 上述 + `discharge`, `radiology`, `image_paths`, `xrays_negbio` | `mortality`           |
| EICU         | `conditions`, `drugs`, `procedures`                          | `mortality`           |
| EICU2        | `conditions`, `procedures`                                   | `mortality`           |
| OMOP         | `conditions`, `drugs`, `procedures`                          | `mortality`           |

---

## 🧪 四、示例代码（以 MIMIC-IV 多模态为例）

```python
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.mortality_prediction import MultimodalMortalityPredictionMIMIC4

# 初始化数据集
mimic4_base = MIMIC4Dataset(
    root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
    tables=["diagnoses_icd", "procedures_icd", "note", "imaging"],
)

# 配置任务
mortality_task = mimic4_base.set_task(MultimodalMortalityPredictionMIMIC4)

# 查看样本
mortality_task.samples[0]
```

输出格式：

```python
{
    'patient_id': '123',
    'visit_id': 'A100',
    'conditions': [...],
    'drugs': [...],
    'procedures': [...],
    'discharge': 'Patient was stable...',
    'radiology': 'Chest x-ray showed...',
    'image_paths': '/path/to/image.jpg',
    'xrays_negbio': [...],
    'label': 0  # 1 = deceased, 0 = survived
}
```

---

## 🧠 五、建模建议

| 组件    | 方法建议                              |
| ----- | --------------------------------- |
| 结构化编码 | Embedding / RNN / Transformer     |
| 文本编码  | ClinicalBERT / BioBERT            |
| 图像编码  | ResNet / EfficientNet             |
| 多模态融合 | 加权拼接 / Attention                  |
| 输出层   | Sigmoid + Binary Cross-Entropy    |
| 评估指标  | AUC-ROC、F1-score、Precision/Recall |

---

## 🧰 六、推荐使用场景

* ICU 预警与病情评估
* 多模态 EHR 风险预测
* 多数据源预后建模（结构化+非结构化）
* 医疗 NLP 与医疗图像融合应用开发

---

如果你需要我帮你：

* 编写用于多模态死亡预测的模型训练脚本
* 制作任务对比表格或可视化图表（如死亡风险分布）
* 进行任务的特征重要性分析（如 SHAP）

请直接告诉我需求即可！

以下是对 PyHealth 中 `pyhealth.tasks.patient_linkage_mimic3_fn` 的中文整理笔记：

---

# 🔗 `pyhealth.tasks.patient_linkage_mimic3_fn`

**任务名称**：患者链接任务（Patient Linkage）
**适用数据集**：MIMIC-III

---

## 📌 一、任务概述

**Patient Linkage Task** 的目标是：

> **判断两条病历记录是否来自于同一个患者。**

这在临床数据融合、重复记录检测、身份匹配等任务中非常关键。

---

## 🧩 二、任务特点

| 属性   | 描述                         |
| ---- | -------------------------- |
| 数据来源 | MIMIC-III 数据集              |
| 输入类型 | 成对病历记录的特征（如条件、药物、操作等）      |
| 输出标签 | `linkage`：是否是同一患者，二分类（0/1） |
| 用途   | 多源数据合并、身份解析、跨医院患者识别等       |

---

## 🗃️ 三、输出样本结构（推测）

虽然官方文档未明确列出 schema，但根据命名规则和常规任务结构，任务样本很可能是如下格式：

```python
{
    "patient_id_1": "101",
    "visit_id_1": "A100",
    "features_1": {...},

    "patient_id_2": "102",
    "visit_id_2": "B205",
    "features_2": {...},

    "label": 0  # 是否为同一患者，1 表示是，0 表示否
}
```

---

## 🧪 四、使用示例（伪代码）

```python
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import patient_linkage_mimic3_fn

# 构造数据集
mimic3_base = MIMIC3Dataset(
    root="/path/to/mimiciii/",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD"],
)

# 设置任务
mimic3_linkage_task = mimic3_base.set_task(patient_linkage_mimic3_fn)

# 查看样本
print(mimic3_linkage_task.samples[0])
```

---

## 🧠 五、建模建议

| 模块    | 建议方法                                     |
| ----- | ---------------------------------------- |
| 特征处理  | 多模态编码（诊断、药物、文本）拼接                        |
| 相似性建模 | Siamese Network / Cross-Encoder          |
| 输出层   | Sigmoid + BCE Loss                       |
| 评估指标  | Accuracy / AUC / F1 / Precision / Recall |

---

## 🎯 六、典型应用场景

* 多医院病历合并
* 异构数据库患者识别
* 患者身份去重任务（Patient De-duplication）

---

如果你希望我帮你：

* 构建样例模型（如 Siamese 架构）
* 补全任务的 schema 或封装函数
* 拓展至 MIMIC-IV、OMOP 或其他数据库

欢迎随时告诉我！


以下是对 PyHealth 中 `pyhealth.tasks.readmission_prediction` 模块的中文系统整理笔记：

---

# 🔁 `pyhealth.tasks.readmission_prediction`

## 📌 任务名称：**再入院预测（Readmission Prediction）**

---

## 🧠 一、任务目标

预测某次住院出院后的 **时间窗口内（如15天、5天）**，该患者是否会再次入院。

* **任务类型**：二分类任务（binary classification）

  * `label = 1` 表示：在指定时间窗口内有再次入院
  * `label = 0` 表示：没有再次入院

---

## 🧩 二、通用输入输出结构

| 字段名         | 类型   | 描述           |
| ----------- | ---- | ------------ |
| patient\_id | str  | 患者唯一标识       |
| visit\_id   | str  | 本次住院就诊标识     |
| conditions  | list | 诊断码序列（如 ICD） |
| procedures  | list | 手术/操作码序列（可选） |
| drugs       | list | 药物编码序列（部分任务） |
| label       | int  | 是否再入院（1/0）   |

---

## 📚 三、子任务函数列表（按数据集划分）

| 函数名称                               | 数据集       | 默认时间窗口 | 特殊说明                                     |
| ---------------------------------- | --------- | ------ | ---------------------------------------- |
| `readmission_prediction_mimic3_fn` | MIMIC-III | 15 天   |                                          |
| `readmission_prediction_mimic4_fn` | MIMIC-IV  | 15 天   |                                          |
| `readmission_prediction_eicu_fn`   | eICU      | 5 天    | 使用 diagnosis, medication, physicalExam 表 |
| `readmission_prediction_eicu_fn2`  | eICU      | 5 天    | 使用 admissionDx, treatment 表              |
| `readmission_prediction_omop_fn`   | OMOP-CDM  | 15 天   | 支持通用 CDM 数据                              |

> 💡 `time_window` 参数可指定预测时间窗，例如 7/15/30 天。

---

## 🔍 四、示例代码：MIMIC-IV

```python
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import readmission_prediction_mimic4_fn

# 加载 MIMIC-IV 数据
mimic4_base = MIMIC4Dataset(
    root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
    tables=["diagnoses_icd", "procedures_icd"],
    code_mapping={"ICD10PROC": "CCSPROC"},
)

# 配置再入院预测任务
mimic4_sample = mimic4_base.set_task(readmission_prediction_mimic4_fn)

# 查看一个样本
print(mimic4_sample.samples[0])
```

输出示例：

```python
{
  "visit_id": "130744",
  "patient_id": "103",
  "conditions": [['42', '109', '19', '122', '98', '663', '58', '51']],
  "procedures": [['1']],
  "label": 0
}
```

---

## 🏥 五、典型应用场景

| 应用场景   | 描述            |
| ------ | ------------- |
| 临床风险评估 | 提前识别高再入院风险患者  |
| 资源优化   | 辅助出院管理和康复干预策略 |
| 医保分析   | 精算再入院率、控制成本   |

---

## 🛠️ 六、建模建议

| 模块   | 推荐方法                            |
| ---- | ------------------------------- |
| 特征编码 | 多热编码、词向量、embedding              |
| 模型结构 | MLP / BiGRU / Transformer       |
| 增强方式 | 加入药物特征、实验室检查                    |
| 评估指标 | AUC / F1 / Precision / Recall 等 |

---

如需：

* 创建完整任务流程模板
* 自定义 `time_window` 预测
* 对比多个数据集在再入院预测上的表现

请告诉我，我可以为你补充代码或可视化支持。

以下是 PyHealth 中 `pyhealth.tasks.sleep_staging` 睡眠分期任务的中文系统整理笔记，覆盖了对三个主要睡眠数据集（ISRUC、Sleep-EDF、SHHS）的支持：

---

# 😴 `pyhealth.tasks.sleep_staging` 睡眠分期任务整理笔记

---

## 🧠 一、任务简介：Sleep Staging

* **目标**：预测 EEG 片段所对应的睡眠阶段
* **任务类型**：多分类（multi-class classification）
* **输出类别**：

  * Awake（清醒）
  * N1、N2、N3（非快速眼动睡眠阶段）
  * REM（快速眼动）
  * （SleepEDF 中还有 N4，一般与 N3 合并）

---

## 🧩 二、通用输入输出格式

| 字段名          | 类型      | 描述                  |
| ------------ | ------- | ------------------- |
| `patient_id` | str     | 患者 ID               |
| `record_id`  | str     | 本次记录的唯一标识           |
| `epoch_path` | str     | `.pkl` 文件路径，包含信号和标签 |
| `label`      | str/int | 睡眠阶段标签（字符或数字编码）     |

---

## 📚 三、支持的数据集任务函数

| 函数名称                        | 数据集               | 支持标签集                      | 默认 Epoch 长度        | 标签选择参数            |
| --------------------------- | ----------------- | -------------------------- | ------------------ | ----------------- |
| `sleep_staging_isruc_fn`    | `ISRUCDataset`    | Awake, N1, N2, N3, REM     | `epoch_seconds=10` | `label_id=1`（专家1） |
| `sleep_staging_sleepedf_fn` | `SleepEDFDataset` | Awake, REM, N1, N2, N3, N4 | `epoch_seconds=30` | 无                 |
| `sleep_staging_shhs_fn`     | `SHHSDataset`     | Awake, N1, N2, N3, REM     | `epoch_seconds=30` | 无                 |

---

## 🛠️ 四、使用示例代码（以 SHHS 为例）

```python
from pyhealth.datasets import SHHSDataset
from pyhealth.tasks import sleep_staging_shhs_fn

# 加载数据集
shhs = SHHSDataset(
    root="/srv/local/data/SHHS/polysomnography",
    dev=True
)

# 设置任务
shhs_ds = shhs.set_task(sleep_staging_shhs_fn)

# 查看样本
print(shhs_ds.samples[0])
```

输出示例：

```python
{
  'record_id': 'shhs1-200001-0',
  'patient_id': 'shhs1-200001',
  'epoch_path': '/.../shhs1-200001-0.pkl',
  'label': '0'  # 数字编码代表具体睡眠阶段
}
```

---

## 🧪 五、细节参数说明

| 参数              | 说明                                              |
| --------------- | ----------------------------------------------- |
| `epoch_seconds` | 每个 epoch 的长度（单位：秒），建议设为 30 或其因数（如 10, 15）       |
| `label_id`      | （仅 ISRUCDataset 支持）表示使用哪一位专家的标签，1 表示专家1，2 表示专家2 |

---

## 🧮 六、建模建议

| 步骤    | 建议方法                                             |
| ----- | ------------------------------------------------ |
| 数据预处理 | 滤波、标准化、多通道拼接                                     |
| 模型结构  | CNN/LSTM/Transformer 或 CNN-BiLSTM 混合             |
| 评估指标  | Accuracy / Cohen’s kappa / F1 / Confusion Matrix |
| 可视化   | Hypnogram 可视化、信号波形展示                             |

---

如需添加：

* 具体的数据加载与可视化代码
* 每个睡眠阶段对应的标签数值定义
* 模型训练流程示例（如 PyTorch）

欢迎继续提问，我可以逐步为你扩展。

在 `PyHealth` 中，`pyhealth.tasks.SleepStagingSleepEDF` 是用于 **Sleep-EDF 数据集的睡眠分期任务** 封装类，它是对底层函数 `sleep_staging_sleepedf_fn` 的高级封装。下面是该任务类的详细结构整理：

---

# 💤 `pyhealth.tasks.SleepStagingSleepEDF` 使用笔记

---

## 📌 类简介

```python
class pyhealth.tasks.SleepStagingSleepEDF(BaseTask)
```

* **功能**：基于 [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) 数据集，进行睡眠分期任务。
* **任务类型**：多分类任务（multi-class classification）
* **数据输入**：多通道 EEG 信号
* **预测目标**：判断每个 epoch 属于哪一个睡眠阶段（如 W、N1、N2、N3、REM）

---

## 🧾 输入输出 schema

| 属性名             | 类型               | 描述                               |
| --------------- | ---------------- | -------------------------------- |
| `task_name`     | `str`            | 任务名，固定为 `"SleepStagingSleepEDF"` |
| `input_schema`  | `Dict[str, str]` | `{ "epoch_path": "signal" }`     |
| `output_schema` | `Dict[str, str]` | `{ "label": "multiclass" }`      |

---

## 📦 示例样本格式（来自 `dataset.samples[0]`）

```python
{
    'record_id': 'SC4001-0',
    'patient_id': 'SC4001',
    'epoch_path': '/your/local/path/SC4001-0.pkl',
    'label': 'W'  # 睡眠阶段标签
}
```

---

## 🧪 使用流程（示例代码）

```python
from pyhealth.datasets import SleepEDFDataset
from pyhealth.tasks import SleepStagingSleepEDF

# 加载 Sleep-EDF 数据集
sleepedf = SleepEDFDataset(
    root="/path/to/sleep-edf-database-expanded-1.0.0/sleep-cassette"
)

# 设置睡眠分期任务
sleepedf_task = sleepedf.set_task(SleepStagingSleepEDF())

# 查看一个样本
print(sleepedf_task.samples[0])
```

---

## 🧠 标签定义（通常情况）

Sleep-EDF 数据集中标签可能为：

| 标签字符           | 描述            |
| -------------- | ------------- |
| `'W'`          | 清醒（Wake）      |
| `'R'`          | REM 睡眠        |
| `'1'`          | N1 阶段         |
| `'2'`          | N2 阶段         |
| `'3'` or `'4'` | N3/N4，可能合并为深睡 |

> 🚨 注意：标签格式具体依赖于处理方式和 `sleep_staging_sleepedf_fn` 的实现，如使用标准 AASM 分级建议合并 N3/N4。

---

## 🔧 进阶：修改 epoch 长度或自定义处理逻辑

如果你希望手动控制 epoch 长度或 label 映射方式，可使用底层函数：

```python
from pyhealth.tasks.sleep_staging import sleep_staging_sleepedf_fn
sleepedf.set_task(lambda record: sleep_staging_sleepedf_fn(record, epoch_seconds=30))
```

---

如你需要构建模型、可视化 EEG、进行模型评估（如混淆矩阵或 Cohen’s Kappa），我可以帮你进一步扩展。是否需要我接下来帮你写一个完整的模型训练 pipeline？

你查阅的是 `pyhealth.tasks.temple_university_EEG_tasks` 模块，它为 **TUAB/TUEV EEG 数据集** 提供了两个核心任务函数，分别用于：

1. TUAB 的异常脑电检测（二分类）
2. TUEV 的脑电事件注释（多分类）

下面我为你总结每个函数的核心功能、使用方式和结构，便于快速理解与实战应用：

---

## 📍 模块路径：

```python
pyhealth.tasks.temple_university_EEG_tasks
```

---

## 1️⃣ EEG 异常检测任务：`EEG_isAbnormal_fn`

### ✅ 任务说明：

用于 TUAB 数据集的 **异常 EEG 检测（二分类）**。目标是判断给定的 EEG 是否异常。

### ✅ 输入格式：

* 单个 `record` 字典（来自 `TUABDataset`），包含字段：

  * `load_from_path`
  * `patient_id`
  * `visit_id`
  * `signal_file`
  * `label_file`
  * `save_to_path`

### ✅ 输出格式：

每个输出 `sample` 是一个字典，格式如下：

```python
{
    'patient_id': 'aaaaamye',
    'visit_id': 's001',
    'record_id': '1',
    'epoch_path': '/.../0.pkl',  # 包含signal与label的保存路径
    'label': 1                   # 1表示异常，0表示正常
}
```

### ✅ 示例代码：

```python
from pyhealth.datasets import TUABDataset
from pyhealth.tasks import EEG_isAbnormal_fn

dataset = TUABDataset(root="/path/to/tuh_eeg_abnormal/v3.0.0/edf/", download=True)
abnormal_task = dataset.set_task(EEG_isAbnormal_fn)
print(abnormal_task.samples[0])
```

---

## 2️⃣ EEG 事件分类任务：`EEG_events_fn`

### ✅ 任务说明：

用于 TUEV 数据集的 **脑电事件分类（6类多分类）**。任务目标是标注 EEG 中的事件类型。

### ✅ 分类标签：

* (1) SPSW: spike and sharp wave
* (2) GPED: generalized periodic epileptiform discharges
* (3) PLED: periodic lateralized epileptiform discharges
* (4) EYEM: eye movement
* (5) ARTF: artifact
* (6) BCKG: background

### ✅ 输入格式：

* 单个 `record` 字典（来自 `TUEVDataset`），字段同上。

### ✅ 输出格式：

```python
{
    'patient_id': '0_00002265',
    'visit_id': '00000001',
    'record_id': 0,
    'epoch_path': '/.../0.pkl',
    'label': 6,                        # 标签数字，对应事件类型
    'offending_channel': array([4.])  # 引发该事件的通道索引
}
```

### ✅ 示例代码：

```python
from pyhealth.datasets import TUEVDataset
from pyhealth.tasks import EEG_events_fn

dataset = TUEVDataset(root="/path/to/tuh_eeg_events/v2.0.0/edf/", download=True)
events_task = dataset.set_task(EEG_events_fn)
print(events_task.samples[0])
```

---

## 🔧 附加函数（内部工具）

| 函数名                                      | 功能                |
| ---------------------------------------- | ----------------- |
| `BuildEvents(signals, times, EventData)` | 构建标注事件，结合信号和时间信息  |
| `convert_signals(signals, Rawdata)`      | 转换原始 EEG 信号格式     |
| `readEDF(fileName)`                      | 从 EDF 文件读取 EEG 数据 |

> 这些函数主要用于底层 signal-label 的对齐和格式转换，可用于自定义任务构建。

---

如你下一步希望使用这些 EEG 数据构建模型（如 CNN/LSTM/Transformer），我可以帮你写出预处理 pipeline 和模型训练代码。需要继续吗？

















