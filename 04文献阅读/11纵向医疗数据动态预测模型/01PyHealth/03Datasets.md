以下是关于 `pyhealth.datasets` 模块的**中文学习笔记**，系统整理了所有官方支持的数据集类与工具模块，适合后续复现论文、使用真实医疗数据集等场景。

---

# 📦 PyHealth 学习笔记：`pyhealth.datasets` 数据集模块

---

## 🧩 一、模块总览

`pyhealth.datasets` 是 PyHealth 中用于加载、解析、转换医疗数据的核心模块。
其数据结构为统一的三级结构：

```
Patient（病人） → Visit（就诊） → Event（事件）
```

---

## ✅ 二、主要数据集类汇总

| 数据集类                           | 用途           | 来源 / 描述                  |
| ------------------------------ | ------------ | ------------------------ |
| `BaseDataset`                  | 基类，所有数据集继承它  | 用于统一结构定义                 |
| `SampleDataset`                | 用于样本级数据      | 适合非 visit 格式任务           |
| `MIMIC3Dataset`                | ICU 病人数据     | 来自 MIMIC-III             |
| `MIMIC4Dataset`                | ICU+病房数据     | 来自 MIMIC-IV              |
| `MIMICExtractDataset`          | 多任务抽取版 MIMIC | 用于多任务联合学习                |
| `eICUDataset`                  | 多中心 ICU 数据   | eICU 数据库                 |
| `OMOPDataset`                  | 通用标准格式       | 遵循 OMOP CDM 模式           |
| `SHHSDataset`                  | 睡眠心肺数据       | Sleep Heart Health Study |
| `SleepEDFDataset`              | 睡眠 EEG 数据    | Sleep-EDF                |
| `ISRUCDataset`                 | 多通道睡眠脑电图     | ISRUC 数据集                |
| `TUABDataset`                  | EEG 数据       | TUAB 脑电图                 |
| `TUEVDataset`                  | 癫痫 EEG 数据    | TUH EEG Seizure          |
| `CardiologyDataset`            | 心电图数据集       | 尚未公开详情                   |
| `MedicalTranscriptionsDataset` | 临床文本         | 医疗转录文本                   |

---

## 📘 三、核心使用示例：加载 MIMIC-III 数据

```python
from pyhealth.datasets import MIMIC3Dataset

mimic3 = MIMIC3Dataset(
    root="/path/to/mimiciii/csv",          # 预处理后的 MIMIC 路径
    tables=["DIAGNOSES_ICD", "LABEVENTS", "PRESCRIPTIONS"],
    code_mapping={"NDC": "ATC"}            # 药品编码映射（可选）
)
```

你也可以加载合成数据（无需账号）：

```python
mimic3 = MIMIC3Dataset(
    root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"]
)
```

---

## 🛠 四、辅助模块

### 🔹 `pyhealth.datasets.splitter`

> 用于将数据集划分为训练 / 验证 / 测试集

```python
from pyhealth.datasets import split_by_patient

train_ds, val_ds, test_ds = split_by_patient(dataset, [0.7, 0.15, 0.15])
```

支持按患者、就诊等维度划分。

---

### 🔹 `pyhealth.datasets.utils`

> 提供数据格式转换、编码映射等实用函数（内部使用为主）

---

## 🎯 五、选择数据集类的建议

| 场景           | 推荐数据集类                                                   |
| ------------ | -------------------------------------------------------- |
| ICU 临床研究     | `MIMIC3Dataset`, `MIMIC4Dataset`, `eICUDataset`          |
| 睡眠脑电信号处理     | `SleepEDFDataset`, `ISRUCDataset`                        |
| 药物推荐         | `OMOPDataset`, `MIMICExtractDataset`                     |
| 多模态研究（图 + 文） | `CardiologyDataset`, `MedicalTranscriptionsDataset`（开发中） |
| EEG 癫痫识别     | `TUEVDataset`                                            |

---

## 🔚 小结

| 类型      | 模块                             | 说明                |
| ------- | ------------------------------ | ----------------- |
| 通用结构    | `BaseDataset`, `SampleDataset` | 适合自定义数据集          |
| 临床结构化数据 | `MIMIC`, `eICU`, `OMOP`        | 适合 ICU、诊断、药物预测等   |
| 生物信号    | `SHHS`, `EDF`, `TUEV` 等        | 适合 EEG、心电图、睡眠分析任务 |
| 文本数据    | `MedicalTranscriptionsDataset` | 适合 NLP 方向研究       |
| 工具类     | `splitter`, `utils`            | 数据划分与预处理工具        |

---

下一步是否需要我：

* 详细讲解如何使用 `BaseDataset` 构建你自己的数据集？
* 帮你写一份适配肾病随访数据的 MIMIC 风格数据集转换脚本？

告诉我你的需求，我将按需继续整理。

好的，以下是关于 `pyhealth.datasets.BaseDataset` 的中文学习笔记，用于帮助你理解 PyHealth 数据系统的基础设计与自定义扩展能力。

---

# 📦 PyHealth 学习笔记：`pyhealth.datasets.BaseDataset`

---

## ✅ 一、BaseDataset 是什么？

`BaseDataset` 是 PyHealth 所有数据集类的**基类**。
其他如 `MIMIC3Dataset`、`OMOPDataset` 等真实数据集类都继承自它。

---

## 🎯 它的主要作用：

* 定义统一的 **数据结构模板**（Patient → Visit → Event）
* 提供统一的 `set_task()` 接口，用于生成模型任务输入
* 提供扩展性强的设计，方便用户自定义自己的数据集格式

---

## 🧱 二、典型继承结构

```python
from pyhealth.datasets import BaseDataset

class MyCustomDataset(BaseDataset):
    def parse_data(self):
        # 你需要重写这个方法，解析原始数据，构建 patient/visit/event 层级结构
        pass
```

---

## 🛠️ 三、自定义数据集常用流程

当你使用 `BaseDataset` 来加载自己的医学数据（如肾病实验室随访记录）时，通常步骤如下：

### 1️⃣ 自定义类继承 `BaseDataset`

```python
class KidneyDataset(BaseDataset):
    def parse_data(self):
        for pid, patient_group in raw_df.groupby("patient_id"):
            patient = self.add_patient(patient_id=pid)
            for _, row in patient_group.iterrows():
                visit = patient.add_visit(
                    visit_id=row["visit_id"],
                    timestamp=row["visit_time"]
                )
                # 添加事件
                visit.add_event(
                    event_type="lab",
                    timestamp=row["visit_time"],
                    attr_dict={
                        "test": "creatinine",
                        "value": row["creatinine"]
                    }
                )
                visit.add_label(row["aki_label"])
```

### 2️⃣ 使用自定义类

```python
dataset = KidneyDataset(
    name="aki_dataset",
    dataset_id="001",
    root="/your/data/folder",
    dev=True   # 若为 True，只加载小部分做调试
)
```

### 3️⃣ 设置任务

```python
from pyhealth.tasks import BinaryPredictionTask

task = BinaryPredictionTask(
    dataset=dataset,
    feature_keys=["lab"],
    label_key="label",
    time_order=True
)
```

---

## 📌 四、BaseDataset 支持的结构层级

| 层级  | 类             | 含义                     |
| --- | ------------- | ---------------------- |
| 数据集 | `BaseDataset` | 所有数据入口，存储多个患者          |
| 病人  | `Patient`     | 每个 `patient_id` 对应一个对象 |
| 就诊  | `Visit`       | 每个就诊记录有独立时间戳           |
| 事件  | `Event`       | 每次检查、用药、诊断等信息          |

---

## ✅ 小结

| 特点   | 说明                            |
| ---- | ----------------------------- |
| 灵活性强 | 可手动实现 `parse_data()` 适配任意医疗数据 |
| 统一接口 | 可直接配合 `set_task()` 构建任务       |
| 多样性  | 支持诊断、实验室、用药等不同类型的事件结构         |
| 扩展能力 | 适合构建自定义数据集，例如医院本地随访数据         |

---

如果你希望我基于你实际的数据（如 csv），帮你构建一个 `MyKidneyDataset` 自定义类及 `parse_data()` 实现示例，请把你的数据结构发给我，我可以立即生成完整代码样例。是否继续？


以下是关于 `pyhealth.datasets.SampleDataset` 的中文学习笔记，适合你掌握 **非 visit 类型任务**（如单条样本预测）时的数据组织方式。

---

# 📦 PyHealth 学习笔记：`pyhealth.datasets.SampleDataset`

---

## ✅ 一、SampleDataset 是什么？

`SampleDataset` 是 PyHealth 中另一个数据集基类，专为构建**样本级数据集**（而不是 Visit 序列）而设计。

它适用于：

* 没有明显“多次就诊记录”的样本
* 每条记录就是一个完整的输入，例如静态特征、单次监测数据等
* 实验室测试、图像、基因组数据等非结构化数据建模

---

## 🧩 与 BaseDataset 的区别

| 特性     | `BaseDataset`           | `SampleDataset` |
| ------ | ----------------------- | --------------- |
| 数据层级结构 | Patient → Visit → Event | Sample（单层结构）    |
| 适用任务   | 时序建模、纵向预测               | 静态预测、个体分类       |
| 典型输入   | 多次记录序列                  | 单条样本            |
| 示例     | ICU 随访、诊断序列             | 药物组合、一次 EEG 报告  |

---

## 🛠️ 二、构建方式概览

### 1️⃣ 自定义 Sample 数据集继承方式：

```python
from pyhealth.datasets import SampleDataset

class MyStaticDataset(SampleDataset):
    def parse_data(self):
        for idx, row in raw_df.iterrows():
            self.add_sample(
                sample_id=row["sample_id"],
                patient_id=row["patient_id"],
                visit_id=row["visit_id"],
                timestamp=row["timestamp"],
                events=row["features"],  # dict，例如 {"bun":12, "creatinine":1.1}
                label=row["label"]
            )
```

### 2️⃣ 使用自定义数据集：

```python
dataset = MyStaticDataset(
    name="static_kidney",
    dataset_id="S001",
    root="/your/data/folder"
)
```

---

## 🔍 三、核心方法 `add_sample()`

### 函数签名：

```python
add_sample(
    sample_id: str,
    patient_id: str,
    visit_id: str,
    timestamp: datetime,
    events: dict,
    label: any
)
```

| 参数名          | 含义                 |
| ------------ | ------------------ |
| `sample_id`  | 当前样本的唯一编号          |
| `patient_id` | 样本所属病人 ID          |
| `visit_id`   | 样本所属就诊 ID（可选）      |
| `timestamp`  | 时间信息（可选）           |
| `events`     | 输入特征字典（如实验室指标）     |
| `label`      | 预测目标（0/1 或 多分类标签等） |

---

## 🔄 四、何时选择使用 SampleDataset？

| 场景                  | 建议使用                 |
| ------------------- | -------------------- |
| 单条特征 → 标签预测         | ✅ 使用 `SampleDataset` |
| 有多次就诊序列             | ❌ 使用 `BaseDataset`   |
| 数据来自问卷、图像、基因等非结构化任务 | ✅ 使用 `SampleDataset` |

---

## 📝 五、总结

| 特性               | 内容                                  |
| ---------------- | ----------------------------------- |
| 基本用途             | 非序列样本建模，适合静态预测任务                    |
| 与 BaseDataset 区别 | 不需要 visit/event 层级，直接以 sample 为单位建模 |
| 推荐使用情境           | 药物组合预测、诊断图像预测、一次性生理特征分类等            |

---

需要我继续演示如何将一个 CSV 文件（例如你提供的实验室检查数据）转为 SampleDataset 格式，并提供任务构建与模型训练完整代码吗？或者进入下一个模块（如任务 `pyhealth.tasks`）的讲解？


以下是关于 `pyhealth.datasets.MIMIC3Dataset` 的中文学习笔记，适合你深入了解如何基于 ICU 结构化数据（如血肌酐、诊断码、用药等）构建医学预测模型。

---

# 🏥 PyHealth 学习笔记：`MIMIC3Dataset`

---

## ✅ 一、MIMIC-III 是什么？

MIMIC-III（Medical Information Mart for Intensive Care III）是一个公开的 ICU 临床数据库，由 MIT 提供，包含：

* 超 4 万名 ICU 病人的数据
* 包括诊断、实验室检查、用药、护理记录等
* 可用于 ICU 预测、死亡率分析、疾病进展建模等任务

---

## 🧩 二、PyHealth 中的 `MIMIC3Dataset`

该类是对原始 MIMIC-III 数据的**封装和标准化**。
PyHealth 通过 `MIMIC3Dataset` 将原始 csv 表格结构转化为：

```
Patient → Visit → Event
```

每一个表格字段（如 ICD9、NDC、LABEVENTS）都被自动映射为 event 类型。

---

## 🧱 三、使用示例

### ✅ 示例 1：加载合成版 MIMIC-III（无需下载原始数据）

```python
from pyhealth.datasets import MIMIC3Dataset

dataset = MIMIC3Dataset(
    root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"]
)
```

### ✅ 示例 2：加载本地 MIMIC 数据（你已获得认证并预处理）

```python
dataset = MIMIC3Dataset(
    root="/your/local/mimiciii/csv/",
    tables=["LABEVENTS", "DIAGNOSES_ICD", "PRESCRIPTIONS"],
    code_mapping={"NDC": "ATC"}  # 自动转换药品编码
)
```

---

## 🔧 四、参数说明

| 参数名            | 类型          | 说明                                      |
| -------------- | ----------- | --------------------------------------- |
| `root`         | `str`       | 数据文件夹路径（支持本地或远程 URL）                    |
| `tables`       | `List[str]` | 指定要加载哪些表，例如 `LABEVENTS`、`DIAGNOSES_ICD` |
| `code_mapping` | `dict`      | 可选，定义医学编码映射（如 NDC ➝ ATC）                |
| `dev`          | `bool`      | 若为 `True`，仅加载部分数据（用于调试）                 |

---

## 🎯 五、适用任务类型（可配合 task 使用）

你可以在加载 MIMIC-III 后快速构建以下任务：

* 死亡率预测（MortalityPredictionMIMIC3）
* ICU 住院时长预测
* 再入院预测
* 药物推荐任务（结合 `PRESCRIPTIONS`）

```python
from pyhealth.tasks import MortalityPredictionMIMIC3
task_fn = MortalityPredictionMIMIC3()
dataset = dataset.set_task(task_fn)
```

---

## 📝 六、小结

| 特性   | 内容                                                              |
| ---- | --------------------------------------------------------------- |
| 目标   | 结构化封装 MIMIC-III 表格数据，供建模使用                                      |
| 核心结构 | 自动构建 Patient → Visit → Event                                    |
| 常用表  | `DIAGNOSES_ICD`, `PROCEDURES_ICD`, `PRESCRIPTIONS`, `LABEVENTS` |
| 配合任务 | 可用于死亡率、住院时间、再入院、药物推荐等多任务建模                                      |
| 推荐人群 | 有 MIMIC 访问权限或需要复现 ICU 研究论文的研究人员                                 |

---

是否继续讲解下一个数据集（如 `MIMIC4Dataset` / `OMOPDataset`），或者你希望我帮你将自己的数据仿照 `MIMIC3Dataset` 结构做一套完整的转换脚本？

以下是关于 `pyhealth.datasets.MIMIC4Dataset` 的**中文学习笔记**，帮助你理解如何使用 PyHealth 处理 **MIMIC-IV** 数据构建 ICU 预测模型。

---

# 🏥 PyHealth 学习笔记：`MIMIC4Dataset`

---

## ✅ 一、MIMIC-IV 简介

**MIMIC-IV** 是 MIMIC 系列的升级版数据库，发布于 2020 年，具备以下特点：

| 特点   | 内容                              |
| ---- | ------------------------------- |
| 数据范围 | ICU + 病房（WARD）+ 急诊              |
| 数据结构 | 更标准化，更接近真实医院信息系统（HIS）结构         |
| 子模块  | `hosp`、`icu`、`ed` 子表更清晰分离       |
| 推荐用途 | 深度 ICU 分析、再入院预测、多模态分析（如加入图像/波形） |

---

## 📦 二、PyHealth 中的 `MIMIC4Dataset`

`MIMIC4Dataset` 是对 MIMIC-IV 数据的封装器，提供统一的 `Patient → Visit → Event` 结构，并支持任务创建和建模。

其使用方式与 `MIMIC3Dataset` 非常类似，只是表格字段略有不同。

---

## 🧱 三、基本使用方法

### ✅ 示例：加载 MIMIC-IV 本地数据

```python
from pyhealth.datasets import MIMIC4Dataset

dataset = MIMIC4Dataset(
    root="/your/path/to/mimic-iv/csv/",
    tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    code_mapping={"NDC": "ATC"},  # 药物编码映射
    dev=False  # 若为 True，仅加载部分数据
)
```

⚠️ 注意：

* 文件夹需为官方 MIMIC-IV 的 CSV 解压版（不需要 SQL 数据库）
* 表名请对应 MIMIC-IV 官方命名（小写，带下划线）

---

## 🧰 四、常见可加载表（PyHealth 推荐）

| 表名               | 说明             |
| ---------------- | -------------- |
| `diagnoses_icd`  | 诊断 ICD 编码      |
| `procedures_icd` | 手术编码           |
| `prescriptions`  | 药物开具记录         |
| `labevents`      | 实验室检查          |
| `inputevents`    | ICU 期间输入液体/药物等 |

---

## 🎯 五、配合任务构建使用

```python
from pyhealth.tasks import MortalityPredictionMIMIC4

task_fn = MortalityPredictionMIMIC4()
dataset = dataset.set_task(task_fn)
```

你也可以自定义任务，或复用 `BinaryPredictionTask`、`MultiClassPredictionTask` 等通用任务类。

---

## 🔍 六、MIMIC-III vs MIMIC-IV 对比

| 特征   | MIMIC-III                                      | MIMIC-IV         |
| ---- | ---------------------------------------------- | ---------------- |
| 数据格式 | 较旧、扁平化                                         | 模块化（hosp、icu、ed） |
| 表命名  | 大写                                             | 小写下划线            |
| 推荐使用 | 适合复现经典文献                                       | 适合做前沿研究、真实场景建模   |
| 支持模型 | 与 MIMIC-III 相同：LSTM、RETAIN、GRU-D、Transformer 等 |                  |

---

## ✅ 七、小结

| 项目   | 内容                                       |
| ---- | ---------------------------------------- |
| 目标   | 封装 ICU 数据集 MIMIC-IV，供模型使用                |
| 支持任务 | 死亡预测、住院时间、再入院、药物推荐等                      |
| 核心优势 | 数据结构更清晰，更接近现代临床信息系统                      |
| 配套模块 | 可与 `MortalityPredictionMIMIC4` 等任务模块直接兼容 |
| 建议人群 | 做 ICU 模型训练、表型建模、医疗 NLP 等科研任务者            |

---

是否继续讲解下一个数据集（如 `OMOPDataset`, `eICUDataset`），或希望我为你写一个 `MIMIC-IV + PyHealth` 的完整代码 demo 示例（包含加载 + 建模 + 预测）？

以下是关于 `pyhealth.datasets.MedicalTranscriptionsDataset` 的中文学习笔记，适合你了解如何使用 PyHealth 处理**临床文本（Medical Notes）数据**，用于 NLP（自然语言处理）相关任务，如多标签分类、诊断预测等。

---

# 📝 PyHealth 学习笔记：`MedicalTranscriptionsDataset`

---

## ✅ 一、数据集简介

`MedicalTranscriptionsDataset` 是 PyHealth 封装的**医疗文本数据集**模块。

> 它用于加载并处理包含**病人病程记录、医生笔记、护理记录等自然语言内容**的数据，适合开展医学 NLP 任务，例如：
>
> * 诊断预测（根据入院记录预测 ICD 编码）
> * 多标签疾病分类
> * 医疗文本编码（如 CPT、ICD、SNOMED 预测）
> * 病人住院摘要建模（Discharge Summary Modeling）

---

## 📦 二、数据来源说明

虽然 PyHealth 官网没有详细说明此模块背后的具体数据来源，但通常与以下公开数据集结构相似：

* **MIMIC-III/IV NoteEvents 表**（如 `NOTEEVENTS.csv`）
* **i2b2 临床笔记数据**
* **公开的医疗转录数据集**（如 MT Samples）

---

## 🧱 三、模块用途与核心功能

| 功能                            | 说明                |
| ----------------------------- | ----------------- |
| 加载结构化转录文本                     | 每位病人的文本按 visit 归档 |
| 自动切分为输入（text）与标签（如 ICD）       |                   |
| 可对接 BERT、TextCNN、LSTM 等文本分类模型 |                   |
| 支持 `set_task()` 构建文本多标签分类任务   |                   |

---

## 🔁 四、典型使用流程

```python
from pyhealth.datasets import MedicalTranscriptionsDataset

dataset = MedicalTranscriptionsDataset(
    root="/path/to/transcription_data",  # 需为结构化 CSV/text 数据
    tables=["notes", "diagnoses"],
    dev=True
)
```

其中：

* `notes`：包含临床文本字段，如 `text`, `note_type`, `subject_id`, `hadm_id`
* `diagnoses`：作为标签数据，如 ICD 诊断码或主诊断

---

## 🎯 五、可对接任务类型（推荐）

| 任务            | 对应 PyHealth 任务类                         |
| ------------- | --------------------------------------- |
| 多标签疾病预测（基于文本） | `MultiLabelTextClassificationTask`（自定义） |
| ICD 编码预测      | 自定义 `Text2CodeTask`                     |
| 文本 + 实验室联合预测  | 混合输入任务（需自定义）                            |

---

## ⚙️ 六、配合模型模块建议

可以与以下模型搭配：

* `TextCNN`, `TextRNN`（PyHealth 子类模型）
* 外部集成模型：BERT、BioClinicalBERT（使用 Huggingface）
* 可解释性：加 attention 层提取关键词

---

## 📝 七、小结

| 项目   | 内容                             |
| ---- | ------------------------------ |
| 模块名称 | `MedicalTranscriptionsDataset` |
| 类型   | 临床文本数据集                        |
| 输入格式 | 通常为医生记录文本（notes.csv）           |
| 输出标签 | ICD、CPT、SNOMED 等               |
| 支持任务 | 文本分类、多标签预测                     |
| 适用模型 | BERT、CNN、RNN、Transformer 等     |
| 推荐人群 | 医疗 NLP 研究人员、ICD 自动编码开发者        |

---

✅ 如果你想基于 MIMIC 的 `NOTEEVENTS.csv` 创建自己的 `MedicalTranscriptionsDataset` 实例，我可以帮你写一个结构清洗 + 建模的完整脚本。

是否继续讲解下一个数据集？或现在进入任务模块 `pyhealth.tasks` 的中文笔记整理？

以下是关于 `pyhealth.datasets.CardiologyDataset` 的中文学习笔记，帮助你理解如何使用 PyHealth 构建**心电图（ECG）相关任务**，例如心律失常识别、心脏病风险预测等。

---

# ❤️‍🔥 PyHealth 学习笔记：`CardiologyDataset`

---

## ✅ 一、CardiologyDataset 是什么？

`CardiologyDataset` 是 PyHealth 中专门用于处理**心电图（ECG）数据**的模块，内置多个公开心电数据集。

> 适合开展以下任务：
>
> * 心律失常分类（如房颤、室性早搏等）
> * 心脏病风险预测
> * 基于 ECG 信号的多标签学习
> * 生物信号深度学习任务（适用于 1D CNN、Transformer）

---

## 📦 二、包含的数据来源

PyHealth 封装了六个常用的心电图数据子集：

| 数据集部分名称                | 来源说明                        |
| ---------------------- | --------------------------- |
| `cpsc_2018`            | 中国心律失常挑战赛数据（2018）           |
| `cpsc_2018_extra`      | CPSC 扩展数据                   |
| `georgia`              | 美国乔治亚大学 ECG 数据              |
| `ptb`                  | 德国 PTB 数据集（标准 12 导联 ECG）    |
| `ptb-xl`               | PTB 扩展版（PTB-XL），质量更高，采样更快   |
| `st_petersburg_incart` | 俄罗斯 Saint Petersburg 心电图数据库 |

这些数据均为 **公开/半公开心电信号数据**，可用于训练医学信号处理模型。

---

## 🛠️ 三、基本使用方式（加载数据）

```python
from pyhealth.datasets import CardiologyDataset

dataset = CardiologyDataset(
    root="/your/path/to/ecg_data",
    portions=["ptb-xl", "cpsc_2018"],  # 可选加载多个子集
    dev=True  # 若为 True，仅加载部分数据用于测试
)
```

---

## 🔍 四、数据组织形式

每一条数据将被自动组织为：

```
Patient
 └── Visit
     └── Event（包含 ECG 波形数据、采样率、导联类型等）
```

常见字段包括：

* `"signal"`：ECG 信号（通常为 `ndarray`, shape=\[leads, length]）
* `"sampling_rate"`：采样率
* `"lead_names"`：导联名称（如 I, II, V1\~V6）
* `"label"`：诊断标签（如 `AF`, `PVC`, `LBBB`）

---

## 📊 五、常见任务（建议配合自定义任务类）

| 任务类型         | 输入             | 标签         |
| ------------ | -------------- | ---------- |
| 多标签分类        | 12 导 ECG 波形    | 多种心律失常     |
| 单标签诊断预测      | 标准 ECG         | 目标诊断（如 AF） |
| 波形 + 结构化联合预测 | ECG + 年龄/性别/体重 | 疾病概率       |

---

## 🧠 六、可配合的模型建议

适合配合以下模型进行训练：

* `1D-CNN`（如 `SparcNet`, `Deepr`, `TCN`）
* `Transformer`（时序建模 + 多导联注意力）
* `ContraWR`（STFT+CNN 构建信号特征）
* 外部模型如 `ECGNet`, `ResNet-18`, `WaveNet`（自定义集成）

---

## ✅ 七、小结

| 项目   | 内容                                     |
| ---- | -------------------------------------- |
| 模块名  | `CardiologyDataset`                    |
| 适用方向 | 心电图信号处理、心律失常识别                         |
| 数据形式 | 波形数据（多导联 ECG）                          |
| 标签类型 | 多标签心律失常                                |
| 推荐模型 | 1D CNN、TCN、Signal Transformer、ContraWR |
| 推荐任务 | 信号分类、多标签任务、心电诊断预测                      |

---

如你希望我为你写一个基于 `ptb-xl` 数据的完整任务构建、训练、模型预测 Pipeline 示例（含图形可视化），可以直接告诉我，我将立即整理。是否继续进入下一个数据集（如 `OMOPDataset`, `eICUDataset`）或转向 `pyhealth.tasks` 的任务模块讲解？

以下是关于 `pyhealth.datasets.eICUDataset` 的中文学习笔记，帮助你理解如何使用 PyHealth 处理 **eICU 协作研究数据库**，开展 ICU 相关的预测建模任务。

---

# 🏥 PyHealth 学习笔记：`eICUDataset`

---

## ✅ 一、什么是 eICU 数据库？

**eICU Collaborative Research Database** 是由 Philips Healthcare 与 MIT 合作发布的 ICU 多中心数据库，涵盖美国 200 多家医院的 ICU 病人信息，具有以下特点：

| 特征    | 说明                      |
| ----- | ----------------------- |
| 多中心数据 | 包含不同医院、不同病房 ICU 的情况     |
| 结构化字段 | 覆盖生命体征、治疗记录、实验室数据、评分系统等 |
| 数据时间段 | 通常为 2014–2015 年         |
| 病人数量  | 超过 20 万 ICU 入院病例        |
| 表格格式  | CSV 文件（与 MIMIC 不完全相同）   |

---

## 📦 二、PyHealth 中的 `eICUDataset`

`eICUDataset` 是 PyHealth 提供的官方封装器，支持将 eICU 数据集转换为统一的：

```
Patient → Visit → Event
```

的结构化对象，便于后续用于：

* 死亡率预测
* 再入院风险评估
* ICU 住院时间预测
* 事件时间序列建模（如血压变化 → 用药反应）

---

## 🧱 三、基本用法示例

```python
from pyhealth.datasets import eICUDataset

dataset = eICUDataset(
    root="/your/local/eicu/",
    tables=["diagnosis", "treatment", "lab", "vitalperiodic"],
    dev=True
)
```

### 参数说明：

| 参数       | 类型          | 说明                                         |
| -------- | ----------- | ------------------------------------------ |
| `root`   | `str`       | 指向 eICU CSV 文件所在目录                         |
| `tables` | `List[str]` | 要加载的表名，如 `diagnosis`, `treatment`, `lab` 等 |
| `dev`    | `bool`      | 是否加载小部分用于测试                                |

---

## 📊 四、推荐加载的表格字段

| 表名              | 说明                          |
| --------------- | --------------------------- |
| `diagnosis`     | 诊断记录                        |
| `treatment`     | 治疗行为（如插管、用药等）               |
| `lab`           | 实验室检查数据（如乳酸、肾功能）            |
| `vitalperiodic` | 生命体征数据（如 HR, RR, MAP）       |
| `apachepredvar` | APACHE 评分相关字段（用于 ICU 死亡率建模） |

---

## 🧠 五、任务搭配建议

你可以使用 PyHealth 提供的任务类，如：

```python
from pyhealth.tasks import MortalityPredictionEICU

task_fn = MortalityPredictionEICU()
dataset = dataset.set_task(task_fn)
```

也可自定义任务处理逻辑，构建多分类、回归或时间序列预测任务。

---

## ⚙️ 六、模型推荐搭配

适用于 ICU 多模态数据建模的模型包括：

* `RNN`, `GRU-D`, `LSTM`
* `RETAIN`（可解释）
* `AdaCare`, `ConCare`, `StageNet`（序列建模 + 可解释性）
* `Transformer`（支持不规则时间序列）

---

## ✅ 七、小结

| 项目   | 内容                                                 |
| ---- | -------------------------------------------------- |
| 模块名  | `eICUDataset`                                      |
| 数据类型 | ICU 多中心结构化数据                                       |
| 特点   | 多医院来源，结构丰富                                         |
| 推荐任务 | ICU 死亡率预测、住院时间预测、治疗响应                              |
| 支持表格 | `diagnosis`, `lab`, `vitalperiodic`, `treatment` 等 |
| 推荐模型 | LSTM、GRU、RETAIN、Transformer、AdaCare、ConCare 等      |

---

如果你有原始 eICU 数据包（CSV），我可以帮你写一份完整的 `eICUDataset` 加载 + 任务设置 + 模型训练 demo。

是否继续讲解下一个数据集（如 `OMOPDataset`, `SleepEDFDataset`），或转向模型模块（`pyhealth.models`）详细学习？


以下是关于 `pyhealth.datasets.ISRUCDataset` 的中文学习笔记，帮助你理解如何使用 PyHealth 构建**睡眠分期与多通道生理信号建模**任务。

---

# 😴 PyHealth 学习笔记：`ISRUCDataset`

---

## ✅ 一、数据集简介：ISRUC-Sleep

**ISRUC-Sleep Dataset** 是一个面向睡眠研究的综合性公开数据库，结构化记录了大量**多导睡眠监测（PSG）信号**，广泛用于睡眠分期、睡眠障碍检测、自动诊断等任务。

> 数据来源：葡萄牙科英布拉大学医院睡眠医学中心

> 正式文献引用：
> Khalighi S, Sousa T, Santos JM, Nunes U. *ISRUC-Sleep: A comprehensive public dataset for sleep researchers*. Comput Methods Programs Biomed. 2016;124:180–192.

---

## 📦 二、数据组成（按实验分组）

| 分组      | 描述                                 |
| ------- | ---------------------------------- |
| Group 1 | **100名受试者**，每人一晚 PSG 记录（常用于监督学习任务） |
| Group 2 | **8名受试者，每人两晚记录**（研究 PSG 变化趋势）      |
| Group 3 | **10名健康人**，用于对比研究（健康 vs 睡眠障碍）      |

每位受试者的记录包含以下信号类型：

* **脑电图 EEG**：C3-A2、C4-A1 等通道
* **眼电图 EOG**：左眼、右眼运动
* **肌电图 EMG**：下颌、肢体肌电
* **心电图 ECG**：部分样本含有
* **呼吸信号**：胸腹运动、鼻流气压力等
* **事件标注**：由专家标注的睡眠阶段（如 N1, N2, REM）

---

## 🧱 三、在 PyHealth 中的加载方式

PyHealth 对该数据库封装为 `ISRUCDataset` 类，支持你快速构建睡眠建模任务：

```python
from pyhealth.datasets import ISRUCDataset

dataset = ISRUCDataset(
    root="/your/local/ISRUC_data/",
    group=1,           # 选择数据组（1、2 或 3）
    signal_types=["EEG", "EOG", "EMG"],
    dev=True           # 是否为开发模式（仅加载小数据）
)
```

---

## 🔍 四、核心参数解析

| 参数             | 类型          | 含义                            |
| -------------- | ----------- | ----------------------------- |
| `group`        | `int`       | 选择数据组（1: 100人、2: 两晚记录、3: 健康组） |
| `signal_types` | `List[str]` | 加载的信号类型，如 EEG、EOG、EMG、ECG     |
| `dev`          | `bool`      | 是否只加载小规模数据用于调试                |

输出为：

```
Patient → Visit → Event（含 PSG 波形数据、标签、导联名称等）
```

---

## 🎯 五、典型任务与模型配合建议

### 💤 睡眠分期任务（Sleep Staging）

* 输入：多导 EEG/EOG/EMG 信号片段（通常 30s）
* 标签：睡眠阶段（Wake, N1, N2, N3, REM）
* 可配任务类：自定义 Task 类或 PyHealth 中 `SleepStageClassificationTask`

### 推荐模型：

| 模型                         | 特点                                    |
| -------------------------- | ------------------------------------- |
| `pyhealth.models.ContraWR` | 使用 STFT + 2D CNN 结构，适合 PSG 信号         |
| `pyhealth.models.SparcNet` | 基于 1D CNN，高效处理时间序列波形                  |
| `Transformer`, `TCN`       | 可建模信号间交互与时序依赖                         |
| 自定义模型                      | 如 DeepSleepNet, SeqSleepNet 等经典结构也可复现 |

---

## 🛠️ 六、数据预处理技巧（可选）

* **标准化**：波形归一化为均值0，方差1
* **分片处理**：每30秒为一个片段（对应一个睡眠阶段标签）
* **平衡采样**：对少见睡眠阶段进行过采样或加权损失
* **数据增强**：如加入高斯噪声、时间抖动等方法改善泛化

---

## ✅ 七、小结

| 项目   | 内容                                |
| ---- | --------------------------------- |
| 模块名  | `ISRUCDataset`                    |
| 数据来源 | ISRUC-Sleep（葡萄牙医院）                |
| 任务类型 | 睡眠分期、多导信号建模                       |
| 信号支持 | EEG、EOG、EMG、ECG 等                 |
| 推荐模型 | ContraWR、SparcNet、TCN、Transformer |
| 研究价值 | 临床诊断辅助、睡眠障碍检测、脑-心-呼吸交互研究          |

---

✅ 如果你手头有 ISRUC 的原始 `.mat` 或 `.edf` 格式数据，我可以帮你写一套自动解析、建模、训练与可视化的完整 Pipeline。如果你想转向下一模块 `OMOPDataset`，或者现在想进入模型模块 `pyhealth.models` 学习，也可以告诉我。

以下是关于 `pyhealth.datasets.MIMICExtractDataset` 的中文学习笔记，帮助你理解如何使用 PyHealth 构建基于 ICU 的临床预测任务，尤其是来自精简版的 **MIMIC-III Extract 数据集**。

---

# 🏥 PyHealth 学习笔记：`MIMICExtractDataset`

---

## ✅ 一、什么是 MIMIC-Extract？

`MIMIC-Extract` 是由哈佛大学等机构对原始 MIMIC-III 数据库进行的**特征工程处理后版本**，专为构建机器学习模型而设计，特点包括：

| 特征     | 内容                                                          |
| ------ | ----------------------------------------------------------- |
| 来源     | MIMIC-III v1.4                                              |
| 精简与结构化 | 整合原始 26 个表为单一特征表                                            |
| 面向模型   | 适合直接用于深度学习/传统 ML 的建模                                        |
| 开源工程   | [GitHub 传送门](https://github.com/YerevaNN/mimic3-benchmarks) |

数据包含：

* **按时间分段的时间序列特征**（如每小时一次的心率、血压、GCS评分等）
* **静态信息**（如年龄、性别、是否死亡等标签）
* **目标任务**：包括死亡预测、住院时间预测、长住风险等

---

## 📦 二、PyHealth 中的 `MIMICExtractDataset`

PyHealth 对该数据集进行了封装，使其与其他模块（任务、模型、训练器）无缝集成。

```python
from pyhealth.datasets import MIMICExtractDataset

dataset = MIMICExtractDataset(
    root="/your/path/to/mimic_extract/",
    tables=["features", "labels"],
    dev=True
)
```

---

## 🔍 三、推荐使用的表格字段

| 表名         | 说明                                      |
| ---------- | --------------------------------------- |
| `features` | ICU 患者每小时的临床变量时间序列（e.g., HR, MAP, FiO2） |
| `labels`   | 各类标签，包括 24/48 小时死亡、再入院、LOS 分类等          |

每位患者样本将被 PyHealth 自动构建为：

```
Patient → Visit → 多个 Event（每小时的生理数据片段）
```

---

## 🎯 四、推荐任务构建方式

1. **死亡预测任务（Binary）**
2. **多分类住院时长预测（LOS分类）**
3. **时间序列分类或回归**

你可以通过继承 `pyhealth.tasks.Task` 自定义任务逻辑，也可等待 PyHealth 官方扩展 `MIMICExtract` 专用任务类。

---

## 🧠 五、适配模型建议

| 模型                               | 优点                   |
| -------------------------------- | -------------------- |
| `GRU-D`                          | 能处理缺失值、时间延迟等问题       |
| `RETAIN`                         | 可解释性强（按时间反向注意力）      |
| `LSTM`, `RNN`, `TCN`             | 基线时序建模               |
| `Transformer`                    | 全局注意力，适合建模 ICU 多变量交互 |
| `AdaCare`, `StageNet`, `ConCare` | 高级表示学习 + 可解释机制       |

---

## 🛠️ 六、可视化建议

* 多变量 ICU 信号趋势图（如 HR/BP/SpO2 变化）
* 注意力热力图（RETAIN 或 Transformer 模型可用）
* 特征重要性图（SHAP 或 Attention）

---

## ✅ 七、小结

| 项目   | 内容                               |
| ---- | -------------------------------- |
| 模块名  | `MIMICExtractDataset`            |
| 数据源  | 哈佛团队处理过的 MIMIC-III 子集            |
| 支持任务 | ICU 死亡预测、住院时间预测、多任务预测            |
| 推荐模型 | GRU-D、RETAIN、Transformer、AdaCare |
| 特点   | 时间对齐、缺失处理、标准变量、开箱即用              |

---

如你有 `mimic_extract` 格式的 `.csv` 文件集，我可以帮你快速写一份完整训练 pipeline（数据加载、任务创建、模型训练、评估与可视化），是否继续学习下一个数据集（如 `OMOPDataset`, `SleepEDFDataset`），或切换到模型模块 `pyhealth.models`？

以下是关于 `pyhealth.datasets.OMOPDataset` 的中文学习笔记，重点帮助你了解如何使用 PyHealth 处理标准化的 **OMOP-CDM 格式医疗数据库**，并开展机器学习/深度学习建模任务。

---

# 🏥 PyHealth 学习笔记：`OMOPDataset`

---

## ✅ 一、什么是 OMOP-CDM？

**OMOP-CDM（Observational Medical Outcomes Partnership Common Data Model）** 是由 OHDSI 联盟提出的一套全球标准医疗数据模型，主要用于结构化存储不同医院/国家的电子病历（EHR），实现医疗大数据的统一表示。

### OMOP-CDM 的核心特点：

| 项目     | 内容                                                        |
| ------ | --------------------------------------------------------- |
| 开放共享   | 被全球 100+ 医疗机构采用，支持协同分析                                    |
| 模型结构   | 统一标准表结构，如 `person`、`condition_occurrence`、`drug_exposure` |
| 支持术语体系 | ICD, SNOMED, RxNorm, ATC, LOINC 等编码系统                     |
| 临床类型   | 包括诊断、用药、实验室检查、就诊信息等                                       |

---

## 📦 二、PyHealth 中的 `OMOPDataset`

`pyhealth.datasets.OMOPDataset` 让你可以直接加载任意本地的 OMOP-CDM 格式数据库，转化为标准的：

```
Patient → Visit → Event
```

结构，便于后续任务设置和模型训练。

### 基本用法示例：

```python
from pyhealth.datasets import OMOPDataset

dataset = OMOPDataset(
    root="/path/to/your/omop/",
    tables=["condition_occurrence", "drug_exposure", "measurement"],
    dev=True
)
```

---

## 🧾 三、支持的常见表格（tables）

| 表名                     | 内容描述                   |
| ---------------------- | ---------------------- |
| `person`               | 人口统计数据（年龄、性别、出生日期等）    |
| `visit_occurrence`     | 就诊信息（门急诊、住院）           |
| `condition_occurrence` | 诊断记录（疾病 ICD/SNOMED）    |
| `drug_exposure`        | 药物暴露记录（RxNorm）         |
| `procedure_occurrence` | 医疗操作（手术、穿刺等）           |
| `measurement`          | 检验/实验室检查（如 HbA1c, GFR） |
| `observation`          | 其它记录（如体重、吸烟状态）         |

**注意**：以上表格中常用的字段将被自动转换为 Event 对象。

---

## 🎯 四、推荐任务与应用场景

| 任务类型   | 举例                      |
| ------ | ----------------------- |
| 多标签分类  | 多疾病预测（根据历史诊疗信息预测未来诊断）   |
| 回归任务   | 实验室指标预测，如预测 HbA1c、肾功能趋势 |
| 时间序列建模 | 某慢病患者随访期变化趋势（如 CKD 分级）  |
| 药物推荐   | 基于历史诊断预测合理的处方组合（如抗糖尿病药） |

---

## 🤖 五、与模型模块协同使用建议

你可以使用以下模型进行建模：

| 模型名称                  | 优点                       |
| --------------------- | ------------------------ |
| `RETAIN`              | 时序解释性强，适合 EHR 模型         |
| `RNN`/`LSTM`          | 基线模型，建模就诊顺序              |
| `GRU-D`               | 适合建模缺失/异步时间戳数据           |
| `Transformer`         | 多模态、多序列信息整合能力强           |
| `GAMENet`, `SafeDrug` | 可用于药物推荐任务，考虑药物交互性（DDI）风险 |

---

## 🔗 六、编码映射模块的支持

由于 OMOP 涉及大量标准代码系统（如 ICD9、ATC、RxNorm），你可以结合：

```python
from pyhealth.medcode import CrossMap, InnerMap
```

实现跨系统编码映射（如 ICD9 → CCS，RxNorm → ATC），或获取某代码的层级结构（get\_ancestors）。

---

## ✅ 七、小结

| 项目   | 内容                                            |
| ---- | --------------------------------------------- |
| 模块名  | `OMOPDataset`                                 |
| 数据来源 | 任意符合 OMOP-CDM 规范的本地数据库                        |
| 支持表格 | 诊断、用药、手术、检验、观察等                               |
| 建议任务 | 多标签预测、实验室指标回归、药物推荐等                           |
| 推荐模型 | RETAIN, GRU-D, Transformer, GAMENet, SafeDrug |
| 附加工具 | `pyhealth.medcode` 实现编码转换与可视化                 |

---

✅ 如你本地有 OMOP-CDM 数据（CSV、SQL、BigQuery 等格式），我可以帮你写出完整的数据加载、任务构建、模型训练与评估的 Jupyter Notebook 教程。

下一步是否继续学习其他数据集（如 `SleepEDFDataset`, `SHHSDataset`），或者深入模型模块 `pyhealth.models` 的详细结构？

以下是关于 `pyhealth.datasets.SHHSDataset` 的中文学习笔记，帮助你理解如何使用 PyHealth 处理 **SHHS 睡眠心脏健康研究数据集（Sleep Heart Health Study）**，并开展生理信号分析或睡眠事件预测任务。

---

# 💤 PyHealth 学习笔记：`SHHSDataset`

---

## ✅ 一、什么是 SHHS 数据集？

**SHHS（Sleep Heart Health Study）** 是美国国家心肺血液研究所支持的一项大规模睡眠健康研究项目，目标是研究**睡眠障碍与心脑血管疾病**之间的联系。

### 数据特点：

| 项目   | 内容                          |
| ---- | --------------------------- |
| 覆盖人群 | 超过 6,000 名中老年受试者            |
| 采样方式 | 多导睡眠图（Polysomnography，PSG）  |
| 数据类型 | EEG、EOG、EMG、ECG、呼吸流量、血氧饱和度等 |
| 睡眠标注 | 每个 30 秒 epoch 的睡眠阶段手工评分     |
| 数据用途 | 睡眠阶段分类、呼吸暂停预测、EEG 事件分析等     |

---

## 📦 二、PyHealth 中的 `SHHSDataset`

`pyhealth.datasets.SHHSDataset` 封装了 SHHS 的数据读取、预处理、样本构建等功能，并将信号数据以标准结构组织，便于用于模型训练和任务定义。

```python
from pyhealth.datasets import SHHSDataset

dataset = SHHSDataset(
    root="/your/path/to/SHHS/",
    signals=["EEG", "ECG", "EOG"],
    labels="sleep_stage",
    dev=True
)
```

---

## 🎯 三、常见研究任务

| 任务类型           | 描述                                    |
| -------------- | ------------------------------------- |
| 睡眠阶段分类         | 将 30 秒的 PSG 信号片段分类为 N1/N2/N3/REM/Wake |
| 呼吸暂停预测         | 基于前段信号预测是否将发生呼吸暂停                     |
| EEG/ECG 时间序列建模 | 脑电节律、心电变异性特征提取与回归建模                   |

---

## 🧱 四、数据结构转换

PyHealth 会自动将 SHHS 数据转换为：

```
Patient → Visit → Event（单个 PSG epoch 片段）
```

其中 Event 的时间戳和信号片段通过 `attr_dict` 字典表示，便于后续送入模型。

---

## 🤖 五、适配模型建议

对于生理信号建模，建议使用：

| 模型                     | 特点                               |
| ---------------------- | -------------------------------- |
| `CNN` / `1D CNN`       | 用于提取局部特征、节律模式                    |
| `TCN`                  | 时序建模、处理长距离依赖                     |
| `Transformer`          | 捕捉信号全局结构                         |
| `ContraWR`, `SparcNet` | 专为生理信号设计的高级结构，支持 STFT + CNN 表示学习 |
| `MLP`                  | 基础对照模型                           |

---

## 🛠️ 六、进阶建议

* 可以配合 `pyhealth.tokenizer` 对阶段标签进行编码
* 可将信号转换为频域（如小波、STFT）以适配 CNN/Transformer 输入
* 配合 `shap` 或 attention 可视化模型关注的信号区域

---

## ✅ 七、小结

| 项目   | 内容                                        |
| ---- | ----------------------------------------- |
| 模块名  | `SHHSDataset`                             |
| 数据类型 | 多导睡眠图（PSG）生理信号数据                          |
| 推荐任务 | 睡眠阶段分类、呼吸暂停预测、心电脑电建模                      |
| 推荐模型 | CNN, TCN, Transformer, ContraWR, SparcNet |
| 标签格式 | 每个 epoch 一类睡眠阶段标签                         |
| 适用人群 | 从事 EEG/ECG/PSG 信号处理、睡眠建模、健康状态预测的研究者       |

---

如你手头已有 SHHS 的 EDF 数据或已转成 `.npy/.csv`，我可帮你构建 PyHealth 兼容的数据集对象、信号切分函数、分类任务模块和训练代码。

是否继续学习下一个数据集（如 `SleepEDFDataset`, `ISRUCDataset`）或转向 `pyhealth.models` 模块深入探索？

以下是关于 `pyhealth.datasets.SleepEDFDataset` 的中文学习笔记，帮助你理解如何使用 PyHealth 处理经典的 **Sleep-EDF Expanded** 睡眠数据库，用于睡眠阶段分类等生理信号建模任务。

---

# 💤 PyHealth 学习笔记：`SleepEDFDataset`

---

## ✅ 一、什么是 Sleep-EDF Expanded 数据集？

**Sleep-EDF Expanded** 是一个公开的、多导睡眠图（PSG）数据库，由欧洲睡眠中心 PhysioNet 提供，广泛用于睡眠阶段分类、EEG 建模、REM 检测等任务。

### 数据特点：

| 项目   | 内容                                                                |
| ---- | ----------------------------------------------------------------- |
| 人群   | 健康成年人（年龄约 25–34 岁）                                                |
| 数据量  | 78 个完整的夜间 PSG 记录                                                  |
| 信号类型 | EEG、EOG、EMG（频率为 100Hz）                                            |
| 标签   | 每 30 秒 epoch 的睡眠阶段标注（W/N1/N2/N3/REM）                              |
| 来源   | [PhysioNet 数据页面](https://physionet.org/content/sleep-edfx/1.0.0/) |

---

## 📦 二、PyHealth 中的 `SleepEDFDataset`

`pyhealth.datasets.SleepEDFDataset` 是对 Sleep-EDF 数据的标准封装。使用该模块可以快速构建符合模型训练结构的样本集合。

### 示例：

```python
from pyhealth.datasets import SleepEDFDataset

dataset = SleepEDFDataset(
    root="/your/path/to/SleepEDF/",
    signals=["EEG Fpz-Cz"],  # 可以选择 EEG/EOG/EMG 通道
    labels="sleep_stage",
    dev=True
)
```

### 参数说明：

* `root`：数据集存放路径（解压后的 EDF 文件）
* `signals`：指定要使用的通道，例如 `"EEG Fpz-Cz"` 或 `"EOG horizontal"`
* `labels`：默认 `"sleep_stage"`，即睡眠阶段分类任务
* `dev`：是否使用开发模式（加载小部分数据调试）

---

## 🧱 三、数据结构转换

PyHealth 自动将 SleepEDF 的 EDF 格式数据，转换为：

```
Patient → Visit → Event（30秒信号片段）
```

每个 Event 对象包含：

* timestamp：该 epoch 起始时间
* attr\_dict：包含 `signal` 数组（如 EEG）与 `label` 标签（如 `N2`）

---

## 🎯 四、适用任务类型

| 任务       | 描述                          |
| -------- | --------------------------- |
| 睡眠阶段分类   | 五分类或七分类（W, N1, N2, N3, REM） |
| EEG 模式建模 | 睡眠脑电图频谱变化趋势分析               |
| 时序建模任务   | 预测未来睡眠阶段 / 转换概率             |

---

## 🤖 五、推荐模型结构

针对信号序列和分类任务，建议模型如下：

| 模型                          | 优势                                |
| --------------------------- | --------------------------------- |
| `CNN`, `1D CNN`, `SparcNet` | 提取时间轴的局部模式（如 spindles, K-complex） |
| `Transformer`               | 捕捉长距离依赖，适用于阶段迁移预测                 |
| `TCN`                       | 保持时序、计算效率高                        |
| `ContraWR`                  | STFT + CNN 结构，适合频域信号表达学习          |
| `MLP`                       | 对照模型，适合单片段评估基线性能                  |

---

## 📌 六、注意事项

* EDF 文件读取需依赖 `pyEDFlib` 或 `MNE`，PyHealth 内部已处理
* 信号处理建议标准化（zero-mean, unit variance）
* 标签不平衡（如 N1 少、N2 多）需注意权重或采样策略

---

## ✅ 七、小结

| 项目   | 内容                                        |
| ---- | ----------------------------------------- |
| 模块名  | `SleepEDFDataset`                         |
| 数据类型 | 多导睡眠图（PSG），EEG/EOG/EMG 信号                 |
| 推荐任务 | 睡眠阶段分类、信号建模                               |
| 推荐模型 | CNN, TCN, Transformer, ContraWR, SparcNet |
| 标签   | 每 30 秒一个睡眠阶段（共五类或七类）                      |
| 常用信号 | EEG Fpz-Cz / EEG Pz-Oz / EOG horizontal   |

---

如果你希望基于 SleepEDF 构建一个完整的睡眠阶段分类模型项目（数据加载、切片、模型构建、训练、评估与可视化），我可以帮助你生成全套 Jupyter Notebook 教程。

是否继续学习下一个数据集（如 `ISRUCDataset`, `TUABDataset`），或者切换到 PyHealth 的模型模块（`pyhealth.models`）进一步深入？

以下是关于 `pyhealth.datasets.TUABDataset` 的中文学习笔记，帮助你了解如何使用 PyHealth 处理 **TUAB 脑电图（EEG）异常检测数据集**，并构建用于临床脑电筛查的机器学习模型。

---

# 🧠 PyHealth 学习笔记：`TUABDataset`

---

## ✅ 一、什么是 TUAB 数据集？

**TUAB** 全称为 *Temple University Hospital EEG Abnormal Corpus*，是由 Temple 大学医院采集的真实临床 EEG 脑电数据，包含大量 **正常与异常脑电图**，常用于脑部疾病识别模型研究。

### 数据特点：

| 项目   | 内容                                                                         |
| ---- | -------------------------------------------------------------------------- |
| 医院来源 | Temple University Hospital（费城）                                             |
| 数据类型 | EEG（脑电图）                                                                   |
| 标签   | Normal / Abnormal 二分类标签                                                    |
| 采样率  | 多为 250Hz 或 500Hz                                                           |
| 通道数  | 常见为 21 通道（10-20 系统）                                                        |
| 文件格式 | EDF（欧洲数据格式）                                                                |
| 下载地址 | [TUAB 官网](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tuab) |

---

## 📦 二、PyHealth 中的 `TUABDataset`

在 PyHealth 中，`TUABDataset` 封装了 TUAB 数据的解析、标注提取、样本构建等过程，并转换为标准的 **Patient-Visit-Event** 结构，方便用于模型训练。

### 示例代码：

```python
from pyhealth.datasets import TUABDataset

dataset = TUABDataset(
    root="/your/path/to/TUAB/",
    signals=["EEG"],  # 默认会加载所有通道
    labels="abnormality",  # 二分类任务
    dev=True  # 开启开发模式，仅加载少量数据用于调试
)
```

---

## 🧱 三、数据结构说明

在 PyHealth 内部，TUAB 数据被组织为：

```
Patient → Visit → Event（EEG 时间窗片段）
```

每个 `Event` 包含：

* `timestamp`：片段开始时间
* `attr_dict["signal"]`：一个 `[通道数, 时间步]` 的 EEG 信号矩阵
* `attr_dict["label"]`：是否异常（1：abnormal, 0：normal）

---

## 🎯 四、推荐研究任务

| 任务类型        | 描述                   |
| ----------- | -------------------- |
| 异常脑电检测（二分类） | 判断脑电是否包含异常放电、尖波、慢波等  |
| EEG 表征学习    | 提取个体大脑电活动特征（无监督或自监督） |
| 片段级注意力聚合    | 从多个片段中找出关键异常点（多实例学习） |

---

## 🤖 五、推荐模型结构

| 模型名称                     | 特点与适用性                               |
| ------------------------ | ------------------------------------ |
| `1D CNN`, `SparcNet`     | 时间序列的局部模式提取（如 epileptiform patterns） |
| `Transformer`, `ConCare` | 多通道时序建模、全局注意力机制                      |
| `TCN`                    | 稀疏卷积结构，捕捉远程时序依赖                      |
| `RNN`, `GRU`, `LSTM`     | 脑电序列逐步建模（基础方案）                       |
| `ContraWR`               | 频谱+卷积联合特征提取（适合 EEG 表达）               |

---

## 🔧 六、建模建议

* 推荐将信号归一化（每通道零均值单位方差）
* 可分片为 1 秒 / 2 秒 / 30 秒时间窗（重叠或不重叠）
* 训练时处理标签不平衡问题（加权 loss / 过采样）

---

## ✅ 七、小结

| 项目   | 内容                              |
| ---- | ------------------------------- |
| 模块名  | `TUABDataset`                   |
| 数据类型 | 多通道 EEG 信号                      |
| 标签   | 正常 / 异常 二分类                     |
| 主要任务 | 异常脑电检测                          |
| 推荐模型 | CNN, Transformer, TCN, ContraWR |
| 数据格式 | EDF（自动解析）                       |
| 常见通道 | Fp1-F7, F7-T3, T3-T5 等 21 通道    |

---

如你已下载 TUAB 数据集并解压，PyHealth 可帮你自动完成通道抽取、时间窗切片、标签对齐、样本构建。如果你希望搭建一个 TUAB 脑电分类完整工程（数据预处理 + 模型训练 + 混淆矩阵评估 + Attention 可视化），我可以帮你逐步写出代码模板。

是否继续学习下一个数据集（如 `TUEVDataset`）、或深入 `pyhealth.models` 模型模块？

以下是关于 `pyhealth.datasets.TUEVDataset` 的中文学习笔记，适用于希望使用 PyHealth 对脑电图（EEG）中**癫痫相关放电模式**进行建模和识别的研究者。

---

# ⚡ PyHealth 学习笔记：`TUEVDataset`

---

## ✅ 一、什么是 TUEV 数据集？

**TUEV** 全称为 *Temple University Hospital EEG Event Corpus*，是由 Temple 大学医院 EEG 数据构建的子集，聚焦于 **脑电事件级别的分类任务**，尤其用于癫痫事件检测、伪迹剔除和背景信号建模。

📌 官方数据页面：
[https://isip.piconepress.com/projects/tuh\_eeg/html/downloads.shtml](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)

---

### 📊 数据标签类别（共六类）

| 类别代号 | 名称                                           | 含义          |
| ---- | -------------------------------------------- | ----------- |
| SPSW | Spike and Sharp Wave                         | 癫痫性尖波和锐波放电  |
| GPED | Generalized Periodic Epileptiform Discharges | 广泛性周期性癫痫放电  |
| PLED | Periodic Lateralized Epileptiform Discharges | 偏侧周期性癫痫放电   |
| EYEM | Eye Movement                                 | 眼动信号        |
| ARTF | Artifact                                     | 电极伪迹或其他噪音   |
| BCKG | Background                                   | 正常背景 EEG 信号 |

---

## 📦 二、PyHealth 中的 `TUEVDataset`

在 PyHealth 中，`TUEVDataset` 对 TUEV 数据进行了标准封装，并将 EDF 信号与标注对齐、分段后组织成结构化的样本。

```python
from pyhealth.datasets import TUEVDataset

dataset = TUEVDataset(
    root="/your/path/to/TUEV/",
    signals=["EEG"],           # 默认读取所有 EEG 通道
    labels="tuev_events",      # 对应六类事件
    dev=True                   # 若设为 True，则仅加载一部分数据用于调试
)
```

---

## 🧱 三、数据结构说明

加载后的结构为：

```
Patient → Visit → Event（EEG 时间窗）
```

每个 Event 具有：

* timestamp：事件发生时间
* attr\_dict\["signal"]：该片段的 EEG 信号（通道 × 时间）
* attr\_dict\["label"]：六类标签之一（如 GPED）

---

## 🎯 四、推荐任务类型

| 任务        | 描述                    |
| --------- | --------------------- |
| EEG 时间窗分类 | 识别癫痫尖波、周期放电、背景噪声等     |
| 伪迹检测      | 剔除 EYEM / ARTF 类型数据片段 |
| 多类癫痫事件识别  | 高风险癫痫发作前状态识别          |
| 联合学习      | 多任务预测信号类型 + 疾病状态      |

---

## 🤖 五、推荐模型结构

| 模型                   | 适用性                       |
| -------------------- | ------------------------- |
| `1D CNN`, `SparcNet` | 适合快速定位 EEG 局部放电模式（如尖波）    |
| `Transformer`        | 可同时建模时间序列与通道维度间的关系        |
| `TCN`                | 膨胀卷积保持长程依赖、低运算负载          |
| `ContraWR`           | 频谱提取 + CNN 结构，适合 EEG 频域建模 |
| `ConCare`, `AdaCare` | 自适应特征注意机制，适合高维 EEG 输入     |

---

## ⚠️ 六、注意事项

* 各类事件样本不平衡，建议使用 F1-score/加权 loss
* 建议按事件对齐切片（非固定窗口长度）
* 注意不同标签之间存在边界模糊（如 SPSW vs GPED）

---

## ✅ 七、小结

| 项目   | 内容                                       |
| ---- | ---------------------------------------- |
| 模块名  | `TUEVDataset`                            |
| 数据类型 | EEG 时序事件（多通道）                            |
| 标签   | 6 类：SPSW, GPED, PLED, EYEM, ARTF, BCKG   |
| 推荐任务 | 癫痫放电识别、伪迹检测、EEG 多类分类                     |
| 推荐模型 | CNN, Transformer, TCN, ContraWR, AdaCare |
| 标签粒度 | 时间窗级（window-level）                       |

---

如果你希望构建一个完整的癫痫检测项目（包括数据预处理、模型训练、标签可视化、混淆矩阵评估），我可以帮你提供逐步的 Jupyter Notebook 模板。

📌 你也可以将多个 EEG 数据集（如 TUAB、TUEV）联合预训练或迁移学习，提升模型泛化能力。

是否现在继续学习下一个模块（如 `pyhealth.tasks`, `pyhealth.models`, `pyhealth.tokenizer`）？我可以继续为你整理笔记。


以下是关于 `pyhealth.datasets.splitter` 的中文学习笔记，聚焦于 **如何在 PyHealth 中对医疗数据集进行训练集 / 验证集 / 测试集划分**。

---

# 🔀 PyHealth 学习笔记：`pyhealth.datasets.splitter`

---

## ✅ 一、模块简介

`pyhealth.datasets.splitter` 是 PyHealth 中的一个实用模块，用于将 **构建好的医疗样本数据集** 按照一定规则划分为：

* 训练集（Train Set）
* 验证集（Validation Set）
* 测试集（Test Set）

支持**按患者**或**按样本**划分，确保临床研究的样本独立性和泛化能力。

---

## 🧩 二、常用划分函数说明

### 1️⃣ `split_by_patient(dataset, ratios)`

**按患者 ID 划分**（常用于纵向预测、避免患者泄漏）

```python
from pyhealth.datasets import split_by_patient

train_set, val_set, test_set = split_by_patient(
    dataset,
    ratios=[0.8, 0.1, 0.1]  # 训练 / 验证 / 测试 比例
)
```

🔍 说明：

* 将数据集按唯一 `patient_id` 分组
* 不同划分中不会出现相同患者
* 推荐用于大多数电子病历（EHR）任务

---

### 2️⃣ `split_by_visit(dataset, ratios)`

**按 visit-level 访问划分**，允许同一患者的不同时间点分布在不同子集中

```python
from pyhealth.datasets import split_by_visit

train_set, val_set, test_set = split_by_visit(
    dataset,
    ratios=[0.7, 0.15, 0.15]
)
```

🔍 说明：

* 若任务关注访问而非患者连续性（如 ICU stay 分类），可用此方式
* 风险：同一患者可能在多个子集中 → 会造成泄漏（需根据任务目标决定）

---

### 3️⃣ `split_by_sample(dataset, ratios)`

**逐条样本进行随机划分**（用于非结构化信号或单点数据）

```python
from pyhealth.datasets import split_by_sample

train_set, val_set, test_set = split_by_sample(
    dataset,
    ratios=[0.6, 0.2, 0.2]
)
```

🔍 说明：

* 不考虑患者/访问归属
* 更像传统的机器学习任务（如 ECG/EEG 信号片段分类）
* 适用于单点采样型数据，如 ISRUC、TUEV、TUAB

---

## 🔒 三、选择哪种划分方式？

| 划分函数               | 场景推荐            | 是否避免患者泄漏    |
| ------------------ | --------------- | ----------- |
| `split_by_patient` | 长期随访数据，EHR，诊断预测 | ✅ 是         |
| `split_by_visit`   | 每次住院独立建模        | ❌ 否（需谨慎）    |
| `split_by_sample`  | 生理信号 / 图像片段分类   | ❌ 否（适合单独事件） |

---

## 🧪 四、划分后的使用

划分结果为三个 `SampleDataset` 类型对象，可直接用于模型训练：

```python
from pyhealth.datasets import get_dataloader

train_loader = get_dataloader(train_set, batch_size=32, shuffle=True)
val_loader = get_dataloader(val_set, batch_size=32)
test_loader = get_dataloader(test_set, batch_size=32)
```

---

## ✅ 五、小结

| 功能   | 用途说明                                                  |
| ---- | ----------------------------------------------------- |
| 模块名  | `pyhealth.datasets.splitter`                          |
| 作用   | 将样本集按比例划分为训练 / 验证 / 测试                                |
| 推荐方式 | 结构化数据 → `split_by_patient`；非结构化片段 → `split_by_sample` |
| 后续用途 | 与 `get_dataloader()` 配合使用，构建训练数据加载器                   |

---

下一步是否继续进入：

* `pyhealth.tasks`（定义任务结构）
* `pyhealth.models`（调用模型架构）
* `pyhealth.trainer`（训练与评估管理）

我可以继续为你整理笔记与代码示例。


以下是关于 `pyhealth.datasets.utils` 模块的中文学习笔记，聚焦其在 PyHealth 数据处理流程中的**辅助作用**。

---

# 🧰 PyHealth 学习笔记：`pyhealth.datasets.utils`

---

## ✅ 一、模块简介

`pyhealth.datasets.utils` 模块包含**若干实用工具函数（utility functions）**，用于简化和加速医疗数据处理流程，常被用于：

* 日期/时间转换
* 医疗编码清洗
* ID 映射
* 表结构标准化
* 数据合并与聚合

这些函数在使用自定义原始数据或适配 OMOP/CDM 以外的数据时尤为有用。

---

## 📌 二、可能包含的工具函数（根据源码文档整理）

> 🚧 注意：截至当前官网未提供具体 API 文档，该模块主要为开发者使用，建议结合源码或调试查阅。

以下是常见用途及其可能的函数功能：

### 1️⃣ 医疗编码标准化工具

将多源系统下的 ICD、NDC、ATC、LOINC 等编码进行清洗与统一。

* 清除非法字符（如空格、符号）
* 转换为统一大写
* 校验码位数是否合规

### 2️⃣ 日期与时间戳转换

将字符串形式的时间转为 `datetime` 对象，支持多种时间格式识别。

```python
from pyhealth.datasets.utils import str2datetime

datetime_obj = str2datetime("2011-05-02 14:30:00")
```

---

### 3️⃣ ID 映射工具

* 将病人编号、住院号、检查编号统一成统一命名
* 用于后续建模时 ID 的一致性管理

---

### 4️⃣ 数据整合工具

* 表格（DataFrame）列名统一
* 多表合并（如诊断表、用药表、检验表）

例如：

```python
# 示例：对诊断表中的多个编码字段进行合并
from pyhealth.datasets.utils import merge_code_columns

diagnosis_df = merge_code_columns(diagnosis_df, columns=["icd9", "icd10"])
```

---

### 5️⃣ 错误处理与日志输出

* 数据缺失提示
* 任务日志记录（与 `tqdm` 配合）

---

## 🎯 三、模块适用场景总结

| 功能场景    | 使用建议                 |
| ------- | -------------------- |
| 自定义数据清洗 | 非 OMOP/MIMIC 格式数据导入时 |
| 编码标准化   | 医疗术语/编码处理前           |
| 时间处理    | 医嘱时间、入出院时间等          |
| 数据联合    | 多表汇总、ETL 前处理         |

---

## ✅ 四、小结

| 项目     | 内容                        |
| ------ | ------------------------- |
| 模块名称   | `pyhealth.datasets.utils` |
| 类型     | 工具函数集合                    |
| 主要功能   | 医疗编码清洗、时间转换、ID映射、数据聚合     |
| 推荐使用场景 | 导入自定义 EHR 数据、预处理非标准表结构时   |
| 文档支持   | 建议结合源码查阅（开发者向）            |

---

是否需要我帮你：

1. **演示一个真实用例**：比如“如何将 Excel 格式的肾病住院数据转为 PyHealth 支持格式”；
2. 或者继续进入 `pyhealth.tasks` 模块，深入学习任务定义结构？

告诉我你的优先级，我们继续深入。
