以下是关于 `pyhealth.data` 模块的**中文学习笔记（数据结构部分）**，这是 PyHealth 的核心基础，所有任务都围绕这套结构展开。

---

# 📦 PyHealth 数据结构详解：`pyhealth.data`

> `pyhealth.data` 模块定义了 PyHealth 中的**原子数据结构**，它是整个系统的底层基础。

主要包括：

1. `Event`（原子事件）
2. `Patient`（病人）
3. `Visit`（就诊，未在官网该节列出，但实际常用）

---

## 🧬 一、`Event`：原子医疗事件

每一个 `Event` 表示一次具体的医疗行为，比如：

* 一次实验室检查（如血肌酐 1.2）
* 一次诊断代码（如 ICD9=428.0）
* 一次用药行为（如使用阿司匹林）

### 🔖 属性

| 属性名          | 含义                           |
| ------------ | ---------------------------- |
| `event_type` | 事件类型（如 lab, condition, drug） |
| `timestamp`  | 时间戳，表示事件发生时间                 |
| `attr_dict`  | 事件的属性字典（如数值、代码名、单位等）         |

### 🧪 示例

```python
from pyhealth.data import Event
from datetime import datetime

event = Event(
    event_type="lab",
    timestamp=datetime(2022, 1, 10),
    attr_dict={"test": "creatinine", "value": 1.2, "unit": "mg/dL"}
)

print(event.event_type)   # "lab"
print(event.timestamp)    # 2022-01-10
print(event.attr_dict)    # {"test": "creatinine", "value": 1.2, "unit": "mg/dL"}
```

### 🔄 工具函数

* `Event.from_dict(d: dict)`：将字典转为 Event 实例

---

## 🧑‍⚕️ 二、`Patient`：病人结构

每个 `Patient` 对象代表一个病人，包含该病人的所有医疗信息（事件集合）。

### 🔖 属性

| 属性名                      | 含义                                      |
| ------------------------ | --------------------------------------- |
| `patient_id`             | 病人 ID                                   |
| `data_source`            | 来源（如 MIMIC-III、eICU、自定义）                |
| `event_type_partitions`  | 事件按照类型的组织，如 lab、condition 等             |
| `get_events(event_type)` | 获取某一类事件，如 `get_events("lab")` 返回所有实验室事件 |

### 🧪 示例

```python
from pyhealth.data import Patient

patient = Patient(
    patient_id="P001",
    data_source="MIMIC"
)

# 添加实验室事件
patient.add_event(
    event_type="lab",
    timestamp=datetime(2022, 1, 10),
    attr_dict={"test": "bun", "value": 12.5, "unit": "mg/dL"}
)

# 获取所有实验室检查记录
labs = patient.get_events("lab")
for event in labs:
    print(event.attr_dict)
```

---

## 🧩 总结：数据结构的逻辑关系图

```
Patient
 └── Visit（在任务中常用结构）
      └── Event
```

在训练模型时，PyHealth 会将 `Patient` 切分为多个 `Visit`，每次就诊是一个子序列，`Visit` 内包含多个 `Event`。

---

## 📚 学习建议

* 理解 `Event` 是最小单位，类似于“医疗词”
* `Patient` 是 EHR 数据的最小个体，构成预测样本的原材料
* 后续模型都建立在这两者之上（→ Task → Model）

---

如需我继续讲解 `Visit` 结构（或从数据中自动构建 visit 序列）或帮你基于这套结构写样例代码，也可以继续告诉我。是否继续进入 `pyhealth.data.Visit` 部分？


非常好，以下是你提供的 `pyhealth.data.Event` 类官方文档内容，整理成**中文学习笔记：结构化+示例形式**，方便长期复习与快速查阅。

---

# 🔬 PyHealth 学习笔记：`pyhealth.data.Event`

## ✅ 一、什么是 Event？

在 PyHealth 中，`Event` 是最基本的数据结构之一，表示一次**临床事件（clinical event）**。
它作为后续任务建模的最小单位，可表示：

| 示例类型    | 示例内容                   |
| ------- | ---------------------- |
| 药物事件    | 使用了“阿司匹林”              |
| 诊断事件    | ICD9 编码“428.0”（心力衰竭）   |
| 实验室检查事件 | 血肌酐值为 1.2 mg/dL，发生于某日期 |

---

## 🧱 二、类定义结构

```python
class pyhealth.data.Event
```

### 🔑 属性说明

| 属性名          | 类型         | 说明                                              |
| ------------ | ---------- | ----------------------------------------------- |
| `event_type` | `str`      | 表示事件类型，如 `"medication"`, `"diagnosis"`, `"lab"` |
| `timestamp`  | `datetime` | 表示事件发生时间                                        |
| `attr_dict`  | `dict`     | 存储该事件的细节，例如药物名称、检查值、单位等                         |

---

## 🧪 三、初始化示例

```python
from pyhealth.data import Event
from datetime import datetime

# 创建一个血肌酐实验室检查事件
event = Event(
    event_type="lab",
    timestamp=datetime(2022, 1, 15),
    attr_dict={
        "test": "creatinine",
        "value": 1.4,
        "unit": "mg/dL"
    }
)

print(event.event_type)    # lab
print(event.timestamp)     # 2022-01-15 00:00:00
print(event.attr_dict)     # {'test': 'creatinine', 'value': 1.4, 'unit': 'mg/dL'}
```

---

## 🔄 四、从字典创建 `Event`：`from_dict()`

如果你有来自数据库或 CSV 的数据行，可以使用 `from_dict()` 快速创建：

```python
# 模拟原始数据行
raw_event = {
    "event_type": "medication",
    "timestamp": datetime(2022, 2, 5),
    "attr_dict": {
        "drug_name": "aspirin",
        "dose": 100,
        "unit": "mg"
    }
}

# 使用类方法构造 Event 实例
event = Event.from_dict(raw_event)

print(event.event_type)    # medication
print(event.attr_dict)     # {'drug_name': 'aspirin', 'dose': 100, 'unit': 'mg'}
```

📌 注意：

* `from_dict()` 是一个类方法（classmethod）
* `attr_dict` 的内容没有强约束，支持任意结构，适合不同类型任务

---

## 📌 五、使用场景总结

| 场景        | 示例                                                            |
| --------- | ------------------------------------------------------------- |
| 表示一次实验室检查 | `"lab", timestamp, {"test": "BUN", "value": 12.5}`            |
| 表示一次诊断    | `"diagnosis", timestamp, {"icd9": "428.0"}`                   |
| 表示一次用药    | `"medication", timestamp, {"drug": "metformin", "dose": 500}` |

`Event` 常被组织进 `Visit`（一次就诊），或直接加入 `Patient` 中。

---

## ✅ 小结

| 特性   | 说明                     |
| ---- | ---------------------- |
| 灵活   | 支持任意医疗事件类型             |
| 可扩展  | `attr_dict` 可存储任意键值对   |
| 易构建  | 支持 `from_dict()` 从字典创建 |
| 基础作用 | 后续任务构建、模型输入依赖它         |

---

如你希望继续了解 `Visit`、`Patient` 的组织方式，或需要我帮你写一组 Event ➝ Visit ➝ Patient 的数据结构构造脚本，我可以继续补充下一节学习笔记。需要继续吗？


非常好！以下是你提供的 `pyhealth.data.Patient` 的官方文档内容，我已整理为**结构清晰、适合长期复习的中文学习笔记**。

---

# 👩‍⚕️ PyHealth 学习笔记：`pyhealth.data.Patient`

---

## ✅ 一、Patient 是什么？

`Patient` 是 PyHealth 中的基础结构之一，代表一个病人及其整个医疗事件记录集合。每个 `Patient`：

* 拥有唯一 ID（`patient_id`）
* 拥有若干次医疗就诊（通过事件序列体现）
* 可以作为任务模型的输入数据

---

## 🧱 二、类定义结构

```python
class pyhealth.data.Patient(patient_id, data_source)
```

| 参数名           | 类型                     | 说明              |
| ------------- | ---------------------- | --------------- |
| `patient_id`  | `str`                  | 病人唯一标识符         |
| `data_source` | `pl.DataFrame`（polars） | 该病人所有事件（按时间戳排序） |

---

## 🔑 三、属性详解

| 属性名                     | 类型                        | 说明                       |
| ----------------------- | ------------------------- | ------------------------ |
| `patient_id`            | `str`                     | 病人编号                     |
| `data_source`           | `pl.DataFrame`            | 存储所有事件的 Polars 表格        |
| `event_type_partitions` | `Dict[str, pl.DataFrame]` | 按事件类型（如 lab、drug）拆分后的事件表 |

---

## 🔄 四、方法：`get_events()`（核心方法）

### 方法签名：

```python
Patient.get_events(
    event_type=None,
    start=None,
    end=None,
    filters=None,
    return_df=False
)
```

### 功能：

获取指定类型 + 时间段 + 条件筛选的事件（返回事件对象或 DataFrame）

---

### ✨ 参数说明：

| 参数名             | 类型            | 说明                                            |
| --------------- | ------------- | --------------------------------------------- |
| `event_type`    | `str` 可选      | 限定获取哪类事件（如 `"lab"`）                           |
| `start` / `end` | `datetime`    | 限定时间范围                                        |
| `filters`       | `List[tuple]` | 属性过滤条件，如 `[("value", "!=", 0)]`               |
| `return_df`     | `bool`        | 若为 `True`，返回 `pl.DataFrame`；否则返回 `Event` 对象列表 |

---

### 🧪 示例

```python
from pyhealth.data import Patient
from datetime import datetime

# 初始化病人对象（构造空数据示例）
patient = Patient(patient_id="P001", data_source=None)

# 假设我们已经向 patient 添加了多次 lab 事件...

# 获取所有 lab 类型的事件
labs = patient.get_events(event_type="lab")

# 获取 2022 年之后的实验室事件
labs_2022 = patient.get_events(
    event_type="lab",
    start=datetime(2022, 1, 1)
)

# 获取值不为 0 的事件（自定义属性过滤）
labs_filtered = patient.get_events(
    event_type="lab",
    filters=[("value", "!=", 0)]
)
```

---

## 🧩 五、`Patient` 的典型使用流程（与 Event 联动）

```python
from pyhealth.data import Event, Patient
from datetime import datetime

# 创建病人
patient = Patient(patient_id="P001", data_source=None)

# 添加事件（模拟）
event1 = Event("lab", datetime(2022, 1, 1), {"test": "BUN", "value": 12.5})
event2 = Event("lab", datetime(2022, 1, 10), {"test": "Creatinine", "value": 1.3})

# 通过 add_event 添加（实际调用 task/dataset 时由系统自动添加）
patient.add_event(event1)
patient.add_event(event2)

# 获取事件
lab_events = patient.get_events(event_type="lab")
for e in lab_events:
    print(e.timestamp, e.attr_dict)
```

---

## 📌 六、使用技巧小结

| 技能           | 示例                                                      |
| ------------ | ------------------------------------------------------- |
| 获取某类事件       | `get_events("lab")`                                     |
| 设定时间段过滤      | `start=datetime(...)`                                   |
| 多条件过滤        | `filters=[("value", "!=", 0), ("unit", "==", "mg/dL")]` |
| 输出 DataFrame | `return_df=True`                                        |

---

## ✅ 总结

| 特性    | 描述                                  |
| ----- | ----------------------------------- |
| 核心作用  | 存储病人所有医疗事件                          |
| 典型结构  | Patient ⟶ \[Event, Event, Event...] |
| 可支持过滤 | 类型、时间、属性                            |
| 常配合使用 | `Visit`、`Event`、`Task`、`Dataset`    |

---

是否继续进入 `Visit` 的结构与使用说明？还是希望我将 Event + Patient 构造逻辑整理成完整的脚本样例（如用于自定义数据构建）？
