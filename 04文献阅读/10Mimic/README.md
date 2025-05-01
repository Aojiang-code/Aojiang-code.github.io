# MIMIC-IV

> https://physionet.org/content/mimiciv/3.1/#files-panel

> Published: Oct. 11, 2024. Version: 3.1
> 出版日期：2024年10月11日。版本：3.1


> MIMIC-IV v3.1 is now available on BigQuery (Nov. 12, 2024, 4:50 p.m.)
> MIMIC-IV v3.1 现已在 BigQuery 上发布 （2024 年 11 月 12 日下午 4：50）

> MIMIC-IV v3.1 is now available on BigQuery. Users may request access via PhysioNet. Currently, MIMIC-IV v3.1 is available on the mimiciv_v3_1_hosp and mimiciv_v3_1_icu schemas. MIMIC-IV v2.2 is available on the mimiciv_v2_2_hosp and mimiciv_v2_2_icu datasets as well as the mimiciv_hosp and mimiciv_icu datasets. On November 25th 2024, we will replace the data on mimiciv_hosp and mimiciv_icu with MIMIC-IV v3.1.
> MIMIC-IV v3.1 现已在 BigQuery 上提供。用户可以通过 PhysioNet 请求访问。目前，MIMIC-IV v3.1 可在 mimiciv_v3_1_hosp 和 mimiciv_v3_1_icu 模式中获得。MIMIC-IV v2.2 可在 mimiciv_v2_2_hosp 和 mimiciv_v2_2_icu 数据集以及 mimiciv_hosp 和 mimiciv_icu 数据集上获得。2024 年 11 月 25 日，我们将用 MIMIC-IV v3.1 替换 mimiciv_hosp 和 mimiciv_icu 上的数据。

> Guidelines for creating datasets and models from MIMIC (April 24, 2024, 10:12 a.m.)
> 从 MIMIC 创建数据集和模型的指南 （2024 年 4 月 24 日上午 10：12）

> We recognize that there is value in creating datasets or models that are either derived from MIMIC or which augment MIMIC in some way (for example, by adding annotations). Here are some guidelines on creating these datasets and models:
> 我们认识到，创建源自 MIMIC 或以某种方式（例如，通过添加注释）增强 MIMIC 的数据集或模型是有价值的。以下是创建这些数据集和模型的一些准则：

> Any derived datasets or models should be treated as containing sensitive information. If you wish to share these resources, they should be shared on PhysioNet under the same agreement as the source data.
> 任何衍生数据集或模型都应视为包含敏感信息 。如果您希望共享这些资源，则应根据与源数据相同的协议在 PhysioNet 上共享。
> If you would like to use the MIMIC acronym in your project name, please include the letters “Ext” (for example, MIMIC-IV-Ext-YOUR-DATASET"). Ext may either indicate “extracted” (e.g. a derived subset) or “extended” (e.g. annotations), depending on your use case.
> 如果您希望在项目名称中使用 MIMIC 首字母缩写 ，请包含字母“Ext”（例如，MIMIC-IV-Ext-YOUR-DATASET”）。Ext 可以表示“提取”（例如派生子集）或“扩展”（例如注释），这取决于您的用例。
> Please select the relevant "Parent Projects" in the Discovery tab of the submission portal when preparing your project for submission.
> 请在准备提交项目时，在提交门户的发现选项卡中选择相关的“父项目”。


## Abstract  摘要
Retrospectively collected medical data has the opportunity to improve patient care through knowledge discovery and algorithm development. Broad reuse of medical data is desirable for the greatest public good, but data sharing must be done in a manner which protects patient privacy. Here we present Medical Information Mart for Intensive Care (MIMIC)-IV, a large deidentified dataset of patients admitted to the emergency department or an intensive care unit at the Beth Israel Deaconess Medical Center in Boston, MA. MIMIC-IV contains data for over 65,000 patients admitted to an ICU and over 200,000 patients admitted to the emergency department. MIMIC-IV incorporates contemporary data and adopts a modular approach to data organization, highlighting data provenance and facilitating both individual and combined use of disparate data sources. MIMIC-IV is intended to carry on the success of MIMIC-III and support a broad set of applications within healthcare.

逆向收集的医疗数据有机会通过知识发现和算法开发来改善患者护理。为了最大的公共利益，医疗数据的广泛重用是可取的，但数据共享必须以保护患者隐私的方式进行。在这里，我们提出了重症监护医疗信息市场（MIMIC）-IV，一个大型的去识别数据集的患者入院急诊科或重症监护室在贝丝以色列女执事医疗中心在波士顿，MA。MIMIC-IV 包含超过 65，000 名 ICU 患者和超过 200，000 名急诊患者的数据。MIMIC-IV 纳入了当代数据，并采用了模块化的数据组织方法，突出数据出处，促进单独和合并使用不同的数据源。MIMIC-IV 旨在继承 MIMIC-III 的成功，并支持医疗保健领域的广泛应用。

## Background  背景
In recent years there has been a concerted move towards the adoption of digital health record systems in hospitals. In the US, nearly 96% of hospitals had a digital electronic health record system (EHR) in 2015 [1]. Retrospectively collected medical data has increasingly been used for epidemiology and predictive modeling. The latter is in part due to the effectiveness of modeling approaches on large datasets [2]. Despite these advances, access to medical data to improve patient care remains a significant challenge. While the reasons for limited sharing of medical data are multifaceted, concerns around patient privacy are highlighted as one of the most significant issues. Although patient studies have shown almost uniform agreement that deidentified medical data should be used to improve medical practice, domain experts continue to debate the optimal mechanisms of doing so. Uniquely, the MIMIC-III database adopted a permissive access scheme which allowed for broad reuse of the data [3]. This mechanism has been successful in the wide use of MIMIC-III in a variety of studies ranging from assessment of treatment efficacy in well defined cohorts to prediction of key patient outcomes such as mortality. MIMIC-IV aims to carry on the success of MIMIC-III, with a number of changes to improve usability of the data and enable more research applications.

近年来，医院采取了一致行动，采用数字健康记录系统。在美国，2015 年近 96%的医院拥有数字电子健康记录系统（EHR）。回溯收集的医疗数据已越来越多地用于流行病学和预测建模。后者部分是由于大型数据集上建模方法的有效性[2]。尽管取得了这些进展，但获取医疗数据以改善患者护理仍然是一项重大挑战。虽然医疗数据共享有限的原因是多方面的，但围绕患者隐私的担忧被强调为最重要的问题之一。尽管患者研究表明，去识别化的医疗数据应用于改善医疗实践，但领域专家仍在继续辩论这样做的最佳机制。独特的是，MIMIC-III 数据库采用了一种允许广泛重复使用数据的许可访问方案[3]。 该机制已成功地在各种研究中广泛使用 MIMIC-III，从明确定义的队列中的治疗疗效评估到关键患者结局（如死亡率）的预测。MIMIC-IV 旨在继承 MIMIC-III 的成功，并进行了一些更改，以提高数据的可用性，并使更多的研究应用成为可能。

## Methods  方法
MIMIC-IV is sourced from two in-hospital database systems: a custom hospital wide EHR and an ICU specific clinical information system. The creation of MIMIC-IV was carried out in three steps:

MIMIC-IV 来源于两个医院内数据库系统：一个定制的医院范围的 EHR 和一个 ICU 特定的临床信息系统。MIMIC-IV 的创建分三个步骤进行：

Acquisition. Data for patients who were admitted to the BIDMC emergency department or one of the intensive care units were extracted from the respective hospital databases. A master patient list was created which contained all medical record numbers corresponding to patients admitted to an ICU or the emergency department between 2008 - 2022. All source tables were filtered to only rows related to patients in the master patient list.

采集从各自的医院数据库中提取了入住 BIDMC 急诊科或重症监护室的患者数据。创建了一个主患者列表，其中包含 2008 年至 2022 年期间入住 ICU 或急诊科的患者的所有医疗记录编号。将所有源表过滤为仅与主患者列表中的患者相关的行。

Preparation. The data were reorganized to better facilitate retrospective data analysis. This included the denormalization of tables, removal of audit trails, and reorganization into fewer tables. The aim of this process is to simplify retrospective analysis of the database. Importantly, data cleaning steps were not performed, to ensure the data reflects a real-world clinical dataset.

准备. 对数据进行了重组，以更好地促进回顾性数据分析。这包括表的非规范化、删除审计跟踪以及重组为更少的表。这一过程的目的是简化数据库的回顾性分析。重要的是，未执行数据清理步骤，以确保数据反映真实世界的临床数据集。

Deidentify. Patient identifiers as stipulated by HIPAA were removed. Patient identifiers were replaced using a random cipher, resulting in deidentified integer identifiers for patients, hospitalizations, and ICU stays. Structured data were filtered using look up tables and allow lists. If necessary, a free-text deidentification algorithm was applied to remove PHI from free-text. Finally, date and times were shifted randomly into the future using an offset measured in days. A single date shift was assigned to each subject_id. As a result, the data for a single patient are internally consistent. For example, if the time between two measures in the database was 4 hours in the raw data, then the calculated time difference in MIMIC-IV will also be 4 hours. Conversely, distinct patients are not temporally comparable. That is, two patients admitted in 2130 were not necessarily admitted in the same year.

否认身份。HIPAA 规定的患者标识符已被删除。使用随机密码替换患者标识符，导致患者、住院和 ICU 停留的去识别整数标识符。使用查找表和允许列表过滤结构化数据。如有必要，应用自由文本去识别算法从自由文本中删除 PHI。最后，使用以天为单位的偏移量将日期和时间随机转移到未来。为每个 subject_id 分配一个日期偏移。因此，单个患者的数据是内部一致的。例如，如果原始数据中数据库中两次测量之间的时间为 4 小时，则 MIMIC-IV 中计算的时间差也为 4 小时。相反，不同的患者在时间上不具有可比性。也就是说，2130 年收治的两名患者不一定是同一年收治的。

After these three steps were carried out, the database was exported to a character based comma delimited format.

完成这三个步骤后，数据库被导出为基于字符的逗号分隔格式。


## Data Description  数据描述
MIMIC-IV is grouped into two modules: hosp, and icu. Organization of the data into these modules reflects their provenance: data in the hosp module is sourced from the hospital wide EHR, while data in the icu module is sourced from the in-ICU clinical information system (MetaVision). A total of 364,627 unique individuals are in MIMIC-IV v3.0, each represented by a unique subject_id. These individuals had 546,028 hospitalizations and 94,458 unique ICU stays.

MIMIC-IV 分为两个模块：hosp 和 icu。将数据组织到这些模块中反映了它们的来源：hosp 模块中的数据来自医院范围的 EHR，而  icu 模块中的数据来自 ICU 临床信息系统（MetaVision）。MIMIC-IV v3.0 中共有 364，627 个独特的个体，每个个体由一个独特的 subject_id 表示。这些个体有 546，028 次住院和 94，458 次独特的 ICU 住院。


### hosp
The hosp module contains detailed data regarding 546,028 unique hospitalizations for 223,452 unique individuals. Measurements in the hosp module are predominantly recorded during the hospital stay, though some tables include data from outside an admitted hospital stay as well (e.g. outpatient or emergency department laboratory tests in labevents). Patient demographics (patients), hospitalizations (admissions), and intra-hospital transfers (transfers) are recorded in the hosp module. Other information in the hosp module includes laboratory measurements (labevents, d_labitems), microbiology cultures (microbiologyevents, d_micro), provider orders (poe, poe_detail), medication administration (emar, emar_detail), medication prescription (prescriptions, pharmacy), hospital billing information (diagnoses_icd, d_icd_diagnoses, procedures_icd, d_icd_procedures, hcpcsevents, d_hcpcs, drgcodes), online medical record data (omr), and service related information (services).

hosp 模块包含关于 223，452 个独特个体的 546，028 次独特住院的详细数据。hosp 模块中的测量值主要记录在住院期间，尽管有些表格也包括住院期间以外的数据（例如， 实验室事件中的门诊或急诊科实验室检查）。患者人口统计数据（ 患者 ）、住院（ 入院 ）和院内转移（ 转移 ）记录在 hosp 模块中。 hosp 模块中的其他信息包括实验室测量（labevents、d_labitems）、微生物培养（microbiologyevents、d_micro）、提供者订单（poe、poe_detail）、药物管理（emar、emar_detail）、药物处方（prescriptions、pharmacy）、医院账单信息（diagnosis_icd、d_icd_diagnosis、procedures_icd、d_icd_procedures、hcpcsevents、d_hcpcs、drgcodes）、在线医疗记录数据（omr）和服务相关信息（services）。

Provider information is available in the provider table. The provider_id column is a deidentified character string which uniquely represents a single care provider. As provider_id is used in different contexts across the module, a prefix is usually present in data tables to contextualize how the provider relates to the event. For example, the provider who admits the patient to the hospital is documented in the admissions table as subject_id. All columns which have a suffix of provider_id may be linked to the provider table.

提供程序信息在提供程序表中提供。provider_id 列是唯一表示单个护理提供者的去识别字符串。由于 provider_id 在模块中的不同上下文中使用，因此数据表中通常会出现一个前缀，以说明提供程序与事件的关联方式。例如，允许患者入院的提供者在入院表中记录为 subject_id。所有后缀为 provider_id 的列都可以链接到 provider 表。


### Deidentified dates and aligning stays to year groups    去标识日期并将住宿与年份组对齐
All dates in MIMIC-IV have been deidentified by shifting the dates into a future time period between 2100 - 2200. This shift is done independently for each patient, and as a result two patients admitted in the deidentified year 2120 cannot be assumed to be admitted in the same year. To provide information about the original time period when a patient was admitted, the patients table provides a set of columns with the "anchor_" prefix. The anchor_year column is a deidentified year occurring sometime between 2100 - 2200, and the anchor_year_group column is one of the following values: "2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", and "2020 - 2022". These pieces of information allow researchers to infer the approximate year a patient received care. For example, if a patient's anchor_year is 2158, and their anchor_year_group is 2011 - 2013, then any hospitalizations for the patient occurring in the year 2158 actually occurred sometime between 2011 - 2013. In order to minimize accidental release of information, only a single anchor_year is provided per subject_id. Consequently, individual stays must be aligned to the anchor year using the respective date (e.g. admittime). Finally, the anchor_age provides the patient age in the given anchor_year. If the patient was over 89 in the anchor_year, this anchor_age has been set to 91 (i.e. all patients over 89 have been grouped together into a single group with value 91, regardless of what their real age was).

MIMIC-IV 中的所有日期都已通过将日期转移到 2100 - 2200 之间的未来时间段来进行识别。这一转变对每个病人都是独立进行的，因此，在 2120 年被取消身份的两名病人不能被认为是在同一年被收治的。为了提供有关患者入院的原始时间段的信息， 患者表提供了一组带有“锚_”前缀的列。 锚_年列是发生在 2100 - 2200 之间的某个时间的去识别年份，并且锚_年_组列是以下值之一：“2008 - 2010”、“2011 - 2013”、“2014 - 2016”、“2017 - 2019”和“2020 - 2022”。这些信息使研究人员能够推断出患者接受治疗的大约年份。 例如，如果患者的锚_年是 2158，并且他们的锚_年_组是 2011 - 2013，则患者在 2158 年发生的任何住院实际上发生在 2011 - 2013 之间的某个时间。为了尽量减少信息的意外泄露，每个 subject_id 仅提供单个锚_年 。因此，必须使用相应的日期（例如， 入院时间 ）将个人住宿与锚年对齐。最后， 锚_年龄提供给定锚_年的患者年龄。如果患者在锚_年中超过 89 岁，则该锚_年龄已设置为 91（即，所有超过 89 岁的患者已被分组到一个值为 91 的组中，无论其真实的年龄如何）。

### Out of hospital linkage of date of death    死亡日期的院外联系
Date of death is available within the dod column of the patients table. Date of death is derived from hospital records and state records. If both exist, hospital records take precedence. State records were matched using a custom rule based linkage algorithm based on name, date of birth, and social security number. State and hospital records for date of death were collected two years after the last patient discharge in MIMIC-IV, which should limit the impact of reporting delays in date of death.
死亡日期在 patients 表的 dod 列中提供。死亡日期来自医院记录和州记录。如果两者都存在，则医院记录优先。使用基于自定义规则的链接算法，根据姓名，出生日期和社会安全号码匹配州记录。在 MIMIC-IV 中，在末例患者出院后 2 年收集了死亡日期的州和医院记录，这应限制死亡日期报告延迟的影响。

Dates of death occurring more than one year after hospital discharge are censored as a part of the deidentification process. As a result, the maximum time of follow up for each patient is exactly one year after their last hospital discharge. For example, if a patient's last hospital discharge occurs on 2150-01-01, then the last possible date of death for the patient is 2151-01-01. If the individual died on or before 2151-01-01, and it was captured in either state or hospital death records, then the dod column will contain the deidentified date of death. If the individual survived for at least one year after their last hospital discharge, then the dod column will have a NULL value.
出院后一年以上发生的死亡日期将被审查，作为去身份化过程的一部分。因此，每名病人的最长随访时间是他们最后一次出院后的一年。例如，如果患者的最后一次出院发生在 2150-01-01，则患者的最后可能死亡日期为 2151-01-01。如果个人在 2151-01-01 或之前死亡，并且在州或医院死亡记录中被记录，那么 dod 栏将包含去识别的死亡日期。如果患者在最后一次出院后存活了至少一年，则 dod 列将具有 NULL 值。


### icu
The icu module contains data sourced from the clinical information system known as MetaVision (iMDSoft). MetaVision tables were denormalized to create a star schema where the icustays and d_items tables link to a set of data tables all suffixed with "events". Data documented in the icu module includes intravenous and fluid inputs (inputevents), ingredients for the aforementioned inputs (ingredientevents), patient outputs (outputevents), procedures (procedureevents), information documented as a date or time (datetimeevents), and other charted information (chartevents). All events tables contain a stay_id column allowing identification of the associated ICU patient in icustays, and an itemid column allowing identification of the concept documented in d_items. Additionally, the caregiver table contains caregiver_id, a deidentified integer representing the care provider who documented data into the system. All events tables (chartevents, datetimeevents, ingredientevents, inputevents, outputevents, procedureevents) have a caregiver_id column which links to the caregiver table.
icu 模块包含来自临床信息系统 MetaVision（iMDSoft）的数据。MetaVision 表被反规范化以创建一个星星模式，其中 icustays 和 d_items 表链接到一组数据表，所有数据表都以“events”为后缀。icu 模块中记录的数据包括静脉和液体输入（ 输入事件 ），上述输入的成分（ 成分事件）， 患者输出（ 输出事件 ），程序（ 程序事件 ），记录为日期或时间的信息（ 日期时间事件 ）和其他图表信息（ 图表事件 ）。所有事件表都包含 stay_id 列和 itemid 列，stay_id 列用于标识 icustays 中的关联 ICU 患者，itemid 列用于标识 d_items 中记录的概念。 此外， 护理人员表包含 carefully_id，一个去识别的整数，表示将数据记录到系统中的护理提供者。所有事件表（chartevents、datetimeevents、ingredientevents、inputevents、outputevents、procedureevents）都有一个链接到护理人员表的 carefully_id 列。

The icu module contains a total of 94,458 ICU stays for 65,366 unique individuals as of MIMIC-IV v3.0. An ICU stay is defined as a contiguous sequence of transfers within a unit of the hospital classified as an ICU, and the icustays table is derived from the transfers table. During the creation of the icustays table, consecutive transfers within an ICU were merged into the same stay_id for analytical convenience, as these transfers are often bed number changes. Importantly, non-consecutive ICU stays remain as unique stay_id in the icustays table. In some cases, these could be considered the "same" ICU stay as the patient was transferred out for a planned procedure. In other cases, these are unanticipated readmissions to the ICU. As there was no systematically perfect method to differentiate these cases, we did not attempt to merge non-consecutive stay_id, and it is up to the investigator to appropriately handle these cases.
截至 MIMIC-IV v3.0，icu 模块包含 65，366 个独特个体的总计 94，458 次 ICU 停留。ICU 住院被定义为在被分类为 ICU 的医院单元内的连续转移序列，并且 ICU 住院表从转移表导出。在创建 icustays 表的过程中，ICU 内的连续转移被合并到同一 stay_id 中以便于分析，因为这些转移通常是床号更改。重要的是，非连续 ICU 停留在 icustays 表中保持为唯一 stay_id。在某些情况下，这些可以被认为是“相同的”ICU 停留，因为患者被转出进行计划的手术。在其他情况下，这些都是意外再入院 ICU。 由于没有系统的完美方法来区分这些病例，我们没有尝试合并非连续 stay_id，由研究者适当处理这些病例。


## Usage Notes  用法注释
The data described here are collected during routine clinical practice and reflect the idiosyncrasies of that practice. Implausible values may be present in the database as an artifact of the archival process.  Researchers should follow best practice guidelines when analyzing the data.
这里描述的数据是在常规临床实践中收集的，反映了该实践的特质。不真实的值可能作为归档过程的工件存在于数据库中。研究人员在分析数据时应遵循最佳实践指南。

### Documentation  文件
Up to date documentation for MIMIC-IV is available on the MIMIC-IV website [4]. We have created an open source repository for the sharing of code and discussion of the database, referred to as the MIMIC Code Repository [5, 6]. The code repository provides a mechanism for shared discussion and analysis of all versions of MIMIC, including MIMIC-IV.
MIMIC-IV 的最新文档可在 MIMIC-IV 网站上获得[4]。我们已经创建了一个用于共享代码和讨论数据库的开源存储库，称为 MIMIC 代码存储库[5，6]。代码存储库提供了一种共享讨论和分析所有版本的 MIMIC（包括 MIMIC-IV）的机制。

### Linking MIMIC-IV to emergency department, note, and chest x-ray data
将 MIMIC-IV 与急诊科、笔记和胸部 X 射线数据相关联
MIMIC-IV is linkable to other MIMIC projects published on PhysioNet. Where possible, we have prefixed the other projects with "MIMIC-IV" to make this clear such as MIMIC-IV-ED. Note that MIMIC-CXR is also linkable although it is not prefixed with MIMIC-IV. Free-text clinical notes are available in MIMIC-IV-Note [7], observations made in the emergency department are available in MIMIC-IV-ED [8], and chest x-rays in MIMIC-CXR [9].
MIMIC-IV 是发布在 PhysioNet 上的其他 MIMIC 项目的一部分。在可能的情况下，我们已经用“MIMIC-IV”作为前缀，以明确这一点，如 MIMIC-IV-ED。请注意，MIMIC-CXR 也是可扩展的，尽管它没有前缀 MIMIC-IV。自由文本临床记录见 MIMIC-IV-Note [7]，急诊室观察结果见 MIMIC-IV-艾德[8]，胸部 X 线片见 MIMIC-CXR [9]。

Linking the other datasets to MIMIC-IV requires two steps. The first step is to match the data using subject_id, taking care to note that MIMIC-IV is a superset of other modules, and sampling biases may be introduced by the linking process. For example, MIMIC-CXR is only available between 2011 - 2016 for patients who were admitted to the emergency department, and this selection bias impacts the patient cohort. The second step involves aligning the dates. Since all modules are deidentified by the same shift, the time periods for measurements overlap. For example, if a patient is admitted to the hospital on 2105-01-01, discharged on 2105-01-03, and has an x-ray in MIMIC-CXR on 2105-01-02, then it is correct to assume the x-ray was taken while the patient was admitted to the hospital.
将其他数据集链接到 MIMIC-IV 需要两个步骤。第一步是使用 subject_id 匹配数据，注意 MIMIC-IV 是其他模块的超集，链接过程可能会引入采样偏差。例如，MIMIC-CXR 仅适用于 2011 - 2016 年急诊科收治的患者，这种选择偏倚影响了患者队列。第二步是调整日期。由于所有模块都通过相同的移位去识别，因此测量的时间段重叠。例如，如果患者于 2105-01-01 入院，于 2105-01-03 出院，并于 2105-01-02 在 MIMIC-CXR 中进行了 X 射线检查，则可以正确假设 X 射线检查是在患者入院时进行的。

### Patient composition  患者组成
MIMIC-IV contains patients admitted to the emergency department and the intensive care unit. While patients admitted to the intensive care unit must have an associated hospitalization, patients may be admitted to the emergency department without being subsequently admitted to the hospital. As a result, the number of patients in MIMIC-IV is much higher than the number of unique patients with hospitalizations. As of MIMIC-IV v3.0 there are 364,627 unique patients, of whom 223,452 had at least one hospitalization (i.e. at least one record in the admissions table). The remaining 141,175 patients were only seen in the emergency department, which can be verified using the transfers table.
MIMIC-IV 包含急诊科和重症监护室收治的患者。虽然入住重症监护室的患者必须有相关的住院治疗，但患者可能会被急诊科收治，而无需随后入院。因此，MIMIC-IV 的患者数量远远高于住院的独特患者数量。截至 MIMIC-IV v3.0，有 364，627 名独特患者，其中 223，452 人至少有一次住院（即入院表中至少有一条记录）。其余 141，175 名患者仅在急诊科就诊，可使用转诊表进行核实。



# Medical Information Mart for Intensive Care

https://mimic.mit.edu/





# Mimic肾衰竭

以下是一个可以在PubMed上使用的英文检索式，用于搜索与“Mimic”“机器学习”“肾衰竭”相关的文献：

**检索式：**
```
(Mimic[Title/Abstract] OR MIMIC[Title/Abstract]) AND (Machine Learning[Title/Abstract]) AND (Renal Failure[Title/Abstract] OR Kidney Failure[Title/Abstract])
```

### 说明：
1. **Mimic**：MIMIC数据库是常用的医疗数据集，用于机器学习和临床研究。
2. **Machine Learning**：直接搜索“机器学习”相关的研究。
3. **Renal Failure/Kidney Failure**：涵盖了“肾衰竭”的不同表达方式。


## 检索结果简介


### 文章1：Predicting the risk of acute kidney injury in patients with acute pancreatitis complicated by sepsis using a stacked ensemble machine learning model: a retrospective study based on the MIMIC database
**使用堆叠集成机器学习模型预测急性胰腺炎并发脓毒症患者的急性肾损伤风险：基于 MIMIC 数据库的回顾性研究**
- **期刊**：BMJ Open（医学4区）
- **发表日期**：2025年2月26日
- **研究目的**：开发并验证一种堆叠集成机器学习模型，用于预测急性胰腺炎并发脓毒症患者的急性肾损伤（AKI）风险。
- **研究设计**：基于公共数据库（MIMIC数据库）的回顾性研究。
- **研究对象**：美国重症监护数据库中的1295例急性胰腺炎并发败血症患者。
- **研究方法**：使用Boruta算法选择变量，构建8种机器学习算法模型，并开发新的堆叠集成模型（Multimodel）。通过AUC、PR曲线、准确度、召回率和F1评分评估模型性能。
- **主要结果**：急性胰腺炎并发脓毒症患者的AKI。
- **研究结果**：最终纳入1295例患者，其中893例（68.9%）发生AKI。Multimodel在内部验证数据集中的AUC值为0.853（95% CI：0.792-0.896），外部验证数据集为0.802（95% CI：0.732-0.861），表现出最佳预测性能。
- **结论**：该堆叠集成模型在内部和外部验证中表现出色，是预测急性胰腺炎并发脓毒症患者AKI的可靠工具。
- **关键词**：急性肾衰竭；成人重症监护；人工智能；机器学习；胰腺疾病；回顾性研究。

### 文章2：Severe acute kidney injury predicting model based on transcontinental databases: a single-centre prospective study
**基于洲际数据库的严重急性肾损伤预测模型的单中心前瞻性研究**
- **期刊**：BMJ Open（医学4区）
- **发表日期**：2022年3月3日
- **研究目的**：使用三个数据库构建模型，预测重症监护病房（ICU）患者48小时内的严重急性肾损伤（AKI）。
- **研究设计**：回顾性和前瞻性队列研究。
- **研究对象**：纳入三个数据库（SHZJU-ICU、MIMIC和AmsterdamUMC）的患者。
- **主要结果**：预测模型效果评价指标。
- **研究结果**：纳入58492例患者，5257例（9.0%）符合重度AKI定义。内部验证中模型最佳AUROC为0.86，外部验证AUROC为0.86。前瞻性验证中灵敏度为0.72，特异性为0.80，AUROC为0.84。
- **结论**：该预测模型基于多中心数据库的动态生命体征和实验室结果，通过前瞻性和外部验证，有望成为临床应用工具。
- **关键词**：成人重症监护；急性肾衰竭；资讯科技；重症监护。

### 文章3：Machine learning for real-time prediction of complications in critical care: a retrospective study
**机器学习用于重症监护并发症的实时预测：一项回顾性研究**
- **期刊**：Lancet Respir Med（医学1区）
- **发表日期**：2018年12月
- **研究目的**：应用深度机器学习方法预测心胸手术后重症监护期间的严重并发症。
- **研究方法**：使用递归神经网络预测术后死亡率、肾衰竭和术后出血。主要数据集为德国三级心血管疾病护理中心的患者数据，外部验证使用MIMIC-III数据集。
- **研究结果**：纳入11492例患者。深度学习模型预测的PPV和敏感性分别为：死亡率0.90和0.85，肾衰竭0.87和0.94，出血0.84和0.74。预测性能显著优于标准临床工具。
- **结论**：深度学习方法在重症监护中的应用可显著提高并发症预测准确性，有望改善临床护理。
- **资金来源**：无具体资金。

### 文章4：Fast and interpretable mortality risk scores for critical care patients
**重症监护患者的快速和可解释的死亡风险评分**
- **期刊**：J Am Med Inform Assoc（医学2区，管理科学2区）
- **发表日期**：2025年4月1日
- **研究目的**：开发一种既准确又可解释的死亡风险评分模型。
- **研究方法**：开发了GroupFasterRisk算法，利用MIMIC III和eICU数据集进行评估。
- **研究结果**：GroupFasterRisk模型优于OASIS和SAPS II评分，与APACHE IV/IVa相当，但参数更少。对于脓毒症、急性心肌梗死、心力衰竭和急性肾衰竭患者，该模型表现更好。
- **结论**：GroupFasterRisk是一种快速、灵活且可解释的程序，可用于死亡率预测。
- **关键词**：可解释的AI；可解释的机器学习；死亡风险；风险评分；稀疏性。

### 文章5：[Predictive value of machine learning for in-hospital mortality for trauma-induced acute respiratory distress syndrome patients: an analysis using the data from MIMIC III]
**机器学习对创伤引起的急性呼吸窘迫综合征患者住院死亡率的预测价值：使用 MIMIC III 数据的分析**
- **期刊**：Zhonghua Wei Zhong Bing Ji Jiu Yi Xue
- **发表日期**：2022年3月
- **研究目的**：探讨机器学习方法对创伤性ARDS患者院内死亡率的预测价值。
- **研究方法**：从MIMIC III数据库中提取数据，使用Logistic回归、XGBoost和人工神经网络模型进行预测。
- **研究结果**：纳入760例患者，Logistic回归、XGBoost和人工神经网络模型的AUC分别为0.737、0.745和0.757，差异无统计学意义。
- **结论**：这些模型对创伤性ARDS患者的院内死亡率有较好的预测价值。

### 文章6：Development and validation of a novel risk-predicted model for early sepsis-associated acute kidney injury in critically ill patients: a retrospective cohort study
**危重患者早期脓毒症相关急性肾损伤的新型风险预测模型的开发和验证：一项回顾性队列研究**
- **期刊**：BMJ Open（医学4区）
- **发表日期**：2025年1月28日
- **研究目的**：开发一种用于检测早期脓毒症相关急性肾损伤（SA-AKI）的预测模型。
- **研究设计**：回顾性队列研究。
- **研究对象**：开发队列纳入7179例脓毒症患者，外部验证队列纳入269例患者。
- **研究结果**：最终模型包括12个危险因素，梯度增强机器模型表现最佳，AUC分别为0.794、0.725和0.707。
- **结论**：该基于网络的临床预测模型是预测脓毒症重症患者早期SA-AKI的可靠工具，需要进一步验证。
- **关键词**：急性肾衰竭；成人重症监护；重症监护。

### 文章7：Multimorbidity states associated with higher mortality rates in organ dysfunction and sepsis: a data-driven analysis in critical care
**与器官功能障碍和脓毒症死亡率较高的相关的多发病状态：重症监护中的数据驱动分析**
- **期刊**：Crit Care（医学1区）
- **发表日期**：2019年7月8日
- **研究目的**：分析多发病状态与器官功能障碍、脓毒症和死亡率的关联。
- **研究方法**：分析MIMIC III数据集中的36390名患者，使用潜在类别分析识别患者亚组。
- **研究结果**：识别出六个不同的多发病亚组，其中“肝脏/成瘾”亚组的不良结局最为普遍。
- **结论**：多发病状态与器官功能障碍、脓毒症和死亡率的高患病率相关，应将其纳入医疗保健模式中。
- **关键词**：数据分析；潜在类别分析；机器学习；多发病；脓毒症。




## 阅读

01Predicting the risk of acute kidney injury in patients with acute pancreatitis complicated by sepsis using a stacked ensemble machine learning model: a retrospective study based on the MIMIC database
01使用堆叠集成机器学习模型预测急性胰腺炎并发脓毒症患者的急性肾损伤风险：基于 MIMIC 数据库的回顾性研究

Severe acute kidney injury predicting model based on transcontinental databases: a single-centre prospective study
基于洲际数据库的严重急性肾损伤预测模型的单中心前瞻性研究


02Machine learning for real-time prediction of complications in critical care: a retrospective study
02机器学习用于重症监护并发症的实时预测：一项回顾性研究

03Fast and interpretable mortality risk scores for critical care patients
03重症监护患者的快速和可解释的死亡风险评分

[Predictive value of machine learning for in-hospital mortality for trauma-induced acute respiratory distress syndrome patients: an analysis using the data from MIMIC III]
[机器学习对创伤引起的急性呼吸窘迫综合征患者住院死亡率的预测价值：使用 MIMIC III 数据的分析]

04Development and validation of a novel risk-predicted model for early sepsis-associated acute kidney injury in critically ill patients: a retrospective cohort study
04危重患者早期脓毒症相关急性肾损伤的新型风险预测模型的开发和验证：一项回顾性队列研究

Multimorbidity states associated with higher mortality rates in organ dysfunction and sepsis: a data-driven analysis in critical care
与器官功能障碍和脓毒症死亡率较高相关的多发病状态：重症监护中的数据驱动分析



## 文献详情


---

**BMJ Open. 2025 Feb 26;15(2):e087427. doi: 10.1136/bmjopen-2024-087427.**

**Predicting the risk of acute kidney injury in patients with acute pancreatitis complicated by sepsis using a stacked ensemble machine learning model: a retrospective study based on the MIMIC database**

**使用堆叠集成机器学习模型预测急性胰腺炎并发脓毒症患者的急性肾损伤风险：基于 MIMIC 数据库的回顾性研究**

**Objective:** This study developed and validated a stacked ensemble machine learning model to predict the risk of acute kidney injury in patients with acute pancreatitis complicated by sepsis.

**目的：** 本研究开发并验证了一种堆叠集成机器学习模型，用于预测急性胰腺炎并发脓毒症患者的急性肾损伤风险。

**Design:** A retrospective study based on patient data from public databases.

**设计：** 基于公共数据库中患者数据的回顾性研究。

**Participants:** This study analysed 1295 patients with acute pancreatitis complicated by septicaemia from the US Intensive Care Database.

**参会人员：** 本研究分析了美国重症监护数据库中 1295 例急性胰腺炎并发败血症患者。

**Methods:** From the MIMIC database, data of patients with acute pancreatitis and sepsis were obtained to construct machine learning models, which were internally and externally validated. The Boruta algorithm was used to select variables. Then, eight machine learning algorithms were used to construct prediction models for acute kidney injury (AKI) occurrence in intensive care unit (ICU) patients. A new stacked ensemble model was developed using the Stacking ensemble method. Model evaluation was performed using area under the receiver operating characteristic curve (AUC), precision-recall (PR) curve, accuracy, recall and F1 score. The Shapley additive explanation (SHAP) method was used to explain the models.

**研究方法：** 从 MIMIC 数据库中获取急性胰腺炎和脓毒症患者的数据，构建机器学习模型，并进行内部和外部验证。使用 Boruta 算法选择变量。然后，使用八种机器学习算法来构建重症监护病房（ICU）患者急性肾损伤（AKI）发生的预测模型。利用叠加系综方法建立了一个新的叠加系综模型。使用受试者工作特征曲线下面积（AUC）、精确度-召回率（PR）曲线、准确度、召回率和 F1 评分进行模型评估。采用 Shapley 加性解释（SHAP）方法对模型进行解释。

**Results:** The final study included 1295 patients with acute pancreatitis complicated by sepsis, among whom 893 cases (68.9%) developed acute kidney injury. We established eight base models, including Logit, SVM, CatBoost, RF, XGBoost, LightGBM, AdaBoost and MLP, as well as a stacked ensemble model called Multimodel. Among all models, Multimodel had an AUC value of 0.853 (95% CI: 0.792 to 0.896) in the internal validation dataset and 0.802 (95% CI: 0.732 to 0.861) in the external validation dataset. This model demonstrated the best predictive performance in terms of discrimination and clinical application.

**结果：** 最终研究纳入了 1295 例急性胰腺炎并发脓毒症患者，其中 893 例（68.9%）发生急性肾损伤。我们建立了 Logit、SVM、CatBoost、RF、XGBoost、LightGBM、AdaBoost 和 MLP 等 8 个基本模型，以及一个称为 Multimodel 的堆叠集成模型。在所有模型中，内部验证数据集中多模型的 AUC 值为 0.853（95% CI：0.792 至 0.896），外部验证数据集中为 0.802（95% CI：0.732 至 0.861）。该模型在区分和临床应用方面表现出最佳的预测性能。

**Conclusion:** The stack ensemble model developed by us achieved AUC values of 0.853 and 0.802 in internal and external validation cohorts respectively and also demonstrated excellent performance in other metrics. It serves as a reliable tool for predicting AKI in patients with acute pancreatitis complicated by sepsis.

**结论：** 我们开发的堆栈集成模型在内部和外部验证队列中分别实现了 0.853 和 0.802 的 AUC 值，并且在其他指标中也表现出出色的性能。它是预测急性胰腺炎并发脓毒症患者 AKI 的可靠工具。

---

**BMJ Open. 2022 Mar 3;12(3):e054092. doi: 10.1136/bmjopen-2021-054092.**

**Severe acute kidney injury predicting model based on transcontinental databases: a single-centre prospective study**

**基于洲际数据库的严重急性肾损伤预测模型的单中心前瞻性研究**

**Objectives:** There are many studies of acute kidney injury (AKI) diagnosis models lack of external validation and prospective validation. We constructed the models using three databases to predict severe AKI within 48 hours in intensive care unit (ICU) patients.

**目的：** 急性肾损伤（AKI）诊断模型的研究很多缺乏外部验证和前瞻性验证。我们使用三个数据库构建模型，以预测重症监护病房（ICU）患者 48 小时内的严重 AKI。

**Design:** A retrospective and prospective cohort study.

**设计：** 回顾性和前瞻性队列研究。

**Setting:** We studied critically ill patients in our database (SHZJU-ICU) and two other public databases, the Medical Information Mart for Intensive Care (MIMIC) and AmsterdamUMC databases, including basic demographics, vital signs and laboratory results. We predicted the diagnosis of severe AKI in patients in the next 48 hours using machine-learning algorithms with the three databases. Then, we carried out real-time severe AKI prediction in the prospective validation study at our centre for 1 year.

**设定：** 我们在我们的数据库（SHZJU-ICU）和另外两个公共数据库（重症监护医学信息市场（MIMIC）和阿姆斯特丹 UMC 数据库）中研究了危重患者，包括基本人口统计学，生命体征和实验室结果。我们使用三个数据库的机器学习算法预测了未来 48 小时内患者严重 AKI 的诊断。然后，我们在我们中心为期 1 年的前瞻性验证研究中进行了实时严重 AKI 预测。

**Participants:** All patients included in three databases with uniform exclusion criteria.

**参会人员：** 所有患者均纳入 3 个具有统一排除标准的数据库。

**Results:** We included 58 492 patients, and a total of 5257 (9.0%) patients met the definition of severe AKI. In the internal validation of the SHZJU-ICU and MIMIC databases, the best area under the receiver operating characteristic curve (AUROC) of the model was 0.86. The external validation results by AmsterdamUMC database were also satisfactory, with the best AUROC of 0.86. A total of 2532 patients were admitted to the centre for prospective validation; 358 positive results were predicted and 344 patients were diagnosed with severe AKI, with the best sensitivity of 0.72, the specificity of 0.80 and the AUROC of 0.84.

**结果：** 我们纳入了 58492 例患者，共有 5257 例（9.0%）患者符合重度 AKI 的定义。在 SHZJU-ICU 和 MIMIC 数据库的内部验证中，模型的最佳受试者工作特征曲线下面积（AUROC）为 0.86。AmsterdamUMC 数据库的外部验证结果也令人满意，最佳 AUROC 为 0.86。共有 2532 例患者入住该中心进行前瞻性验证；预测 358 例阳性结果，344 例患者诊断为重度 AKI，最佳灵敏度为 0.72，特异性为 0.80，AUROC 为 0.84。

**Conclusion:** The prediction model of severe AKI exhibits promises as a clinical application based on dynamic vital signs and laboratory results of multicentre databases with prospective and external validation.

**结论：** 基于动态生命体征和多中心数据库的实验室结果，通过前瞻性和外部验证，严重 AKI 的预测模型有望成为临床应用。

---

**Lancet Respir Med. 2018 Dec;6(12):905-914. doi: 10.1016/S2213-2600(18)30300-X. Epub 2018 Sep 28.**

**Machine learning for real-time prediction of complications in critical care: a retrospective study**

**机器学习用于重症监护并发症的实时预测：一项回顾性研究**

**Background:** The large amount of clinical signals in intensive care units can easily overwhelm health-care personnel and can lead to treatment delays, suboptimal care, or clinical errors. The aim of this study was to apply deep machine learning methods to predict severe complications during critical care in real time after cardiothoracic surgery.

**背景：** 重症监护室中的大量临床信号很容易使卫生保健人员不堪重负，并可能导致治疗延迟、次优护理或临床错误。本研究的目的是应用深度机器学习方法来预测心胸手术后真实的重症监护期间的严重并发症。

**Methods:** We used deep learning methods (recurrent neural networks) to predict several severe complications (mortality, renal failure with a need for renal replacement therapy, and postoperative bleeding leading to operative revision) in post cardiosurgical care in real time. Adult patients who underwent major open heart surgery from Jan 1, 2000, to Dec 31, 2016, in a German tertiary care centre for cardiovascular diseases formed the main derivation dataset. We measured the accuracy and timeliness of the deep learning model's forecasts and compared predictive quality to that of established standard-of-care clinical reference tools (clinical rule for postoperative bleeding, Simplified Acute Physiology Score II for mortality, and the Kidney Disease: Improving Global Outcomes staging criteria for acute renal failure) using positive predictive value (PPV), negative predictive value, sensitivity, specificity, area under the curve (AUC), and the F1 measure (which computes a harmonic mean of sensitivity and PPV). Results were externally retrospectively validated with 5898 cases from the published MIMIC-III dataset.

**方法：** 我们使用深度学习方法（递归神经网络）来真实的预测心脏手术后护理中的几种严重并发症（死亡率、需要肾脏替代治疗的肾衰竭和导致手术翻修的术后出血）。从 2000 年 1 月 1 日至 2016 年 12 月 31 日在德国三级心血管疾病护理中心接受大型心脏直视手术的成年患者形成了主要的衍生数据集。我们测量了深度学习模型预测的准确性和及时性，并将预测质量与已建立的标准护理临床参考工具进行了比较（术后出血的临床规则，死亡率的简化急性生理学评分 II，以及肾脏疾病：改善急性肾衰竭的全球结局分期标准）使用阳性预测值（PPV）、阴性预测值、敏感性、特异性，曲线下面积（AUC）和 F1 测量（计算灵敏度和 PPV 的调和平均值）。结果通过来自已发表 MIMIC-III 数据集的 5898 例病例进行了外部回顾性验证。

**Findings:** Of 47 559 intensive care admissions (corresponding to 42 007 patients), we included 11 492 (corresponding to 9269 patients). The deep learning models yielded accurate predictions with the following PPV and sensitivity scores: PPV 0·90 and sensitivity 0·85 for mortality, 0·87 and 0·94 for renal failure, and 0·84 and 0·74 for bleeding. The predictions significantly outperformed the standard clinical reference tools, improving the absolute complication prediction AUC by 0·29 (95% CI 0·23-0·35) for bleeding, by 0·24 (0·19-0·29) for mortality, and by 0·24 (0·13-0·35) for renal failure (p<0·0001 for all three analyses). The deep learning methods showed accurate predictions immediately after patient admission to the intensive care unit. We also observed an increase in performance in our validation cohort when the machine learning approach was tested against clinical reference tools, with absolute improvements in AUC of 0·09 (95% CI 0·03-0·15; p=0·0026) for bleeding, of 0·18 (0·07-0·29; p=0·0013) for mortality, and of 0·25 (0·18-0·32; p<0·0001) for renal failure.

**调查结果：** 在 47559 例重症监护入院（对应 42007 例患者）中，我们纳入了 11492 例（对应 9269 例患者）。深度学习模型通过以下 PPV 和敏感性评分进行了准确的预测：死亡率的 PPV 为 0.90，敏感性为 0.85，肾衰竭为 0.87 和 0.94，出血为 0.84 和 0.74。预测结果显著优于标准临床参考工具，出血绝对并发症预测 AUC 提高了 0.29（95% CI 0.23 - 0.35），死亡率提高了 0.24（0.19 - 0.29），肾衰竭提高了 0.24（0.13 - 0.35）（所有三项分析的 p<0.0001）。深度学习方法在患者进入重症监护室后立即显示出准确的预测。我们还观察到，当机器学习方法与临床参考工具进行测试时，我们的验证队列的性能有所提高，AUC 的绝对改善为 0.09（95% CI 0.03 - 0.15; p= 0.0026），死亡率为 0.18（0.07 - 0.29; p= 0.0013），肾衰竭为 0.25（0.18 - 0.32; p<0.0001）。

**Interpretation:** The observed improvements in prediction for all three investigated clinical outcomes have the potential to improve critical care. These findings are noteworthy in that they use routinely collected clinical data exclusively, without the need for any manual processing. The deep machine learning method showed AUC scores that significantly surpass those of clinical reference tools, especially soon after admission. Taken together, these properties are encouraging for prospective deployment in critical care settings to direct the staff's attention towards patients who are most at risk.

**释义：** 观察到的所有三个研究的临床结果的预测改善有可能改善重症监护。这些发现值得注意，因为它们仅使用常规收集的临床数据，而不需要任何手动处理。深度机器学习方法显示 AUC 评分显著超过临床参考工具，特别是在入院后不久。总的来说，这些属性是令人鼓舞的前瞻性部署在重症监护环境中，以引导工作人员的注意力对谁是最危险的病人。

---

**J Am Med Inform Assoc. 2025 Apr 1;32(4):736-747. doi: 10.1093/jamia/ocae318.**

**Fast and interpretable mortality risk scores for critical care patients**

**重症监护患者的快速和可解释的死亡风险评分**

**Objective:** Prediction of mortality in intensive care unit (ICU) patients typically relies on black box models (that are unacceptable for use in hospitals) or hand-tuned interpretable models (that might lead to the loss in performance). We aim to bridge the gap between these 2 categories by building on modern interpretable machine learning (ML) techniques to design interpretable mortality risk scores that are as accurate as black boxes.

**目的：** 重症监护病房（ICU）患者死亡率的预测通常依赖于黑箱模型（在医院中使用是不可接受的）或手动调整的可解释模型（可能导致性能损失）。我们的目标是通过建立现代可解释的机器学习（ML）技术来弥合这两个类别之间的差距，以设计可解释的死亡风险评分，这些评分与黑匣子一样准确。

**Material and methods:** We developed a new algorithm, GroupFasterRisk, which has several important benefits: it uses both hard and soft direct sparsity regularization, it incorporates group sparsity to allow more cohesive models, it allows for monotonicity constraint to include domain knowledge, and it produces many equally good models, which allows domain experts to choose among them. For evaluation, we leveraged the largest existing public ICU monitoring datasets (MIMIC III and eICU).

**材料和方法：** 我们开发了一种新的算法，GroupFasterRisk，它有几个重要的好处：它使用硬和软直接稀疏正则化，它结合了组稀疏性，以允许更有凝聚力的模型，它允许单调性约束，包括领域知识，它产生了许多同样好的模型，这使得领域专家可以在其中进行选择。为了进行评估，我们利用了现有最大的公共 ICU 监测数据集（MIMIC III 和 eICU）。

**Results:** Models produced by GroupFasterRisk outperformed OASIS and SAPS II scores and performed similarly to APACHE IV/IVa while using at most a third of the parameters. For patients with sepsis/septicemia, acute myocardial infarction, heart failure, and acute kidney failure, GroupFasterRisk models outperformed OASIS and SOFA. Finally, different mortality prediction ML approaches performed better based on variables selected by GroupFasterRisk as compared to OASIS variables.

**结果：** GroupFasterRisk 生成的模型优于 OASIS 和 SAPS II 评分，并且在使用至多三分之一的参数时与 APACHE IV/IVa 相似。对于脓毒症/败血症、急性心肌梗死、心力衰竭和急性肾衰竭患者，GroupFasterRisk 模型优于 OASIS 和 SOFA。最后，与 OASIS 变量相比，基于 GroupFasterRisk 选择的变量，不同的死亡率预测 ML 方法表现更好。

**Discussion:** GroupFasterRisk's models performed better than risk scores currently used in hospitals, and on par with black box ML models, while being orders of magnitude sparser. Because GroupFasterRisk produces a variety of risk scores, it allows design flexibility-the key enabler of practical model creation.

**讨论内容：** GroupFasterRisk 的模型比目前医院使用的风险评分更好，与黑盒 ML 模型相当，但数量级更稀疏。由于 GroupFasterRisk 可生成各种风险评分，因此可实现设计灵活性-这是创建实际模型的关键因素。

**Conclusion:** GroupFasterRisk is a fast, accessible, and flexible procedure that allows learning a diverse set of sparse risk scores for mortality prediction.

**结论：** GroupFasterRisk 是一个快速、方便和灵活的程序，允许学习用于死亡率预测的各种稀疏风险评分。

---

**Zhonghua Wei Zhong Bing Ji Jiu Yi Xue. 2022 Mar;34(3):260-264. doi: 10.3760/cma.j.cn121430-20211117-01741.**

**[Predictive value of machine learning for in-hospital mortality for trauma-induced acute respiratory distress syndrome patients: an analysis using the data from MIMIC III]**

**[机器学习对创伤引起的急性呼吸窘迫综合征患者住院死亡率的预测价值：使用 MIMIC III 数据的分析]**

**Objective:** To investigate the value of machine learning methods for predicting in-hospital mortality in trauma patients with acute respiratory distress syndrome (ARDS).

**目的：** 探讨机器学习方法对创伤急性呼吸窘迫综合征（ARDS）患者院内死亡率的预测价值。

**Methods:** A retrospective non-intervention case-control study was performed. Trauma patients with ARDS met the Berlin definition were extracted from the the Medical Information Mart for Intensive Care III (MIMIC III) database. The basic information [including gender, age, body mass index (BMI), pH, oxygenation index, laboratory indexes, length of stay in the intensive care unit (ICU), the proportion of mechanical ventilation (MV) or continuous renal replacement therapy (CRRT), acute physiology score III (APS III), sequential organ failure score (SOFA) and simplified acute physiology score II (SAPS II)], complications (including hypertension, diabetes, infection, acute hemorrhagic anemia, sepsis, shock, acidosis and pneumonia) and prognosis data of patients were collected. Multivariate Logistic regression analysis was used to screen meaningful variables (P < 0.05). Logistic regression model, XGBoost model and artificial neural network model were constructed, and the receiver operator characteristic curve (ROC) was performed to evaluate the predictive value of the three models for in-hospital mortality in trauma patients with ARDS.

**研究方法：** 进行了一项回顾性非干预性病例对照研究。符合柏林定义的 ARDS 创伤患者从重症监护医学信息市场 III（MIMIC III）数据库中提取。基本信息[包括性别、年龄、体重指数（BMI）、pH、氧合指数、实验室指标、重症监护室（ICU）住院时间、机械通气（MV）或连续性肾脏替代治疗（CRRT）的比例、急性生理学评分 III（APS III）、序贯器官衰竭评分（SOFA）和简化急性生理学评分 II（SAPS II）]，收集患者的并发症（包括高血压、糖尿病、感染、急性出血性贫血、脓毒症、休克、酸中毒和肺炎）和预后数据。多因素 Logistic 回归分析筛选有意义的变量（P < 0.05）。分别构建 Logistic 回归模型、XGBoost 模型和人工神经网络模型，应用受试者工作特征曲线（ROC）评价 3 种模型对创伤合并 ARDS 患者住院死亡率的预测价值。

**Results:** A total of 760 trauma patients with ARDS were enrolled, including 346 mild cases, 301 moderate cases and 113 severe cases; 618 cases survived and 142 cases died in hospital; 736 cases received MV and 65 cases received CRRT. Multivariate Logistic regression analysis screened out significant variables, including age [odds ratio (OR) = 1.035, 95% confidence interval (95%CI) was 1.020-1.050, P < 0.001], BMI (OR = 0.949, 95%CI was 0.917-0.983, P = 0.003), blood urea nitrogen (BUN; OR = 1.019, 95%CI was 1.004-1.033, P = 0.010), lactic acid (Lac; OR = 1.213, 95%CI was 1.124-1.309, P < 0.001), red cell volume distribution width (RDW; OR = 1.249, 95%CI was 1.102-1.416, P < 0.001), hematocrit (HCT, OR = 1.057, 95%CI was 1.019-1.097, P = 0.003), hypertension (OR = 0.614, 95%CI was 0.389-0.968, P = 0.036), infection (OR = 0.463, 95%CI was 0.289-0.741, P = 0.001), acute renal failure (OR = 2.021, 95%CI was 1.267-3.224, P = 0.003) and sepsis (OR = 2.105, 95%CI was 1.265-3.502, P = 0.004). All the above variables were used to construct the model. Logistic regression model, XGBoost model and artificial neural network model predicted in-hospital mortality with area under the curve (AUC) of 0.737 (95%CI was 0.659-0.815), 0.745 (95%CI was 0.672-0.819) and 0.757 (95%CI was 0.680-0.884), respectively. There was no significant difference between any two models (all P > 0.05).

**结果：** 共入组 760 例创伤并发 ARDS 患者，其中轻度 346 例，中度 301 例，重度 113 例；存活 618 例，院内死亡 142 例；MV 736 例，CRRT 65 例。多因素 Logistic 回归分析筛选出年龄[OR = 1.035，95%CI 为 1.020-1.050，P < 0.001]、体重指数（BMI）、（OR = 0.949，95%CI为0.917-0.983，P = 0.003），血尿素氮（BUN; OR = 1.019，95%CI为1.004-1.033，P = 0.010），乳酸（Lac; OR = 1.213，95%CI为1.124-1.309，P < 0.001），红细胞体积分布宽度（RDW; OR = 1.249，95%CI为1.102-1.416，P < 0.001），红细胞压积（HCT，OR = 1.057，95%CI为1.019-1.097，P = 0.003），高血压（OR = 0.614，95%CI为0.389-0.968，P = 0.036），感染（OR = 0.463，95%CI为0.289-0.741，P = 0.001），急性肾功能衰竭（OR = 2.021，95%CI为1.267-3.224，P = 0.003）和脓毒症（OR = 2.105，95%CI为1.265-3.502，P = 0.004）。所有上述变量均用于构建模型。Logistic 回归模型、XGBoost 模型和人工神经网络模型预测住院死亡率的曲线下面积（AUC）分别为 0.737（95%CI 为 0.659-0.815）、0.745（95%CI 为 0.672-0.819）和 0.757（95%CI 为 0.680-0.884）。两种模型间比较差异无统计学意义（均 P > 0.05）。

**Conclusions:** Logistic regression model, XGBoost model and artificial neural network model including age, BMI, BUN, Lac, RDW, HCT, hypertension, infection, acute renal failure and sepsis have good predictive value for in-hospital mortality of trauma patients with ARDS.

**结论：** Logistic 回归模型、XGBoost 模型和人工神经网络模型包括年龄、BMI、BUN、Lac、RDW、HCT、高血压、感染、急性肾功能衰竭和脓毒症对创伤合并 ARDS 患者的院内死亡率有较好的预测价值。

---

**BMJ Open. 2025 Jan 28;15(1):e088404. doi: 10.1136/bmjopen-2024-088404.**

**Development and validation of a novel risk-predicted model for early sepsis-associated acute kidney injury in critically ill patients: a retrospective cohort study**

**危重患者早期脓毒症相关急性肾损伤的新型风险预测模型的开发和验证：一项回顾性队列研究**

**Objectives:** This study aimed to develop a prediction model for the detection of early sepsis-associated acute kidney injury (SA-AKI), which is defined as AKI diagnosed within 48 hours of a sepsis diagnosis.

**目的：** 本研究旨在开发一种用于检测早期脓毒症相关急性肾损伤（SA-AKI）的预测模型，SA-AKI 定义为脓毒症诊断后 48 小时内诊断的 AKI。

**Design:** A retrospective study design was employed. It is not linked to a clinical trial. Data for patients with sepsis included in the development cohort were extracted from the Medical Information Mart for Intensive Care IV (MIMIC-IV) database. The least absolute shrinkage and selection operator regression method was used to screen the risk factors, and the final screened risk factors were constructed into four machine learning models to determine an optimal model. External validation was performed using another single-centre intensive care unit (ICU) database.

**设计：** 采用回顾性研究设计。它与临床试验无关。从重症监护医学信息市场 IV（MIMIC-IV）数据库中提取开发队列中纳入的脓毒症患者数据。采用最小绝对收缩和选择算子回归方法筛选危险因素，将最终筛选出的危险因素构建成 4 个机器学习模型，确定最优模型。使用另一个单中心重症监护室（ICU）数据库进行外部验证。

**Setting:** Data for the development cohort were obtained from the MIMIC-IV 2.0 database, which is a large publicly available database that contains information on patients admitted to the ICUs of Beth Israel Deaconess Medical Center in Boston, Massachusetts, USA, from 2008 to 2019. The external validation cohort was generated from a single-centre ICU database from China.

**设定：** 开发队列的数据来自 MIMIC-IV 2.0 数据库，该数据库是一个大型公开数据库，包含 2008 年至 2019 年美国马萨诸塞州波士顿贝斯以色列女执事医疗中心 ICU 收治的患者信息。外部验证队列来自中国单中心 ICU 数据库。

**Participants:** A total of 7179 critically ill patients with sepsis were included in the development cohort and 269 patients with sepsis were included in the external validation cohort.

**参会人员：** 共有 7179 例脓毒症危重患者被纳入开发队列，269 例脓毒症患者被纳入外部验证队列。

**Results:** A total of 12 risk factors (age, weight, atrial fibrillation, chronic coronary syndrome, central venous pressure, urine output, temperature, lactate, pH, difference in alveolar-arterial oxygen pressure, prothrombin time and mechanical ventilation) were included in the final prediction model. The gradient boosting machine model showed the best performance, and the areas under the receiver operating characteristic curve of the model in the development cohort, internal validation cohort and external validation cohort were 0.794, 0.725 and 0.707, respectively. Additionally, to aid interpretation and clinical application, SHapley Additive exPlanations techniques and a web version calculation were applied.

**结果：** 最终的预测模型共包括 12 个危险因素（年龄、体重、房颤、慢性冠状动脉综合征、中心静脉压、尿量、体温、乳酸盐、pH 值、肺泡 - 动脉氧分压差、凝血酶原时间和机械通气）。梯度增强机器模型表现出最好的性能，模型在开发队列、内部验证队列和外部验证队列中的受试者工作特征曲线下的面积分别为 0.794、0.725 和 0.707。此外，为了帮助解释和临床应用，应用了 SHapley 加法解释技术和网络版本计算。

**Conclusions:** This web-based clinical prediction model represents a reliable tool for predicting early SA-AKI in critically ill patients with sepsis. The model was externally validated using another ICU cohort and exhibited good predictive ability. Additional validation is needed to support the utility and implementation of this model.

**结论：** 这种基于网络的临床预测模型是预测脓毒症重症患者早期 SA-AKI 的可靠工具。该模型使用另一个 ICU 队列进行了外部验证，并表现出良好的预测能力。需要额外的验证来支持该模型的实用性和实施。

---

**Crit Care. 2019 Jul 8;23(1):247. doi: 10.1186/s13054-019-2486-6.**

**Multimorbidity states associated with higher mortality rates in organ dysfunction and sepsis: a data-driven analysis in critical care**

**与器官功能障碍和脓毒症死亡率较高相关的多发病状态：重症监护中的数据驱动分析**

**Background:** Sepsis remains a complex medical problem and a major challenge in healthcare. Diagnostics and outcome predictions are focused on physiological parameters with less consideration given to patients' medical background. Given the aging population, not only are diseases becoming increasingly prevalent but occur more frequently in combinations ("multimorbidity"). We hypothesized the existence of patient subgroups in critical care with distinct multimorbidity states. We further hypothesize that certain multimorbidity states associate with higher rates of organ failure, sepsis, and mortality co-occurring with these clinical problems.

**背景：** 脓毒症仍然是一个复杂的医疗问题和医疗保健的主要挑战。诊断和结果预测集中在生理参数上，较少考虑患者的医学背景。鉴于人口老龄化，疾病不仅变得越来越普遍，而且更频繁地以组合形式发生（“多发病”）。我们假设存在重症监护患者亚组，具有不同的多发病状态。我们进一步假设，某些多发病状态与器官衰竭、脓毒症和死亡率较高相关，这些疾病与这些临床问题同时发生。

**Methods:** We analyzed 36,390 patients from the open source Medical Information Mart for Intensive Care III (MIMIC III) dataset. Morbidities were defined based on Elixhauser categories, a well-established scheme distinguishing 30 classes of chronic diseases. We used latent class analysis to identify distinct patient subgroups based on demographics, admission type, and morbidity compositions and compared the prevalence of organ dysfunction, sepsis, and inpatient mortality for each subgroup.

**研究方法：** 我们分析了来自开源重症监护医学信息市场 III（MIMIC III）数据集的 36,390 名患者。根据 Elixhauser 分类定义发病率，这是一个区分 30 类慢性疾病的完善方案。我们使用潜在类别分析来识别基于人口统计学、入院类型和发病率组成的不同患者亚组，并比较每个亚组的器官功能障碍、脓毒症和住院死亡率的患病率。

**Results:** We identified six clinically distinct multimorbidity subgroups labeled based on their dominant Elixhauser disease classes. The "cardiopulmonary" and "cardiac" subgroups consisted of older patients with a high prevalence of cardiopulmonary conditions and constituted 6.1% and 26.4% of study cohort respectively. The "young" subgroup included 23.5% of the cohort composed of young and healthy patients. The "hepatic/addiction" subgroup, constituting 9.8% of the cohort, consisted of middle-aged patients (mean age of 52.25, 95% CI 51.85-52.65) with the high rates of depression (20.1%), alcohol abuse (47.75%), drug abuse (18.2%), and liver failure (67%). The "complicated diabetics" and "uncomplicated diabetics" subgroups constituted 9.4% and 24.8% of the study cohort respectively. The complicated diabetics subgroup demonstrated higher rates of end-organ complications (88.3% prevalence of renal failure). Rates of organ dysfunction and sepsis ranged 19.6-69% and 12.5-46.7% respectively in the six subgroups. Mortality co-occurring with organ dysfunction and sepsis ranges was 8.4-23.8% and 11.7-27.4% respectively. These adverse outcomes were most prevalent in the hepatic/addiction subgroup.

**结果：** 我们确定了六个临床上不同的多发病亚组标记的基础上，他们占主导地位的 Elixhauser 疾病类别。“心肺”和“心脏”亚组由心肺疾病患病率较高的老年患者组成，分别占研究队列的 6.1%和 26.4%。“年轻”亚组包括 23.5%的年轻健康患者。“肝脏/成瘾”亚组占队列的 9.8%，由中年患者（平均年龄 52.25 岁，95% CI 51.85-52.65）组成，抑郁（20.1%）、酒精滥用（47.75%）、药物滥用（18.2%）和肝功能衰竭（67%）的发生率较高。“并发糖尿病”和“无并发糖尿病”亚组分别占研究队列的 9.4%和 24.8%。糖尿病并发症亚组的终末器官并发症发生率较高（肾功能衰竭患病率为 88.3%）。6 个亚组的器官功能障碍和脓毒症发生率分别为 19.6-69%和 12.5-46.7%。与器官功能障碍和脓毒症同时发生的死亡率范围分别为 8.4-23.8%和 11.7-27.4%。这些不良后果在肝脏/成瘾亚组中最为普遍。

**Conclusion:** We identify distinct multimorbidity states that associate with relatively higher prevalence of organ dysfunction, sepsis, and co-occurring mortality. The findings promote the incorporation of multimorbidity in healthcare models and the shift away from the current single-disease paradigm in clinical practice, training, and trial design.

**结论：** 我们确定了不同的多发病状态，这些状态与器官功能障碍、脓毒症和并发死亡率的相对较高的患病率相关。这些发现促进了医疗保健模式中多疾病的整合，并在临床实践，培训和试验设计中摆脱了当前的单一疾病范式。
















































