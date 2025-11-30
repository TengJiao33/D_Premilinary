
* **`Manhattan_Rodents_2023_2025.csv` (鼠患投诉数据)**
    * **描述**: 来自 NYC 311 热线的鼠患（Rodent）投诉记录。
    * **数据类型**: 地理点数据。
    * **时间跨度**: 2023年 - 2025年。
    * **关键字段**:
        * `created_date`: 投诉时间
        * `location_type`: 地点类型（如：3+ Family Apt. Building, Sidewalk 等）
        * `latitude` / `longitude`: 具体地理坐标
        * `community_board`: 所属社区委员会

* **`Manhattan_Garbage_Ton_201701_202510.csv` (垃圾清运量数据)**
    * **描述**: 曼哈顿各社区每月收集的垃圾吨数统计。
    * **数据类型**: 时间序列数据 (Time Series)。
    * **时间跨度**: 2017年1月 - 2025年10月。
    * **关键字段**:
        * `refusetonscollected`: 普通生活垃圾吨数
        * `papertonscollected`: 纸张回收吨数
        * `resorganicstons`: 有机/厨余垃圾吨数
        * `communitydistrict`: 社区代码

### B. 背景特征数据 (社会经济/人口普查)
这部分数据来自 2019-2023 ACS (美国社区调查)，用于描绘每个社区的人口画像。

* **`Dem_1923_CDTA.xlsx` (Demographics - 人口统计)**
    * 包含：年龄分布、性别比例、种族构成。
* **`Econ_1923_CDTA.xlsx` (Economics - 经济状况)**
    * 包含：就业率、收入水平、贫困状况、健康保险覆盖率。
* **`Hous_1923_CDTA.xlsx` (Housing - 住房状况)**
    * 包含：房屋空置率、租金成本、房屋结构类型、居住年限。
* **辅助文件**:
    * `* - Data Dictionary.csv`: 数据字典，解释了人口普查数据中列名的具体含义（如 `PopU5` 代表 5岁以下人口）。
    * `* - About.csv`: 数据来源说明。

### C. 汇总分析数据
这是经过清洗和聚合处理后的核心分析表。

* **`Manhattan_Data_Current_2023_2025.csv`**
    * **描述**: 将上述卫生数据与社会经济数据按“社区”进行合并后的宽表。
    * **核心字段**:
        * `CD_ID`: 社区 ID
        * `Rat_Complaints`: 鼠患投诉总数
        * `Monthly_Trash_Tons`: 月均垃圾吨数
        * `Population`: 社区人口
        * `Median_Income`: 收入中位数


**注意**: 人口普查数据（ACS）采用的是 CDTA（Community District Tabulation Areas）地理标准，已在汇总表中尽可能与传统的 Community District (CD) 进行了匹配。