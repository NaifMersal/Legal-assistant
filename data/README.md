## 📄 README: Saudi Laws Scraped Data Structure (with Examples)

This document details the hierarchical structure of the scraped Saudi laws data, providing concrete examples at each level to illustrate the format.

-----

### 1\. Overall Data Hierarchy

The final output is a single **JSON object** representing the entire hierarchy of laws, structured as a nested dictionary:

| Level | Key Type | Example Key |
| :---: | :--- | :--- |
| **1** | **Main Category Name** | `"أنظمة أساسية"` (Fundamental Systems) |
| **2** | **Sub-Category Name** | `"الأنظمة الأساسية"` (The Fundamental Systems) |
| **3** | **Law Title** | `"النظام الأساسي للحكم"` (The Basic Law of Governance) |
| **4** | **Law Data Object** | (Contains `brief`, `metadata`, `parts`) |

#### **Example 1: Root Level (Main Categories)**

The top level organizes content by broad legislative type:

```json
{
  "أنظمة أساسية": {
    "الأنظمة الأساسية": {
      "النظام الأساسي للحكم": {
        /* ... Law Data ... */
      }
    }
  },
  "أنظمة عادية": {
    "أنظمة إدارية": {
      "نظام مكافحة الرشوة": {
        /* ... Law Data ... */
      }
    },
    "أنظمة تجارية": {
      /* ... */
    }
  }
}
```

-----

### 2\. Law Data Object Structure

The content for each law is stored under its title and contains a summary (`brief`), descriptive details (`metadata`), and the text content (`parts`).

#### **Example 2: Second Level (Law Metadata)**

The `brief` and `metadata` sections provide quick facts about the law:

```json
{
  "أنظمة أساسية": {
    "الأنظمة الأساسية": {
      "النظام الأساسي للحكم": {
        "brief": "يتضمن العناوين التالية: المبادئ العامة، نظام الحكم، مقومات المجتمع السعودي، المبادئ الاقتصادية، الحقوق والواجبات، سلطات الدولة، الشئون المالية، أحكام عامة....",
        "metadata": {
          "الاسم": "النظام الأساسي للحكم",
          "تاريخ الإصدار": "1412/08/27 هـ  الموافق : 01/03/1992 مـ",
          "تاريخ النشر": "1412/09/02  هـ الموافق : 06/03/1992 مـ",
          "الحالة": "ساري"
        },
        "parts": {
          /* ... Parts/Articles Structure ... */
        }
      }
    }
  }
}
```

-----

### 3\. The `parts` Dictionary Structure

The `parts` dictionary holds the actual text of the law, grouped by chapter or section. Each value is a **list of Article Objects**.

#### **Example 3: Multi-Part Law (Standard Structure)**

Laws divided into multiple chapters (e.g., "الباب الأول," "الباب الثاني") use the title of the section as the key:

| Key | Description |
| :--- | :--- |
| **Part Title** | The header text defining a major section/chapter. |
| **Value** | A list of articles belonging to that part. |

```json
/* Key: Law_Title -> parts */
{
  "الباب الأول :  المبادئ العامة": [
    {
      "id": 0,
      "Article_Title": "المادة الأولى",
      "status": "Active",
      "Article_Text": "المملكة العربية السعودية، دولة عربية إسلامية، ذات..."
    },
    {
      "id": 1,
      "Article_Title": "المادة الثانية",
      "status": "Active",
      "Article_Text": "دين الدولة الإسلام، ودستورها كتاب الله وسنة رسوله..."
    }
  ],
  "الباب الثاني : نظام الحكم": [
    {
      "id": 4,
      "Article_Title": "المادة الخامسة",
      "status": "Modified",
      "Article_Text": "أ - نظام الحكم في المملكة العربية السعودية، ملكي...."
    }
  ]
}
```

#### **Special Key: 'main' (Single-Part Laws)**

For laws that consist of a single continuous body of articles (i.e., they lack internal chapter titles), all articles are grouped under the default key: **`'main'`**.

```json
/* Key: Law_Title -> parts */
{
  "main": [
    {
      "id": 150,
      "Article_Title": "المادة رقم 1",
      "status": "Active",
      "Article_Text": "تطبق أحكام هذا النظام على جميع المخالفات التي ترتكب..."
    },
    {
      "id": 151,
      "Article_Title": "المادة رقم 2",
      "status": "Active",
      "Article_Text": "تختص هيئة التحقيق والادعاء العام بالتحقيق في المخالفات..."
    }
    // ... all other articles ...
  ]
}
```

-----

### 4\. Fallback Structure (للائحة - Regulation)

The code includes a **FALLBACK LOGIC** to handle unstructured pages, common for **اللائحة (Regulation)** documents, where individual article divs (`article_item`) are not present.

| Field | Description |
| :--- | :--- |
| **Purpose** | To capture the entire text of an unstructured regulation as a single entry. |
| **Structure** | The `parts` dictionary will contain the key **`'main'`**, holding a list with only **one** Article Object. |

#### **Example 4: Fallback Law (Unstructured Regulation)**

This structure indicates the entire text content was extracted as a single block:

```json
/* Key: Law_Title -> parts */
{
  "main": [
    {
      "id": 500,
      "Article_Title": "نص اللائحة", 
      "status": "Active", 
      "Article_Text": "الباب الأول : أحكام عامة \n المادة 1 : يُطلق على هذه اللائحة اسم (لائحة....) وتطبق على.... \n الباب الثاني : الإجراءات .... إلخ (الـنـص الـكـامـل)" 
    }
  ]
}
```
### 📊 5. Exploratory Data Analysis (EDA)

A statistical summary of the dataset are available in the notebook:

**`Simple_EDA.ipynb`**

This includes:

* Distribution of articles per law and per part
* Token length analysis (articles, parts, laws, briefs)
* Article status statistics (Active, Modified, Canceled)