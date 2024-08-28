# ExportedCCSDocument

**Title:** ExportedCCSDocument

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |

**Description:** Document model for Docling.

<details>
<summary>
<strong> <a name="name"></a>1. [Required] Property ExportedCCSDocument > _name</strong>  

</summary>
<blockquote>

**Title:**  Name

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="type"></a>2. [Optional] Property ExportedCCSDocument > type</strong>  

</summary>
<blockquote>

**Title:** Type

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `"pdf-document"`                                                          |

<blockquote>

| Any of(Option)           |
| ------------------------ |
| [item 0](#type_anyOf_i0) |
| [item 1](#type_anyOf_i1) |

<blockquote>

### <a name="type_anyOf_i0"></a>2.1. Property `ExportedCCSDocument > type > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

### <a name="type_anyOf_i1"></a>2.2. Property `ExportedCCSDocument > type > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description"></a>3. [Required] Property ExportedCCSDocument > description</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | Yes                                                                       |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/CCSDocumentDescription                                            |

**Description:** Description in document.

<details>
<summary>
<strong> <a name="description_title"></a>3.1. [Optional] Property ExportedCCSDocument > description > title</strong>  

</summary>
<blockquote>

**Title:** Title

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                        |
| ------------------------------------- |
| [item 0](#description_title_anyOf_i0) |
| [item 1](#description_title_anyOf_i1) |

<blockquote>

#### <a name="description_title_anyOf_i0"></a>3.1.1. Property `ExportedCCSDocument > description > title > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

#### <a name="description_title_anyOf_i1"></a>3.1.2. Property `ExportedCCSDocument > description > title > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_abstract"></a>3.2. [Optional] Property ExportedCCSDocument > description > abstract</strong>  

</summary>
<blockquote>

**Title:** Abstract

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#description_abstract_anyOf_i0) |
| [item 1](#description_abstract_anyOf_i1) |

<blockquote>

#### <a name="description_abstract_anyOf_i0"></a>3.2.1. Property `ExportedCCSDocument > description > abstract > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                      | Description |
| ---------------------------------------------------- | ----------- |
| [item 0 items](#description_abstract_anyOf_i0_items) | -           |

##### <a name="autogenerated_heading_2"></a>3.2.1.1. ExportedCCSDocument > description > abstract > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

#### <a name="description_abstract_anyOf_i1"></a>3.2.2. Property `ExportedCCSDocument > description > abstract > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_authors"></a>3.3. [Optional] Property ExportedCCSDocument > description > authors</strong>  

</summary>
<blockquote>

**Title:** Authors

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                          |
| --------------------------------------- |
| [item 0](#description_authors_anyOf_i0) |
| [item 1](#description_authors_anyOf_i1) |

<blockquote>

#### <a name="description_authors_anyOf_i0"></a>3.3.1. Property `ExportedCCSDocument > description > authors > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be               | Description |
| --------------------------------------------- | ----------- |
| [Author](#description_authors_anyOf_i0_items) | Author.     |

##### <a name="autogenerated_heading_3"></a>3.3.1.1. ExportedCCSDocument > description > authors > anyOf > item 0 > Author

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/Author                                          |

**Description:** Author.

<details>
<summary>
<strong> <a name="description_authors_anyOf_i0_items_name"></a>3.3.1.1.1. [Required] Property ExportedCCSDocument > description > authors > anyOf > item 0 > Author > name</strong>  

</summary>
<blockquote>

**Title:** Name

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_authors_anyOf_i0_items_id"></a>3.3.1.1.2. [Optional] Property ExportedCCSDocument > description > authors > anyOf > item 0 > Author > id</strong>  

</summary>
<blockquote>

**Title:** Id

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                            |
| --------------------------------------------------------- |
| [item 0](#description_authors_anyOf_i0_items_id_anyOf_i0) |
| [item 1](#description_authors_anyOf_i0_items_id_anyOf_i1) |

<blockquote>

###### <a name="description_authors_anyOf_i0_items_id_anyOf_i0"></a>3.3.1.1.2.1. Property `ExportedCCSDocument > description > authors > anyOf > item 0 > Author > id > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_authors_anyOf_i0_items_id_anyOf_i1"></a>3.3.1.1.2.2. Property `ExportedCCSDocument > description > authors > anyOf > item 0 > Author > id > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_authors_anyOf_i0_items_source"></a>3.3.1.1.3. [Optional] Property ExportedCCSDocument > description > authors > anyOf > item 0 > Author > source</strong>  

</summary>
<blockquote>

**Title:** Source

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                |
| ------------------------------------------------------------- |
| [item 0](#description_authors_anyOf_i0_items_source_anyOf_i0) |
| [item 1](#description_authors_anyOf_i0_items_source_anyOf_i1) |

<blockquote>

###### <a name="description_authors_anyOf_i0_items_source_anyOf_i0"></a>3.3.1.1.3.1. Property `ExportedCCSDocument > description > authors > anyOf > item 0 > Author > source > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_authors_anyOf_i0_items_source_anyOf_i1"></a>3.3.1.1.3.2. Property `ExportedCCSDocument > description > authors > anyOf > item 0 > Author > source > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_authors_anyOf_i0_items_affiliations"></a>3.3.1.1.4. [Optional] Property ExportedCCSDocument > description > authors > anyOf > item 0 > Author > affiliations</strong>  

</summary>
<blockquote>

**Title:** Affiliations

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                      |
| ------------------------------------------------------------------- |
| [item 0](#description_authors_anyOf_i0_items_affiliations_anyOf_i0) |
| [item 1](#description_authors_anyOf_i0_items_affiliations_anyOf_i1) |

<blockquote>

###### <a name="description_authors_anyOf_i0_items_affiliations_anyOf_i0"></a>3.3.1.1.4.1. Property `ExportedCCSDocument > description > authors > anyOf > item 0 > Author > affiliations > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                | Description  |
| ------------------------------------------------------------------------------ | ------------ |
| [Affiliation](#description_authors_anyOf_i0_items_affiliations_anyOf_i0_items) | Affiliation. |

###### <a name="autogenerated_heading_4"></a>3.3.1.1.4.1.1. ExportedCCSDocument > description > authors > anyOf > item 0 > Author > affiliations > anyOf > item 0 > Affiliation

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/Affiliation                                     |

**Description:** Affiliation.

<details>
<summary>
<strong> <a name="description_authors_anyOf_i0_items_affiliations_anyOf_i0_items_name"></a>3.3.1.1.4.1.1.1. [Required] Property ExportedCCSDocument > description > authors > anyOf > item 0 > Author > affiliations > anyOf > item 0 > Affiliation > name</strong>  

</summary>
<blockquote>

**Title:** Name

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_authors_anyOf_i0_items_affiliations_anyOf_i0_items_id"></a>3.3.1.1.4.1.1.2. [Optional] Property ExportedCCSDocument > description > authors > anyOf > item 0 > Author > affiliations > anyOf > item 0 > Affiliation > id</strong>  

</summary>
<blockquote>

**Title:** Id

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                        |
| ------------------------------------------------------------------------------------- |
| [item 0](#description_authors_anyOf_i0_items_affiliations_anyOf_i0_items_id_anyOf_i0) |
| [item 1](#description_authors_anyOf_i0_items_affiliations_anyOf_i0_items_id_anyOf_i1) |

<blockquote>

###### <a name="description_authors_anyOf_i0_items_affiliations_anyOf_i0_items_id_anyOf_i0"></a>3.3.1.1.4.1.1.2.1. Property `ExportedCCSDocument > description > authors > anyOf > item 0 > Author > affiliations > anyOf > item 0 > Affiliation > id > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_authors_anyOf_i0_items_affiliations_anyOf_i0_items_id_anyOf_i1"></a>3.3.1.1.4.1.1.2.2. Property `ExportedCCSDocument > description > authors > anyOf > item 0 > Author > affiliations > anyOf > item 0 > Affiliation > id > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_authors_anyOf_i0_items_affiliations_anyOf_i0_items_source"></a>3.3.1.1.4.1.1.3. [Optional] Property ExportedCCSDocument > description > authors > anyOf > item 0 > Author > affiliations > anyOf > item 0 > Affiliation > source</strong>  

</summary>
<blockquote>

**Title:** Source

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                            |
| ----------------------------------------------------------------------------------------- |
| [item 0](#description_authors_anyOf_i0_items_affiliations_anyOf_i0_items_source_anyOf_i0) |
| [item 1](#description_authors_anyOf_i0_items_affiliations_anyOf_i0_items_source_anyOf_i1) |

<blockquote>

###### <a name="description_authors_anyOf_i0_items_affiliations_anyOf_i0_items_source_anyOf_i0"></a>3.3.1.1.4.1.1.3.1. Property `ExportedCCSDocument > description > authors > anyOf > item 0 > Author > affiliations > anyOf > item 0 > Affiliation > source > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_authors_anyOf_i0_items_affiliations_anyOf_i0_items_source_anyOf_i1"></a>3.3.1.1.4.1.1.3.2. Property `ExportedCCSDocument > description > authors > anyOf > item 0 > Author > affiliations > anyOf > item 0 > Affiliation > source > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="description_authors_anyOf_i0_items_affiliations_anyOf_i1"></a>3.3.1.1.4.2. Property `ExportedCCSDocument > description > authors > anyOf > item 0 > Author > affiliations > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

#### <a name="description_authors_anyOf_i1"></a>3.3.2. Property `ExportedCCSDocument > description > authors > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_affiliations"></a>3.4. [Optional] Property ExportedCCSDocument > description > affiliations</strong>  

</summary>
<blockquote>

**Title:** Affiliations

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                               |
| -------------------------------------------- |
| [item 0](#description_affiliations_anyOf_i0) |
| [item 1](#description_affiliations_anyOf_i1) |

<blockquote>

#### <a name="description_affiliations_anyOf_i0"></a>3.4.1. Property `ExportedCCSDocument > description > affiliations > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                         | Description  |
| ------------------------------------------------------- | ------------ |
| [Affiliation](#description_affiliations_anyOf_i0_items) | Affiliation. |

##### <a name="autogenerated_heading_5"></a>3.4.1.1. ExportedCCSDocument > description > affiliations > anyOf > item 0 > Affiliation

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/Affiliation                                     |

**Description:** Affiliation.

<details>
<summary>
<strong> <a name="description_affiliations_anyOf_i0_items_name"></a>3.4.1.1.1. [Required] Property ExportedCCSDocument > description > affiliations > anyOf > item 0 > Affiliation > name</strong>  

</summary>
<blockquote>

**Title:** Name

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_affiliations_anyOf_i0_items_id"></a>3.4.1.1.2. [Optional] Property ExportedCCSDocument > description > affiliations > anyOf > item 0 > Affiliation > id</strong>  

</summary>
<blockquote>

**Title:** Id

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                 |
| -------------------------------------------------------------- |
| [item 0](#description_affiliations_anyOf_i0_items_id_anyOf_i0) |
| [item 1](#description_affiliations_anyOf_i0_items_id_anyOf_i1) |

<blockquote>

###### <a name="description_affiliations_anyOf_i0_items_id_anyOf_i0"></a>3.4.1.1.2.1. Property `ExportedCCSDocument > description > affiliations > anyOf > item 0 > Affiliation > id > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_affiliations_anyOf_i0_items_id_anyOf_i1"></a>3.4.1.1.2.2. Property `ExportedCCSDocument > description > affiliations > anyOf > item 0 > Affiliation > id > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_affiliations_anyOf_i0_items_source"></a>3.4.1.1.3. [Optional] Property ExportedCCSDocument > description > affiliations > anyOf > item 0 > Affiliation > source</strong>  

</summary>
<blockquote>

**Title:** Source

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                     |
| ------------------------------------------------------------------ |
| [item 0](#description_affiliations_anyOf_i0_items_source_anyOf_i0) |
| [item 1](#description_affiliations_anyOf_i0_items_source_anyOf_i1) |

<blockquote>

###### <a name="description_affiliations_anyOf_i0_items_source_anyOf_i0"></a>3.4.1.1.3.1. Property `ExportedCCSDocument > description > affiliations > anyOf > item 0 > Affiliation > source > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_affiliations_anyOf_i0_items_source_anyOf_i1"></a>3.4.1.1.3.2. Property `ExportedCCSDocument > description > affiliations > anyOf > item 0 > Affiliation > source > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

#### <a name="description_affiliations_anyOf_i1"></a>3.4.2. Property `ExportedCCSDocument > description > affiliations > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_subjects"></a>3.5. [Optional] Property ExportedCCSDocument > description > subjects</strong>  

</summary>
<blockquote>

**Title:** Subjects

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#description_subjects_anyOf_i0) |
| [item 1](#description_subjects_anyOf_i1) |

<blockquote>

#### <a name="description_subjects_anyOf_i0"></a>3.5.1. Property `ExportedCCSDocument > description > subjects > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                      | Description |
| ---------------------------------------------------- | ----------- |
| [item 0 items](#description_subjects_anyOf_i0_items) | -           |

##### <a name="autogenerated_heading_6"></a>3.5.1.1. ExportedCCSDocument > description > subjects > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

#### <a name="description_subjects_anyOf_i1"></a>3.5.2. Property `ExportedCCSDocument > description > subjects > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_keywords"></a>3.6. [Optional] Property ExportedCCSDocument > description > keywords</strong>  

</summary>
<blockquote>

**Title:** Keywords

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#description_keywords_anyOf_i0) |
| [item 1](#description_keywords_anyOf_i1) |

<blockquote>

#### <a name="description_keywords_anyOf_i0"></a>3.6.1. Property `ExportedCCSDocument > description > keywords > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                      | Description |
| ---------------------------------------------------- | ----------- |
| [item 0 items](#description_keywords_anyOf_i0_items) | -           |

##### <a name="autogenerated_heading_7"></a>3.6.1.1. ExportedCCSDocument > description > keywords > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

#### <a name="description_keywords_anyOf_i1"></a>3.6.2. Property `ExportedCCSDocument > description > keywords > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publication_date"></a>3.7. [Optional] Property ExportedCCSDocument > description > publication_date</strong>  

</summary>
<blockquote>

**Title:** Publication Date

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                   |
| ------------------------------------------------ |
| [item 0](#description_publication_date_anyOf_i0) |
| [item 1](#description_publication_date_anyOf_i1) |

<blockquote>

#### <a name="description_publication_date_anyOf_i0"></a>3.7.1. Property `ExportedCCSDocument > description > publication_date > anyOf > item 0`

|              |             |
| ------------ | ----------- |
| **Type**     | `string`    |
| **Required** | No          |
| **Format**   | `date-time` |

</blockquote>
<blockquote>

#### <a name="description_publication_date_anyOf_i1"></a>3.7.2. Property `ExportedCCSDocument > description > publication_date > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_languages"></a>3.8. [Optional] Property ExportedCCSDocument > description > languages</strong>  

</summary>
<blockquote>

**Title:** Languages

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                            |
| ----------------------------------------- |
| [item 0](#description_languages_anyOf_i0) |
| [item 1](#description_languages_anyOf_i1) |

<blockquote>

#### <a name="description_languages_anyOf_i0"></a>3.8.1. Property `ExportedCCSDocument > description > languages > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                       | Description |
| ----------------------------------------------------- | ----------- |
| [item 0 items](#description_languages_anyOf_i0_items) | -           |

##### <a name="autogenerated_heading_8"></a>3.8.1.1. ExportedCCSDocument > description > languages > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

#### <a name="description_languages_anyOf_i1"></a>3.8.2. Property `ExportedCCSDocument > description > languages > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_license"></a>3.9. [Optional] Property ExportedCCSDocument > description > license</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                      |
| --------------------------------------------------- |
| [DescriptionLicense](#description_license_anyOf_i0) |
| [item 1](#description_license_anyOf_i1)             |

<blockquote>

#### <a name="description_license_anyOf_i0"></a>3.9.1. Property `ExportedCCSDocument > description > license > anyOf > DescriptionLicense`

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/DescriptionLicense                              |

**Description:** Licence in document description.

<details>
<summary>
<strong> <a name="description_license_anyOf_i0_code"></a>3.9.1.1. [Optional] Property ExportedCCSDocument > description > license > anyOf > DescriptionLicense > code</strong>  

</summary>
<blockquote>

**Title:** Code

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                        |
| ----------------------------------------------------- |
| [item 0](#description_license_anyOf_i0_code_anyOf_i0) |
| [item 1](#description_license_anyOf_i0_code_anyOf_i1) |

<blockquote>

###### <a name="description_license_anyOf_i0_code_anyOf_i0"></a>3.9.1.1.1. Property `ExportedCCSDocument > description > license > anyOf > DescriptionLicense > code > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_license_anyOf_i0_code_anyOf_i1"></a>3.9.1.1.2. Property `ExportedCCSDocument > description > license > anyOf > DescriptionLicense > code > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_license_anyOf_i0_text"></a>3.9.1.2. [Optional] Property ExportedCCSDocument > description > license > anyOf > DescriptionLicense > text</strong>  

</summary>
<blockquote>

**Title:** Text

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                        |
| ----------------------------------------------------- |
| [item 0](#description_license_anyOf_i0_text_anyOf_i0) |
| [item 1](#description_license_anyOf_i0_text_anyOf_i1) |

<blockquote>

###### <a name="description_license_anyOf_i0_text_anyOf_i0"></a>3.9.1.2.1. Property `ExportedCCSDocument > description > license > anyOf > DescriptionLicense > text > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_license_anyOf_i0_text_anyOf_i1"></a>3.9.1.2.2. Property `ExportedCCSDocument > description > license > anyOf > DescriptionLicense > text > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

#### <a name="description_license_anyOf_i1"></a>3.9.2. Property `ExportedCCSDocument > description > license > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publishers"></a>3.10. [Optional] Property ExportedCCSDocument > description > publishers</strong>  

</summary>
<blockquote>

**Title:** Publishers

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                             |
| ------------------------------------------ |
| [item 0](#description_publishers_anyOf_i0) |
| [item 1](#description_publishers_anyOf_i1) |

<blockquote>

#### <a name="description_publishers_anyOf_i0"></a>3.10.1. Property `ExportedCCSDocument > description > publishers > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                        | Description |
| ------------------------------------------------------ | ----------- |
| [item 0 items](#description_publishers_anyOf_i0_items) | -           |

##### <a name="autogenerated_heading_9"></a>3.10.1.1. ExportedCCSDocument > description > publishers > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

#### <a name="description_publishers_anyOf_i1"></a>3.10.2. Property `ExportedCCSDocument > description > publishers > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_url_refs"></a>3.11. [Optional] Property ExportedCCSDocument > description > url_refs</strong>  

</summary>
<blockquote>

**Title:** Url Refs

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#description_url_refs_anyOf_i0) |
| [item 1](#description_url_refs_anyOf_i1) |

<blockquote>

#### <a name="description_url_refs_anyOf_i0"></a>3.11.1. Property `ExportedCCSDocument > description > url_refs > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                      | Description |
| ---------------------------------------------------- | ----------- |
| [item 0 items](#description_url_refs_anyOf_i0_items) | -           |

##### <a name="autogenerated_heading_10"></a>3.11.1.1. ExportedCCSDocument > description > url_refs > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

#### <a name="description_url_refs_anyOf_i1"></a>3.11.2. Property `ExportedCCSDocument > description > url_refs > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_references"></a>3.12. [Optional] Property ExportedCCSDocument > description > references</strong>  

</summary>
<blockquote>

**Title:** References

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                             |
| ------------------------------------------ |
| [item 0](#description_references_anyOf_i0) |
| [item 1](#description_references_anyOf_i1) |

<blockquote>

#### <a name="description_references_anyOf_i0"></a>3.12.1. Property `ExportedCCSDocument > description > references > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                      | Description                                 |
| ---------------------------------------------------- | ------------------------------------------- |
| [Identifier](#description_references_anyOf_i0_items) | Unique identifier of a Docling data object. |

##### <a name="autogenerated_heading_11"></a>3.12.1.1. ExportedCCSDocument > description > references > anyOf > item 0 > Identifier

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/Identifier                                      |

**Description:** Unique identifier of a Docling data object.

<details>
<summary>
<strong> <a name="description_references_anyOf_i0_items_type"></a>3.12.1.1.1. [Required] Property ExportedCCSDocument > description > references > anyOf > item 0 > Identifier > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** A string representing a collection or database that contains this data object.

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_references_anyOf_i0_items_value"></a>3.12.1.1.2. [Required] Property ExportedCCSDocument > description > references > anyOf > item 0 > Identifier > value</strong>  

</summary>
<blockquote>

**Title:** Value

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** The identifier value of the data object within a collection or database.

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_references_anyOf_i0_items__name"></a>3.12.1.1.3. [Required] Property ExportedCCSDocument > description > references > anyOf > item 0 > Identifier > _name</strong>  

</summary>
<blockquote>

**Title:** _Name

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** A unique identifier of the data object across Docling, consisting of the concatenation of type and value in lower case, separated by hash (#).

| Restrictions                      |                                                                     |
| --------------------------------- | ------------------------------------------------------------------- |
| **Must match regular expression** | ```^.+#.+$``` [Test](https://regex101.com/?regex=%5E.%2B%23.%2B%24) |

</blockquote>
</details>

</blockquote>
<blockquote>

#### <a name="description_references_anyOf_i1"></a>3.12.2. Property `ExportedCCSDocument > description > references > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publication"></a>3.13. [Optional] Property ExportedCCSDocument > description > publication</strong>  

</summary>
<blockquote>

**Title:** Publication

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** List of publication journals or venues.

<blockquote>

| Any of(Option)                              |
| ------------------------------------------- |
| [item 0](#description_publication_anyOf_i0) |
| [item 1](#description_publication_anyOf_i1) |

<blockquote>

#### <a name="description_publication_anyOf_i0"></a>3.13.1. Property `ExportedCCSDocument > description > publication > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                        | Description                                |
| ------------------------------------------------------ | ------------------------------------------ |
| [Publication](#description_publication_anyOf_i0_items) | Publication details of a journal or venue. |

##### <a name="autogenerated_heading_12"></a>3.13.1.1. ExportedCCSDocument > description > publication > anyOf > item 0 > Publication

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/Publication                                     |

**Description:** Publication details of a journal or venue.

<details>
<summary>
<strong> <a name="description_publication_anyOf_i0_items_identifiers"></a>3.13.1.1.1. [Optional] Property ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > identifiers</strong>  

</summary>
<blockquote>

**Title:** Identifiers

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Unique identifiers of a publication venue.

<blockquote>

| Any of(Option)                                                         |
| ---------------------------------------------------------------------- |
| [item 0](#description_publication_anyOf_i0_items_identifiers_anyOf_i0) |
| [item 1](#description_publication_anyOf_i0_items_identifiers_anyOf_i1) |

<blockquote>

###### <a name="description_publication_anyOf_i0_items_identifiers_anyOf_i0"></a>3.13.1.1.1.1. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > identifiers > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                  | Description                                 |
| -------------------------------------------------------------------------------- | ------------------------------------------- |
| [Identifier](#description_publication_anyOf_i0_items_identifiers_anyOf_i0_items) | Unique identifier of a Docling data object. |

###### <a name="autogenerated_heading_13"></a>3.13.1.1.1.1.1. ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > identifiers > anyOf > item 0 > Identifier

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/Identifier                                      |

**Description:** Unique identifier of a Docling data object.

<details>
<summary>
<strong> <a name="description_publication_anyOf_i0_items_identifiers_anyOf_i0_items_type"></a>3.13.1.1.1.1.1.1. [Required] Property ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > identifiers > anyOf > item 0 > Identifier > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** A string representing a collection or database that contains this data object.

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publication_anyOf_i0_items_identifiers_anyOf_i0_items_value"></a>3.13.1.1.1.1.1.2. [Required] Property ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > identifiers > anyOf > item 0 > Identifier > value</strong>  

</summary>
<blockquote>

**Title:** Value

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** The identifier value of the data object within a collection or database.

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publication_anyOf_i0_items_identifiers_anyOf_i0_items__name"></a>3.13.1.1.1.1.1.3. [Required] Property ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > identifiers > anyOf > item 0 > Identifier > _name</strong>  

</summary>
<blockquote>

**Title:** _Name

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** A unique identifier of the data object across Docling, consisting of the concatenation of type and value in lower case, separated by hash (#).

| Restrictions                      |                                                                     |
| --------------------------------- | ------------------------------------------------------------------- |
| **Must match regular expression** | ```^.+#.+$``` [Test](https://regex101.com/?regex=%5E.%2B%23.%2B%24) |

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="description_publication_anyOf_i0_items_identifiers_anyOf_i1"></a>3.13.1.1.1.2. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > identifiers > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publication_anyOf_i0_items_name"></a>3.13.1.1.2. [Required] Property ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > name</strong>  

</summary>
<blockquote>

**Title:** Name

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** Name of the publication.

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publication_anyOf_i0_items_alternate_names"></a>3.13.1.1.3. [Optional] Property ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > alternate_names</strong>  

</summary>
<blockquote>

**Title:** Alternate Names

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Other names or abbreviations of this publication.

<blockquote>

| Any of(Option)                                                             |
| -------------------------------------------------------------------------- |
| [item 0](#description_publication_anyOf_i0_items_alternate_names_anyOf_i0) |
| [item 1](#description_publication_anyOf_i0_items_alternate_names_anyOf_i1) |

<blockquote>

###### <a name="description_publication_anyOf_i0_items_alternate_names_anyOf_i0"></a>3.13.1.1.3.1. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > alternate_names > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                        | Description |
| -------------------------------------------------------------------------------------- | ----------- |
| [item 0 items](#description_publication_anyOf_i0_items_alternate_names_anyOf_i0_items) | -           |

###### <a name="autogenerated_heading_14"></a>3.13.1.1.3.1.1. ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > alternate_names > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_publication_anyOf_i0_items_alternate_names_anyOf_i1"></a>3.13.1.1.3.2. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > alternate_names > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publication_anyOf_i0_items_type"></a>3.13.1.1.4. [Optional] Property ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > type</strong>  

</summary>
<blockquote>

**Title:** Type

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Type of publication (journal article, conference, review,...).

<blockquote>

| Any of(Option)                                                  |
| --------------------------------------------------------------- |
| [item 0](#description_publication_anyOf_i0_items_type_anyOf_i0) |
| [item 1](#description_publication_anyOf_i0_items_type_anyOf_i1) |

<blockquote>

###### <a name="description_publication_anyOf_i0_items_type_anyOf_i0"></a>3.13.1.1.4.1. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > type > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                             | Description |
| --------------------------------------------------------------------------- | ----------- |
| [item 0 items](#description_publication_anyOf_i0_items_type_anyOf_i0_items) | -           |

###### <a name="autogenerated_heading_15"></a>3.13.1.1.4.1.1. ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > type > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_publication_anyOf_i0_items_type_anyOf_i1"></a>3.13.1.1.4.2. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > type > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publication_anyOf_i0_items_pages"></a>3.13.1.1.5. [Optional] Property ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > pages</strong>  

</summary>
<blockquote>

**Title:** Pages

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Page range in the publication.

<blockquote>

| Any of(Option)                                                   |
| ---------------------------------------------------------------- |
| [item 0](#description_publication_anyOf_i0_items_pages_anyOf_i0) |
| [item 1](#description_publication_anyOf_i0_items_pages_anyOf_i1) |

<blockquote>

###### <a name="description_publication_anyOf_i0_items_pages_anyOf_i0"></a>3.13.1.1.5.1. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > pages > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_publication_anyOf_i0_items_pages_anyOf_i1"></a>3.13.1.1.5.2. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > pages > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publication_anyOf_i0_items_issue"></a>3.13.1.1.6. [Optional] Property ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > issue</strong>  

</summary>
<blockquote>

**Title:** Issue

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Publication issue (issue number).

<blockquote>

| Any of(Option)                                                   |
| ---------------------------------------------------------------- |
| [item 0](#description_publication_anyOf_i0_items_issue_anyOf_i0) |
| [item 1](#description_publication_anyOf_i0_items_issue_anyOf_i1) |

<blockquote>

###### <a name="description_publication_anyOf_i0_items_issue_anyOf_i0"></a>3.13.1.1.6.1. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > issue > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_publication_anyOf_i0_items_issue_anyOf_i1"></a>3.13.1.1.6.2. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > issue > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publication_anyOf_i0_items_volume"></a>3.13.1.1.7. [Optional] Property ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > volume</strong>  

</summary>
<blockquote>

**Title:** Volume

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Publication volume.

<blockquote>

| Any of(Option)                                                    |
| ----------------------------------------------------------------- |
| [item 0](#description_publication_anyOf_i0_items_volume_anyOf_i0) |
| [item 1](#description_publication_anyOf_i0_items_volume_anyOf_i1) |

<blockquote>

###### <a name="description_publication_anyOf_i0_items_volume_anyOf_i0"></a>3.13.1.1.7.1. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > volume > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_publication_anyOf_i0_items_volume_anyOf_i1"></a>3.13.1.1.7.2. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > volume > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_publication_anyOf_i0_items_url"></a>3.13.1.1.8. [Optional] Property ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > url</strong>  

</summary>
<blockquote>

**Title:** Url

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** URL on the publication site.

<blockquote>

| Any of(Option)                                                 |
| -------------------------------------------------------------- |
| [item 0](#description_publication_anyOf_i0_items_url_anyOf_i0) |
| [item 1](#description_publication_anyOf_i0_items_url_anyOf_i1) |

<blockquote>

###### <a name="description_publication_anyOf_i0_items_url_anyOf_i0"></a>3.13.1.1.8.1. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > url > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Format**   | `uri`    |

| Restrictions   |   |
| -------------- | - |
| **Min length** | 1 |

</blockquote>
<blockquote>

###### <a name="description_publication_anyOf_i0_items_url_anyOf_i1"></a>3.13.1.1.8.2. Property `ExportedCCSDocument > description > publication > anyOf > item 0 > Publication > url > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

#### <a name="description_publication_anyOf_i1"></a>3.13.2. Property `ExportedCCSDocument > description > publication > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_reference_count"></a>3.14. [Optional] Property ExportedCCSDocument > description > reference_count</strong>  

</summary>
<blockquote>

**Title:** Reference Count

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Total number of documents referenced by this document.

<blockquote>

| Any of(Option)                                  |
| ----------------------------------------------- |
| [item 0](#description_reference_count_anyOf_i0) |
| [item 1](#description_reference_count_anyOf_i1) |

<blockquote>

#### <a name="description_reference_count_anyOf_i0"></a>3.14.1. Property `ExportedCCSDocument > description > reference_count > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &ge; 0 |

</blockquote>
<blockquote>

#### <a name="description_reference_count_anyOf_i1"></a>3.14.2. Property `ExportedCCSDocument > description > reference_count > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_citation_count"></a>3.15. [Optional] Property ExportedCCSDocument > description > citation_count</strong>  

</summary>
<blockquote>

**Title:** Citation Count

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Total number of citations that this document has received (number of documents in whose bibliography this document appears).

<blockquote>

| Any of(Option)                                 |
| ---------------------------------------------- |
| [item 0](#description_citation_count_anyOf_i0) |
| [item 1](#description_citation_count_anyOf_i1) |

<blockquote>

#### <a name="description_citation_count_anyOf_i0"></a>3.15.1. Property `ExportedCCSDocument > description > citation_count > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &ge; 0 |

</blockquote>
<blockquote>

#### <a name="description_citation_count_anyOf_i1"></a>3.15.2. Property `ExportedCCSDocument > description > citation_count > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_citation_date"></a>3.16. [Optional] Property ExportedCCSDocument > description > citation_date</strong>  

</summary>
<blockquote>

**Title:** Citation Count Date

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Last update date of the citation count.

<blockquote>

| Any of(Option)                                |
| --------------------------------------------- |
| [item 0](#description_citation_date_anyOf_i0) |
| [item 1](#description_citation_date_anyOf_i1) |

<blockquote>

#### <a name="description_citation_date_anyOf_i0"></a>3.16.1. Property `ExportedCCSDocument > description > citation_date > anyOf > item 0`

|              |             |
| ------------ | ----------- |
| **Type**     | `string`    |
| **Required** | No          |
| **Format**   | `date-time` |

</blockquote>
<blockquote>

#### <a name="description_citation_date_anyOf_i1"></a>3.16.2. Property `ExportedCCSDocument > description > citation_date > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_advanced"></a>3.17. [Optional] Property ExportedCCSDocument > description > advanced</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                              |
| ------------------------------------------- |
| [BaseModel](#description_advanced_anyOf_i0) |
| [item 1](#description_advanced_anyOf_i1)    |

<blockquote>

#### <a name="description_advanced_anyOf_i0"></a>3.17.1. Property `ExportedCCSDocument > description > advanced > anyOf > BaseModel`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BaseModel                                                         |

</blockquote>
<blockquote>

#### <a name="description_advanced_anyOf_i1"></a>3.17.2. Property `ExportedCCSDocument > description > advanced > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_analytics"></a>3.18. [Optional] Property ExportedCCSDocument > description > analytics</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                               |
| -------------------------------------------- |
| [BaseModel](#description_analytics_anyOf_i0) |
| [item 1](#description_analytics_anyOf_i1)    |

<blockquote>

#### <a name="description_analytics_anyOf_i0"></a>3.18.1. Property `ExportedCCSDocument > description > analytics > anyOf > BaseModel`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BaseModel                                                         |

</blockquote>
<blockquote>

#### <a name="description_analytics_anyOf_i1"></a>3.18.2. Property `ExportedCCSDocument > description > analytics > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_logs"></a>3.19. [Required] Property ExportedCCSDocument > description > logs</strong>  

</summary>
<blockquote>

**Title:** Logs

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | Yes     |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be | Description                                      |
| ------------------------------- | ------------------------------------------------ |
| [Log](#description_logs_items)  | Log entry to describe an ETL task on a document. |

#### <a name="autogenerated_heading_16"></a>3.19.1. ExportedCCSDocument > description > logs > Log

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/Log                                             |

**Description:** Log entry to describe an ETL task on a document.

<details>
<summary>
<strong> <a name="description_logs_items_task"></a>3.19.1.1. [Optional] Property ExportedCCSDocument > description > logs > Log > task</strong>  

</summary>
<blockquote>

**Title:** Task

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** An identifier of this task. It may be used to identify this task from other tasks of the same agent and type.

<blockquote>

| Any of(Option)                                  |
| ----------------------------------------------- |
| [item 0](#description_logs_items_task_anyOf_i0) |
| [item 1](#description_logs_items_task_anyOf_i1) |

<blockquote>

###### <a name="description_logs_items_task_anyOf_i0"></a>3.19.1.1.1. Property `ExportedCCSDocument > description > logs > Log > task > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_logs_items_task_anyOf_i1"></a>3.19.1.1.2. Property `ExportedCCSDocument > description > logs > Log > task > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_logs_items_agent"></a>3.19.1.2. [Required] Property ExportedCCSDocument > description > logs > Log > agent</strong>  

</summary>
<blockquote>

**Title:** Agent

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** The Docling agent that performed the task, e.g., CCS or CXS.

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_logs_items_type"></a>3.19.1.3. [Required] Property ExportedCCSDocument > description > logs > Log > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** A task category.

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_logs_items_comment"></a>3.19.1.4. [Optional] Property ExportedCCSDocument > description > logs > Log > comment</strong>  

</summary>
<blockquote>

**Title:** Comment

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** A description of the task or any comments in natural language.

<blockquote>

| Any of(Option)                                     |
| -------------------------------------------------- |
| [item 0](#description_logs_items_comment_anyOf_i0) |
| [item 1](#description_logs_items_comment_anyOf_i1) |

<blockquote>

###### <a name="description_logs_items_comment_anyOf_i0"></a>3.19.1.4.1. Property `ExportedCCSDocument > description > logs > Log > comment > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_logs_items_comment_anyOf_i1"></a>3.19.1.4.2. Property `ExportedCCSDocument > description > logs > Log > comment > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_logs_items_date"></a>3.19.1.5. [Required] Property ExportedCCSDocument > description > logs > Log > date</strong>  

</summary>
<blockquote>

**Title:** Date

|              |             |
| ------------ | ----------- |
| **Type**     | `string`    |
| **Required** | Yes         |
| **Format**   | `date-time` |

**Description:** A string representation of the task execution datetime in ISO 8601 format.

</blockquote>
</details>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_collection"></a>3.20. [Optional] Property ExportedCCSDocument > description > collection</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** The collection information of this document.

<blockquote>

| Any of(Option)                                             |
| ---------------------------------------------------------- |
| [CollectionDocumentInfo](#description_collection_anyOf_i0) |
| [item 1](#description_collection_anyOf_i1)                 |

<blockquote>

#### <a name="description_collection_anyOf_i0"></a>3.20.1. Property `ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo`

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/CollectionDocumentInfo                          |

**Description:** Information of a collection of type Document.

<details>
<summary>
<strong> <a name="description_collection_anyOf_i0_name"></a>3.20.1.1. [Optional] Property ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo > name</strong>  

</summary>
<blockquote>

**Title:** Name

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Name of the collection.

<blockquote>

| Any of(Option)                                           |
| -------------------------------------------------------- |
| [item 0](#description_collection_anyOf_i0_name_anyOf_i0) |
| [item 1](#description_collection_anyOf_i0_name_anyOf_i1) |

<blockquote>

###### <a name="description_collection_anyOf_i0_name_anyOf_i0"></a>3.20.1.1.1. Property `ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo > name > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_collection_anyOf_i0_name_anyOf_i1"></a>3.20.1.1.2. Property `ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo > name > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_collection_anyOf_i0_type"></a>3.20.1.2. [Required] Property ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |         |
| ------------ | ------- |
| **Type**     | `const` |
| **Required** | Yes     |

**Description:** The collection type.

Must be one of:
* "Document"
Specific value: `"Document"`

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_collection_anyOf_i0_version"></a>3.20.1.3. [Optional] Property ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo > version</strong>  

</summary>
<blockquote>

**Title:** Version

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** The version of this collection model.

<blockquote>

| Any of(Option)                                              |
| ----------------------------------------------------------- |
| [item 0](#description_collection_anyOf_i0_version_anyOf_i0) |
| [item 1](#description_collection_anyOf_i0_version_anyOf_i1) |

<blockquote>

###### <a name="description_collection_anyOf_i0_version_anyOf_i0"></a>3.20.1.3.1. Property `ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo > version > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

| Restrictions                      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Must match regular expression** | ```^(?P<major>0\|[1-9]\d*)\.(?P<minor>0\|[1-9]\d*)\.(?P<patch>0\|[1-9]\d*)(?:-(?P<prerelease>(?:0\|[1-9]\d*\|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0\|[1-9]\d*\|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$``` [Test](https://regex101.com/?regex=%5E%28%3FP%3Cmajor%3E0%7C%5B1-9%5D%5Cd%2A%29%5C.%28%3FP%3Cminor%3E0%7C%5B1-9%5D%5Cd%2A%29%5C.%28%3FP%3Cpatch%3E0%7C%5B1-9%5D%5Cd%2A%29%28%3F%3A-%28%3FP%3Cprerelease%3E%28%3F%3A0%7C%5B1-9%5D%5Cd%2A%7C%5Cd%2A%5Ba-zA-Z-%5D%5B0-9a-zA-Z-%5D%2A%29%28%3F%3A%5C.%28%3F%3A0%7C%5B1-9%5D%5Cd%2A%7C%5Cd%2A%5Ba-zA-Z-%5D%5B0-9a-zA-Z-%5D%2A%29%29%2A%29%29%3F%28%3F%3A%5C%2B%28%3FP%3Cbuildmetadata%3E%5B0-9a-zA-Z-%5D%2B%28%3F%3A%5C.%5B0-9a-zA-Z-%5D%2B%29%2A%29%29%3F%24) |

</blockquote>
<blockquote>

###### <a name="description_collection_anyOf_i0_version_anyOf_i1"></a>3.20.1.3.2. Property `ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo > version > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_collection_anyOf_i0_alias"></a>3.20.1.4. [Optional] Property ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo > alias</strong>  

</summary>
<blockquote>

**Title:** Alias

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** A list of tags (aliases) for the collection.

<blockquote>

| Any of(Option)                                            |
| --------------------------------------------------------- |
| [item 0](#description_collection_anyOf_i0_alias_anyOf_i0) |
| [item 1](#description_collection_anyOf_i0_alias_anyOf_i1) |

<blockquote>

###### <a name="description_collection_anyOf_i0_alias_anyOf_i0"></a>3.20.1.4.1. Property `ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo > alias > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                       | Description |
| --------------------------------------------------------------------- | ----------- |
| [item 0 items](#description_collection_anyOf_i0_alias_anyOf_i0_items) | -           |

###### <a name="autogenerated_heading_17"></a>3.20.1.4.1.1. ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo > alias > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="description_collection_anyOf_i0_alias_anyOf_i1"></a>3.20.1.4.2. Property `ExportedCCSDocument > description > collection > anyOf > CollectionDocumentInfo > alias > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

#### <a name="description_collection_anyOf_i1"></a>3.20.2. Property `ExportedCCSDocument > description > collection > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_acquisition"></a>3.21. [Optional] Property ExportedCCSDocument > description > acquisition</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Information on how the document was obtained, for data governance purposes.

<blockquote>

| Any of(Option)                                   |
| ------------------------------------------------ |
| [Acquisition](#description_acquisition_anyOf_i0) |
| [item 1](#description_acquisition_anyOf_i1)      |

<blockquote>

#### <a name="description_acquisition_anyOf_i0"></a>3.21.1. Property `ExportedCCSDocument > description > acquisition > anyOf > Acquisition`

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/Acquisition                                     |

**Description:** Information on how the data was obtained.

<details>
<summary>
<strong> <a name="description_acquisition_anyOf_i0_type"></a>3.21.1.1. [Required] Property ExportedCCSDocument > description > acquisition > anyOf > Acquisition > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

**Description:** The method to obtain the data.

Must be one of:
* "API"
* "FTP"
* "Download"
* "Link"
* "Web scraping/Crawling"
* "Other"

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_acquisition_anyOf_i0_date"></a>3.21.1.2. [Optional] Property ExportedCCSDocument > description > acquisition > anyOf > Acquisition > date</strong>  

</summary>
<blockquote>

**Title:** Date

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** A string representation of the acquisition datetime in ISO 8601 format.

<blockquote>

| Any of(Option)                                            |
| --------------------------------------------------------- |
| [item 0](#description_acquisition_anyOf_i0_date_anyOf_i0) |
| [item 1](#description_acquisition_anyOf_i0_date_anyOf_i1) |

<blockquote>

###### <a name="description_acquisition_anyOf_i0_date_anyOf_i0"></a>3.21.1.2.1. Property `ExportedCCSDocument > description > acquisition > anyOf > Acquisition > date > anyOf > item 0`

|              |             |
| ------------ | ----------- |
| **Type**     | `string`    |
| **Required** | No          |
| **Format**   | `date-time` |

</blockquote>
<blockquote>

###### <a name="description_acquisition_anyOf_i0_date_anyOf_i1"></a>3.21.1.2.2. Property `ExportedCCSDocument > description > acquisition > anyOf > Acquisition > date > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_acquisition_anyOf_i0_link"></a>3.21.1.3. [Optional] Property ExportedCCSDocument > description > acquisition > anyOf > Acquisition > link</strong>  

</summary>
<blockquote>

**Title:** Link

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Link to the data source of this document.

<blockquote>

| Any of(Option)                                            |
| --------------------------------------------------------- |
| [item 0](#description_acquisition_anyOf_i0_link_anyOf_i0) |
| [item 1](#description_acquisition_anyOf_i0_link_anyOf_i1) |

<blockquote>

###### <a name="description_acquisition_anyOf_i0_link_anyOf_i0"></a>3.21.1.3.1. Property `ExportedCCSDocument > description > acquisition > anyOf > Acquisition > link > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Format**   | `uri`    |

| Restrictions   |   |
| -------------- | - |
| **Min length** | 1 |

</blockquote>
<blockquote>

###### <a name="description_acquisition_anyOf_i0_link_anyOf_i1"></a>3.21.1.3.2. Property `ExportedCCSDocument > description > acquisition > anyOf > Acquisition > link > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="description_acquisition_anyOf_i0_size"></a>3.21.1.4. [Optional] Property ExportedCCSDocument > description > acquisition > anyOf > Acquisition > size</strong>  

</summary>
<blockquote>

**Title:** Size

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** Size in bytes of the raw document from the data source.

<blockquote>

| Any of(Option)                                            |
| --------------------------------------------------------- |
| [item 0](#description_acquisition_anyOf_i0_size_anyOf_i0) |
| [item 1](#description_acquisition_anyOf_i0_size_anyOf_i1) |

<blockquote>

###### <a name="description_acquisition_anyOf_i0_size_anyOf_i0"></a>3.21.1.4.1. Property `ExportedCCSDocument > description > acquisition > anyOf > Acquisition > size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &ge; 0 |

</blockquote>
<blockquote>

###### <a name="description_acquisition_anyOf_i0_size_anyOf_i1"></a>3.21.1.4.2. Property `ExportedCCSDocument > description > acquisition > anyOf > Acquisition > size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

#### <a name="description_acquisition_anyOf_i1"></a>3.21.2. Property `ExportedCCSDocument > description > acquisition > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info"></a>4. [Required] Property ExportedCCSDocument > file-info</strong>  

</summary>
<blockquote>

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | Yes                                                     |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/CCSFileInfoObject                               |

**Description:** File info object.

<details>
<summary>
<strong> <a name="file-info_filename"></a>4.1. [Required] Property ExportedCCSDocument > file-info > filename</strong>  

</summary>
<blockquote>

**Title:** Filename

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** The name of a persistent object that created this data object

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_filename-prov"></a>4.2. [Optional] Property ExportedCCSDocument > file-info > filename-prov</strong>  

</summary>
<blockquote>

**Title:** Filename-Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** The provenance of this data object, e.g. an archive file, a URL, or any other repository.

<blockquote>

| Any of(Option)                              |
| ------------------------------------------- |
| [item 0](#file-info_filename-prov_anyOf_i0) |
| [item 1](#file-info_filename-prov_anyOf_i1) |

<blockquote>

#### <a name="file-info_filename-prov_anyOf_i0"></a>4.2.1. Property `ExportedCCSDocument > file-info > filename-prov > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

#### <a name="file-info_filename-prov_anyOf_i1"></a>4.2.2. Property `ExportedCCSDocument > file-info > filename-prov > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_document-hash"></a>4.3. [Required] Property ExportedCCSDocument > file-info > document-hash</strong>  

</summary>
<blockquote>

**Title:** Document-Hash

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** A unique identifier of this data object within a collection of a Docling database

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_-pages"></a>4.4. [Optional] Property ExportedCCSDocument > file-info > #-pages</strong>  

</summary>
<blockquote>

**Title:** #-Pages

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                       |
| ------------------------------------ |
| [item 0](#file-info_-pages_anyOf_i0) |
| [item 1](#file-info_-pages_anyOf_i1) |

<blockquote>

#### <a name="file-info_-pages_anyOf_i0"></a>4.4.1. Property `ExportedCCSDocument > file-info > #-pages > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
<blockquote>

#### <a name="file-info_-pages_anyOf_i1"></a>4.4.2. Property `ExportedCCSDocument > file-info > #-pages > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_collection-name"></a>4.5. [Optional] Property ExportedCCSDocument > file-info > collection-name</strong>  

</summary>
<blockquote>

**Title:** Collection-Name

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                |
| --------------------------------------------- |
| [item 0](#file-info_collection-name_anyOf_i0) |
| [item 1](#file-info_collection-name_anyOf_i1) |

<blockquote>

#### <a name="file-info_collection-name_anyOf_i0"></a>4.5.1. Property `ExportedCCSDocument > file-info > collection-name > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

#### <a name="file-info_collection-name_anyOf_i1"></a>4.5.2. Property `ExportedCCSDocument > file-info > collection-name > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_description"></a>4.6. [Optional] Property ExportedCCSDocument > file-info > description</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                            |
| --------------------------------------------------------- |
| [CCSFileInfoDescription](#file-info_description_anyOf_i0) |
| [item 1](#file-info_description_anyOf_i1)                 |

<blockquote>

#### <a name="file-info_description_anyOf_i0"></a>4.6.1. Property `ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription`

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/CCSFileInfoDescription                          |

**Description:** File info description.

<details>
<summary>
<strong> <a name="file-info_description_anyOf_i0_author"></a>4.6.1.1. [Optional] Property ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > author</strong>  

</summary>
<blockquote>

**Title:** Author

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                            |
| --------------------------------------------------------- |
| [item 0](#file-info_description_anyOf_i0_author_anyOf_i0) |
| [item 1](#file-info_description_anyOf_i0_author_anyOf_i1) |

<blockquote>

###### <a name="file-info_description_anyOf_i0_author_anyOf_i0"></a>4.6.1.1.1. Property `ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > author > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                       | Description |
| --------------------------------------------------------------------- | ----------- |
| [item 0 items](#file-info_description_anyOf_i0_author_anyOf_i0_items) | -           |

###### <a name="autogenerated_heading_18"></a>4.6.1.1.1.1. ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > author > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="file-info_description_anyOf_i0_author_anyOf_i1"></a>4.6.1.1.2. Property `ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > author > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_description_anyOf_i0_keywords"></a>4.6.1.2. [Optional] Property ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > keywords</strong>  

</summary>
<blockquote>

**Title:** Keywords

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                              |
| ----------------------------------------------------------- |
| [item 0](#file-info_description_anyOf_i0_keywords_anyOf_i0) |
| [item 1](#file-info_description_anyOf_i0_keywords_anyOf_i1) |

<blockquote>

###### <a name="file-info_description_anyOf_i0_keywords_anyOf_i0"></a>4.6.1.2.1. Property `ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > keywords > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="file-info_description_anyOf_i0_keywords_anyOf_i1"></a>4.6.1.2.2. Property `ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > keywords > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_description_anyOf_i0_subject"></a>4.6.1.3. [Optional] Property ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > subject</strong>  

</summary>
<blockquote>

**Title:** Subject

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                             |
| ---------------------------------------------------------- |
| [item 0](#file-info_description_anyOf_i0_subject_anyOf_i0) |
| [item 1](#file-info_description_anyOf_i0_subject_anyOf_i1) |

<blockquote>

###### <a name="file-info_description_anyOf_i0_subject_anyOf_i0"></a>4.6.1.3.1. Property `ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > subject > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="file-info_description_anyOf_i0_subject_anyOf_i1"></a>4.6.1.3.2. Property `ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > subject > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_description_anyOf_i0_title"></a>4.6.1.4. [Optional] Property ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > title</strong>  

</summary>
<blockquote>

**Title:** Title

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                           |
| -------------------------------------------------------- |
| [item 0](#file-info_description_anyOf_i0_title_anyOf_i0) |
| [item 1](#file-info_description_anyOf_i0_title_anyOf_i1) |

<blockquote>

###### <a name="file-info_description_anyOf_i0_title_anyOf_i0"></a>4.6.1.4.1. Property `ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > title > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="file-info_description_anyOf_i0_title_anyOf_i1"></a>4.6.1.4.2. Property `ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > title > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_description_anyOf_i0_creation_date"></a>4.6.1.5. [Optional] Property ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > creation_date</strong>  

</summary>
<blockquote>

**Title:** Creation Date

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                   |
| ---------------------------------------------------------------- |
| [item 0](#file-info_description_anyOf_i0_creation_date_anyOf_i0) |
| [item 1](#file-info_description_anyOf_i0_creation_date_anyOf_i1) |

<blockquote>

###### <a name="file-info_description_anyOf_i0_creation_date_anyOf_i0"></a>4.6.1.5.1. Property `ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > creation_date > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="file-info_description_anyOf_i0_creation_date_anyOf_i1"></a>4.6.1.5.2. Property `ExportedCCSDocument > file-info > description > anyOf > CCSFileInfoDescription > creation_date > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

#### <a name="file-info_description_anyOf_i1"></a>4.6.2. Property `ExportedCCSDocument > file-info > description > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_page-hashes"></a>4.7. [Optional] Property ExportedCCSDocument > file-info > page-hashes</strong>  

</summary>
<blockquote>

**Title:** Page-Hashes

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                            |
| ----------------------------------------- |
| [item 0](#file-info_page-hashes_anyOf_i0) |
| [item 1](#file-info_page-hashes_anyOf_i1) |

<blockquote>

#### <a name="file-info_page-hashes_anyOf_i0"></a>4.7.1. Property `ExportedCCSDocument > file-info > page-hashes > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                        | Description     |
| ------------------------------------------------------ | --------------- |
| [PageReference](#file-info_page-hashes_anyOf_i0_items) | Page reference. |

##### <a name="autogenerated_heading_19"></a>4.7.1.1. ExportedCCSDocument > file-info > page-hashes > anyOf > item 0 > PageReference

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/PageReference                                                     |

**Description:** Page reference.

<details>
<summary>
<strong> <a name="file-info_page-hashes_anyOf_i0_items_hash"></a>4.7.1.1.1. [Required] Property ExportedCCSDocument > file-info > page-hashes > anyOf > item 0 > PageReference > hash</strong>  

</summary>
<blockquote>

**Title:** Hash

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_page-hashes_anyOf_i0_items_model"></a>4.7.1.1.2. [Required] Property ExportedCCSDocument > file-info > page-hashes > anyOf > item 0 > PageReference > model</strong>  

</summary>
<blockquote>

**Title:** Model

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="file-info_page-hashes_anyOf_i0_items_page"></a>4.7.1.1.3. [Required] Property ExportedCCSDocument > file-info > page-hashes > anyOf > item 0 > PageReference > page</strong>  

</summary>
<blockquote>

**Title:** Page

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
</details>

</blockquote>
<blockquote>

#### <a name="file-info_page-hashes_anyOf_i1"></a>4.7.2. Property `ExportedCCSDocument > file-info > page-hashes > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="main-text"></a>5. [Optional] Property ExportedCCSDocument > main-text</strong>  

</summary>
<blockquote>

**Title:** Main-Text

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                |
| ----------------------------- |
| [item 0](#main-text_anyOf_i0) |
| [item 1](#main-text_anyOf_i1) |

<blockquote>

### <a name="main-text_anyOf_i0"></a>5.1. Property `ExportedCCSDocument > main-text > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be           | Description |
| ----------------------------------------- | ----------- |
| [item 0 items](#main-text_anyOf_i0_items) | -           |

#### <a name="autogenerated_heading_20"></a>5.1.1. ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |

<blockquote>

| Any of(Option)                                 |
| ---------------------------------------------- |
| [Ref](#main-text_anyOf_i0_items_anyOf_i0)      |
| [BaseText](#main-text_anyOf_i0_items_anyOf_i1) |

<blockquote>

##### <a name="main-text_anyOf_i0_items_anyOf_i0"></a>5.1.1.1. Property `ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > Ref`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/Ref                                                               |

**Description:** Reference.

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i0_name"></a>5.1.1.1.1. [Required] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > Ref > name</strong>  

</summary>
<blockquote>

**Title:** Name

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i0_type"></a>5.1.1.1.2. [Required] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > Ref > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i0_ref"></a>5.1.1.1.3. [Required] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > Ref > $ref</strong>  

</summary>
<blockquote>

**Title:** $Ref

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

</blockquote>
<blockquote>

##### <a name="main-text_anyOf_i0_items_anyOf_i1"></a>5.1.1.2. Property `ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BaseText                                                          |

**Description:** Base model for text objects.

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i1_text"></a>5.1.1.2.1. [Required] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > text</strong>  

</summary>
<blockquote>

**Title:** Text

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i1_type"></a>5.1.1.2.2. [Required] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i1_name"></a>5.1.1.2.3. [Optional] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > name</strong>  

</summary>
<blockquote>

**Title:** Name

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                             |
| ---------------------------------------------------------- |
| [item 0](#main-text_anyOf_i0_items_anyOf_i1_name_anyOf_i0) |
| [item 1](#main-text_anyOf_i0_items_anyOf_i1_name_anyOf_i1) |

<blockquote>

###### <a name="main-text_anyOf_i0_items_anyOf_i1_name_anyOf_i0"></a>5.1.1.2.3.1. Property `ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > name > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="main-text_anyOf_i0_items_anyOf_i1_name_anyOf_i1"></a>5.1.1.2.3.2. Property `ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > name > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i1_font"></a>5.1.1.2.4. [Optional] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > font</strong>  

</summary>
<blockquote>

**Title:** Font

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                             |
| ---------------------------------------------------------- |
| [item 0](#main-text_anyOf_i0_items_anyOf_i1_font_anyOf_i0) |
| [item 1](#main-text_anyOf_i0_items_anyOf_i1_font_anyOf_i1) |

<blockquote>

###### <a name="main-text_anyOf_i0_items_anyOf_i1_font_anyOf_i0"></a>5.1.1.2.4.1. Property `ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > font > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="main-text_anyOf_i0_items_anyOf_i1_font_anyOf_i1"></a>5.1.1.2.4.2. Property `ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > font > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i1_prov"></a>5.1.1.2.5. [Optional] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov</strong>  

</summary>
<blockquote>

**Title:** Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                             |
| ---------------------------------------------------------- |
| [item 0](#main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0) |
| [item 1](#main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i1) |

<blockquote>

###### <a name="main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0"></a>5.1.1.2.5.1. Property `ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                | Description |
| -------------------------------------------------------------- | ----------- |
| [Prov](#main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0_items) | Provenance. |

###### <a name="autogenerated_heading_21"></a>5.1.1.2.5.1.1. ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov > anyOf > item 0 > Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/Prov                                                              |

**Description:** Provenance.

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0_items_bbox"></a>5.1.1.2.5.1.1.1. [Required] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov > anyOf > item 0 > Prov > bbox</strong>  

</summary>
<blockquote>

**Title:** Bbox

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                 | Description |
| ------------------------------------------------------------------------------- | ----------- |
| [bbox items](#main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0_items_bbox_items) | -           |

###### <a name="autogenerated_heading_22"></a>5.1.1.2.5.1.1.1.1. ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov > anyOf > item 0 > Prov > bbox > bbox items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0_items_page"></a>5.1.1.2.5.1.1.2. [Required] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov > anyOf > item 0 > Prov > page</strong>  

</summary>
<blockquote>

**Title:** Page

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0_items_span"></a>5.1.1.2.5.1.1.3. [Required] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov > anyOf > item 0 > Prov > span</strong>  

</summary>
<blockquote>

**Title:** Span

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | Yes                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                 | Description |
| ------------------------------------------------------------------------------- | ----------- |
| [span items](#main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0_items_span_items) | -           |

###### <a name="autogenerated_heading_23"></a>5.1.1.2.5.1.1.3.1. ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov > anyOf > item 0 > Prov > span > span items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0_items___ref_s3_data"></a>5.1.1.2.5.1.1.4. [Optional] Property ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data</strong>  

</summary>
<blockquote>

**Title:**   Ref S3 Data

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                          |
| --------------------------------------------------------------------------------------- |
| [item 0](#main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0_items___ref_s3_data_anyOf_i0) |
| [item 1](#main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0_items___ref_s3_data_anyOf_i1) |

<blockquote>

###### <a name="main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0_items___ref_s3_data_anyOf_i0"></a>5.1.1.2.5.1.1.4.1. Property `ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i0_items___ref_s3_data_anyOf_i1"></a>5.1.1.2.5.1.1.4.2. Property `ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="main-text_anyOf_i0_items_anyOf_i1_prov_anyOf_i1"></a>5.1.1.2.5.2. Property `ExportedCCSDocument > main-text > anyOf > item 0 > item 0 items > anyOf > BaseText > prov > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>

</blockquote>

</blockquote>
<blockquote>

### <a name="main-text_anyOf_i1"></a>5.2. Property `ExportedCCSDocument > main-text > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="figures"></a>6. [Optional] Property ExportedCCSDocument > figures</strong>  

</summary>
<blockquote>

**Title:** Figures

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)              |
| --------------------------- |
| [item 0](#figures_anyOf_i0) |
| [item 1](#figures_anyOf_i1) |

<blockquote>

### <a name="figures_anyOf_i0"></a>6.1. Property `ExportedCCSDocument > figures > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be     | Description |
| ----------------------------------- | ----------- |
| [BaseCell](#figures_anyOf_i0_items) | Base cell.  |

#### <a name="autogenerated_heading_24"></a>6.1.1. ExportedCCSDocument > figures > anyOf > item 0 > BaseCell

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BaseCell                                                          |

**Description:** Base cell.

<details>
<summary>
<strong> <a name="figures_anyOf_i0_items_bounding-box"></a>6.1.1.1. [Optional] Property ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > bounding-box</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                        |
| --------------------------------------------------------------------- |
| [BoundingBoxContainer](#figures_anyOf_i0_items_bounding-box_anyOf_i0) |
| [item 1](#figures_anyOf_i0_items_bounding-box_anyOf_i1)               |

<blockquote>

###### <a name="figures_anyOf_i0_items_bounding-box_anyOf_i0"></a>6.1.1.1.1. Property `ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > bounding-box > anyOf > BoundingBoxContainer`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BoundingBoxContainer                                              |

**Description:** Bounding box container.

<details>
<summary>
<strong> <a name="figures_anyOf_i0_items_bounding-box_anyOf_i0_min"></a>6.1.1.1.1.1. [Required] Property ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > bounding-box > anyOf > BoundingBoxContainer > min</strong>  

</summary>
<blockquote>

**Title:** Min

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                      | Description |
| -------------------------------------------------------------------- | ----------- |
| [min items](#figures_anyOf_i0_items_bounding-box_anyOf_i0_min_items) | -           |

###### <a name="autogenerated_heading_25"></a>6.1.1.1.1.1.1. ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > bounding-box > anyOf > BoundingBoxContainer > min > min items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="figures_anyOf_i0_items_bounding-box_anyOf_i0_max"></a>6.1.1.1.1.2. [Required] Property ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > bounding-box > anyOf > BoundingBoxContainer > max</strong>  

</summary>
<blockquote>

**Title:** Max

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                      | Description |
| -------------------------------------------------------------------- | ----------- |
| [max items](#figures_anyOf_i0_items_bounding-box_anyOf_i0_max_items) | -           |

###### <a name="autogenerated_heading_26"></a>6.1.1.1.1.2.1. ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > bounding-box > anyOf > BoundingBoxContainer > max > max items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="figures_anyOf_i0_items_bounding-box_anyOf_i1"></a>6.1.1.1.2. Property `ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > bounding-box > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="figures_anyOf_i0_items_prov"></a>6.1.1.2. [Optional] Property ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov</strong>  

</summary>
<blockquote>

**Title:** Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                  |
| ----------------------------------------------- |
| [item 0](#figures_anyOf_i0_items_prov_anyOf_i0) |
| [item 1](#figures_anyOf_i0_items_prov_anyOf_i1) |

<blockquote>

###### <a name="figures_anyOf_i0_items_prov_anyOf_i0"></a>6.1.1.2.1. Property `ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                     | Description |
| --------------------------------------------------- | ----------- |
| [Prov](#figures_anyOf_i0_items_prov_anyOf_i0_items) | Provenance. |

###### <a name="autogenerated_heading_27"></a>6.1.1.2.1.1. ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/Prov                                                              |

**Description:** Provenance.

<details>
<summary>
<strong> <a name="figures_anyOf_i0_items_prov_anyOf_i0_items_bbox"></a>6.1.1.2.1.1.1. [Required] Property ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > bbox</strong>  

</summary>
<blockquote>

**Title:** Bbox

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                      | Description |
| -------------------------------------------------------------------- | ----------- |
| [bbox items](#figures_anyOf_i0_items_prov_anyOf_i0_items_bbox_items) | -           |

###### <a name="autogenerated_heading_28"></a>6.1.1.2.1.1.1.1. ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > bbox > bbox items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="figures_anyOf_i0_items_prov_anyOf_i0_items_page"></a>6.1.1.2.1.1.2. [Required] Property ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > page</strong>  

</summary>
<blockquote>

**Title:** Page

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="figures_anyOf_i0_items_prov_anyOf_i0_items_span"></a>6.1.1.2.1.1.3. [Required] Property ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > span</strong>  

</summary>
<blockquote>

**Title:** Span

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | Yes                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                      | Description |
| -------------------------------------------------------------------- | ----------- |
| [span items](#figures_anyOf_i0_items_prov_anyOf_i0_items_span_items) | -           |

###### <a name="autogenerated_heading_29"></a>6.1.1.2.1.1.3.1. ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > span > span items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="figures_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data"></a>6.1.1.2.1.1.4. [Optional] Property ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > __ref_s3_data</strong>  

</summary>
<blockquote>

**Title:**   Ref S3 Data

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                               |
| ---------------------------------------------------------------------------- |
| [item 0](#figures_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0) |
| [item 1](#figures_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1) |

<blockquote>

###### <a name="figures_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0"></a>6.1.1.2.1.1.4.1. Property `ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="figures_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1"></a>6.1.1.2.1.1.4.2. Property `ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="figures_anyOf_i0_items_prov_anyOf_i1"></a>6.1.1.2.2. Property `ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > prov > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="figures_anyOf_i0_items_text"></a>6.1.1.3. [Optional] Property ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > text</strong>  

</summary>
<blockquote>

**Title:** Text

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                  |
| ----------------------------------------------- |
| [item 0](#figures_anyOf_i0_items_text_anyOf_i0) |
| [item 1](#figures_anyOf_i0_items_text_anyOf_i1) |

<blockquote>

###### <a name="figures_anyOf_i0_items_text_anyOf_i0"></a>6.1.1.3.1. Property `ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > text > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="figures_anyOf_i0_items_text_anyOf_i1"></a>6.1.1.3.2. Property `ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > text > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="figures_anyOf_i0_items_type"></a>6.1.1.4. [Required] Property ExportedCCSDocument > figures > anyOf > item 0 > BaseCell > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

</blockquote>
<blockquote>

### <a name="figures_anyOf_i1"></a>6.2. Property `ExportedCCSDocument > figures > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables"></a>7. [Optional] Property ExportedCCSDocument > tables</strong>  

</summary>
<blockquote>

**Title:** Tables

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)             |
| -------------------------- |
| [item 0](#tables_anyOf_i0) |
| [item 1](#tables_anyOf_i1) |

<blockquote>

### <a name="tables_anyOf_i0"></a>7.1. Property `ExportedCCSDocument > tables > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be | Description |
| ------------------------------- | ----------- |
| [Table](#tables_anyOf_i0_items) | Table.      |

#### <a name="autogenerated_heading_30"></a>7.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/Table                                                             |

**Description:** Table.

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_bounding-box"></a>7.1.1.1. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > bounding-box</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                       |
| -------------------------------------------------------------------- |
| [BoundingBoxContainer](#tables_anyOf_i0_items_bounding-box_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_bounding-box_anyOf_i1)               |

<blockquote>

###### <a name="tables_anyOf_i0_items_bounding-box_anyOf_i0"></a>7.1.1.1.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > bounding-box > anyOf > BoundingBoxContainer`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BoundingBoxContainer                                              |

**Description:** Bounding box container.

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_bounding-box_anyOf_i0_min"></a>7.1.1.1.1.1. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > bounding-box > anyOf > BoundingBoxContainer > min</strong>  

</summary>
<blockquote>

**Title:** Min

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                     | Description |
| ------------------------------------------------------------------- | ----------- |
| [min items](#tables_anyOf_i0_items_bounding-box_anyOf_i0_min_items) | -           |

###### <a name="autogenerated_heading_31"></a>7.1.1.1.1.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > bounding-box > anyOf > BoundingBoxContainer > min > min items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_bounding-box_anyOf_i0_max"></a>7.1.1.1.1.2. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > bounding-box > anyOf > BoundingBoxContainer > max</strong>  

</summary>
<blockquote>

**Title:** Max

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                     | Description |
| ------------------------------------------------------------------- | ----------- |
| [max items](#tables_anyOf_i0_items_bounding-box_anyOf_i0_max_items) | -           |

###### <a name="autogenerated_heading_32"></a>7.1.1.1.1.2.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > bounding-box > anyOf > BoundingBoxContainer > max > max items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_bounding-box_anyOf_i1"></a>7.1.1.1.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > bounding-box > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_prov"></a>7.1.1.2. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > prov</strong>  

</summary>
<blockquote>

**Title:** Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                 |
| ---------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_prov_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_prov_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_prov_anyOf_i0"></a>7.1.1.2.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > prov > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                    | Description |
| -------------------------------------------------- | ----------- |
| [Prov](#tables_anyOf_i0_items_prov_anyOf_i0_items) | Provenance. |

###### <a name="autogenerated_heading_33"></a>7.1.1.2.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > prov > anyOf > item 0 > Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/Prov                                                              |

**Description:** Provenance.

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_prov_anyOf_i0_items_bbox"></a>7.1.1.2.1.1.1. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > prov > anyOf > item 0 > Prov > bbox</strong>  

</summary>
<blockquote>

**Title:** Bbox

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                     | Description |
| ------------------------------------------------------------------- | ----------- |
| [bbox items](#tables_anyOf_i0_items_prov_anyOf_i0_items_bbox_items) | -           |

###### <a name="autogenerated_heading_34"></a>7.1.1.2.1.1.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > prov > anyOf > item 0 > Prov > bbox > bbox items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_prov_anyOf_i0_items_page"></a>7.1.1.2.1.1.2. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > prov > anyOf > item 0 > Prov > page</strong>  

</summary>
<blockquote>

**Title:** Page

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_prov_anyOf_i0_items_span"></a>7.1.1.2.1.1.3. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > prov > anyOf > item 0 > Prov > span</strong>  

</summary>
<blockquote>

**Title:** Span

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | Yes                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                     | Description |
| ------------------------------------------------------------------- | ----------- |
| [span items](#tables_anyOf_i0_items_prov_anyOf_i0_items_span_items) | -           |

###### <a name="autogenerated_heading_35"></a>7.1.1.2.1.1.3.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > prov > anyOf > item 0 > Prov > span > span items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data"></a>7.1.1.2.1.1.4. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > prov > anyOf > item 0 > Prov > __ref_s3_data</strong>  

</summary>
<blockquote>

**Title:**   Ref S3 Data

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                              |
| --------------------------------------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0"></a>7.1.1.2.1.1.4.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1"></a>7.1.1.2.1.1.4.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_prov_anyOf_i1"></a>7.1.1.2.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > prov > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_text"></a>7.1.1.3. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > text</strong>  

</summary>
<blockquote>

**Title:** Text

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                 |
| ---------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_text_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_text_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_text_anyOf_i0"></a>7.1.1.3.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > text > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_text_anyOf_i1"></a>7.1.1.3.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > text > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_type"></a>7.1.1.4. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_-cols"></a>7.1.1.5. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > #-cols</strong>  

</summary>
<blockquote>

**Title:** #-Cols

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_-rows"></a>7.1.1.6. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > #-rows</strong>  

</summary>
<blockquote>

**Title:** #-Rows

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data"></a>7.1.1.7. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data</strong>  

</summary>
<blockquote>

**Title:** Data

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                 |
| ---------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_data_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_data_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0"></a>7.1.1.7.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0`

|              |                  |
| ------------ | ---------------- |
| **Type**     | `array of array` |
| **Required** | No               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                            | Description |
| ---------------------------------------------------------- | ----------- |
| [item 0 items](#tables_anyOf_i0_items_data_anyOf_i0_items) | -           |

###### <a name="autogenerated_heading_36"></a>7.1.1.7.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                        | Description |
| ---------------------------------------------------------------------- | ----------- |
| [item 0 items items](#tables_anyOf_i0_items_data_anyOf_i0_items_items) | -           |

###### <a name="autogenerated_heading_37"></a>7.1.1.7.1.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |

<blockquote>

| Any of(Option)                                                            |
| ------------------------------------------------------------------------- |
| [GlmTableCell](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0) |
| [TableCell](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1)    |

<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0"></a>7.1.1.7.1.1.1.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/GlmTableCell                                                      |

**Description:** Glm Table cell.

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_bbox"></a>7.1.1.7.1.1.1.1.1. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > bbox</strong>  

</summary>
<blockquote>

**Title:** Bbox

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                    |
| --------------------------------------------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_bbox_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_bbox_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_bbox_anyOf_i0"></a>7.1.1.7.1.1.1.1.1.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > bbox > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                               | Description |
| --------------------------------------------------------------------------------------------- | ----------- |
| [item 0 items](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_bbox_anyOf_i0_items) | -           |

###### <a name="autogenerated_heading_38"></a>7.1.1.7.1.1.1.1.1.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > bbox > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_bbox_anyOf_i1"></a>7.1.1.7.1.1.1.1.1.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > bbox > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_spans"></a>7.1.1.7.1.1.1.1.2. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > spans</strong>  

</summary>
<blockquote>

**Title:** Spans

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                     |
| ---------------------------------------------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_spans_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_spans_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_spans_anyOf_i0"></a>7.1.1.7.1.1.1.1.2.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > spans > anyOf > item 0`

|              |                  |
| ------------ | ---------------- |
| **Type**     | `array of array` |
| **Required** | No               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                                | Description |
| ---------------------------------------------------------------------------------------------- | ----------- |
| [item 0 items](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_spans_anyOf_i0_items) | -           |

###### <a name="autogenerated_heading_39"></a>7.1.1.7.1.1.1.1.2.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > spans > anyOf > item 0 > item 0 items

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | No                 |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                                            | Description |
| ---------------------------------------------------------------------------------------------------------- | ----------- |
| [item 0 items items](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_spans_anyOf_i0_items_items) | -           |

###### <a name="autogenerated_heading_40"></a>7.1.1.7.1.1.1.1.2.1.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > spans > anyOf > item 0 > item 0 items > item 0 items items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_spans_anyOf_i1"></a>7.1.1.7.1.1.1.1.2.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > spans > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_text"></a>7.1.1.7.1.1.1.1.3. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > text</strong>  

</summary>
<blockquote>

**Title:** Text

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_type"></a>7.1.1.7.1.1.1.1.4. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col"></a>7.1.1.7.1.1.1.1.5. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > col</strong>  

</summary>
<blockquote>

**Title:** Col

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                   |
| -------------------------------------------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col_anyOf_i0"></a>7.1.1.7.1.1.1.1.5.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > col > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col_anyOf_i1"></a>7.1.1.7.1.1.1.1.5.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > col > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col-header"></a>7.1.1.7.1.1.1.1.6. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > col-header</strong>  

</summary>
<blockquote>

**Title:** Col-Header

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col-span"></a>7.1.1.7.1.1.1.1.7. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > col-span</strong>  

</summary>
<blockquote>

**Title:** Col-Span

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                        |
| ------------------------------------------------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col-span_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col-span_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col-span_anyOf_i0"></a>7.1.1.7.1.1.1.1.7.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > col-span > anyOf > item 0`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | No                 |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                                   | Description |
| ------------------------------------------------------------------------------------------------- | ----------- |
| [item 0 items](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col-span_anyOf_i0_items) | -           |

###### <a name="autogenerated_heading_41"></a>7.1.1.7.1.1.1.1.7.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > col-span > anyOf > item 0 > item 0 items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_col-span_anyOf_i1"></a>7.1.1.7.1.1.1.1.7.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > col-span > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row"></a>7.1.1.7.1.1.1.1.8. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > row</strong>  

</summary>
<blockquote>

**Title:** Row

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                   |
| -------------------------------------------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row_anyOf_i0"></a>7.1.1.7.1.1.1.1.8.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > row > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row_anyOf_i1"></a>7.1.1.7.1.1.1.1.8.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > row > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row-header"></a>7.1.1.7.1.1.1.1.9. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > row-header</strong>  

</summary>
<blockquote>

**Title:** Row-Header

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row-span"></a>7.1.1.7.1.1.1.1.10. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > row-span</strong>  

</summary>
<blockquote>

**Title:** Row-Span

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                        |
| ------------------------------------------------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row-span_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row-span_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row-span_anyOf_i0"></a>7.1.1.7.1.1.1.1.10.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > row-span > anyOf > item 0`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | No                 |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                                   | Description |
| ------------------------------------------------------------------------------------------------- | ----------- |
| [item 0 items](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row-span_anyOf_i0_items) | -           |

###### <a name="autogenerated_heading_42"></a>7.1.1.7.1.1.1.1.10.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > row-span > anyOf > item 0 > item 0 items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i0_row-span_anyOf_i1"></a>7.1.1.7.1.1.1.1.10.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > GlmTableCell > row-span > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1"></a>7.1.1.7.1.1.1.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/TableCell                                                         |

**Description:** Table cell.

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_bbox"></a>7.1.1.7.1.1.1.2.1. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell > bbox</strong>  

</summary>
<blockquote>

**Title:** Bbox

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                    |
| --------------------------------------------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_bbox_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_bbox_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_bbox_anyOf_i0"></a>7.1.1.7.1.1.1.2.1.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell > bbox > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                               | Description |
| --------------------------------------------------------------------------------------------- | ----------- |
| [item 0 items](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_bbox_anyOf_i0_items) | -           |

###### <a name="autogenerated_heading_43"></a>7.1.1.7.1.1.1.2.1.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell > bbox > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_bbox_anyOf_i1"></a>7.1.1.7.1.1.1.2.1.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell > bbox > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_spans"></a>7.1.1.7.1.1.1.2.2. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell > spans</strong>  

</summary>
<blockquote>

**Title:** Spans

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                     |
| ---------------------------------------------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_spans_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_spans_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_spans_anyOf_i0"></a>7.1.1.7.1.1.1.2.2.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell > spans > anyOf > item 0`

|              |                  |
| ------------ | ---------------- |
| **Type**     | `array of array` |
| **Required** | No               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                                | Description |
| ---------------------------------------------------------------------------------------------- | ----------- |
| [item 0 items](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_spans_anyOf_i0_items) | -           |

###### <a name="autogenerated_heading_44"></a>7.1.1.7.1.1.1.2.2.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell > spans > anyOf > item 0 > item 0 items

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | No                 |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                                            | Description |
| ---------------------------------------------------------------------------------------------------------- | ----------- |
| [item 0 items items](#tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_spans_anyOf_i0_items_items) | -           |

###### <a name="autogenerated_heading_45"></a>7.1.1.7.1.1.1.2.2.1.1.1. ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell > spans > anyOf > item 0 > item 0 items > item 0 items items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_spans_anyOf_i1"></a>7.1.1.7.1.1.1.2.2.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell > spans > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_text"></a>7.1.1.7.1.1.1.2.3. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell > text</strong>  

</summary>
<blockquote>

**Title:** Text

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_data_anyOf_i0_items_items_anyOf_i1_type"></a>7.1.1.7.1.1.1.2.4. [Required] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 0 > item 0 items > item 0 items items > anyOf > TableCell > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

</blockquote>

</blockquote>

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_data_anyOf_i1"></a>7.1.1.7.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > data > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="tables_anyOf_i0_items_model"></a>7.1.1.8. [Optional] Property ExportedCCSDocument > tables > anyOf > item 0 > Table > model</strong>  

</summary>
<blockquote>

**Title:** Model

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                  |
| ----------------------------------------------- |
| [item 0](#tables_anyOf_i0_items_model_anyOf_i0) |
| [item 1](#tables_anyOf_i0_items_model_anyOf_i1) |

<blockquote>

###### <a name="tables_anyOf_i0_items_model_anyOf_i0"></a>7.1.1.8.1. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > model > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="tables_anyOf_i0_items_model_anyOf_i1"></a>7.1.1.8.2. Property `ExportedCCSDocument > tables > anyOf > item 0 > Table > model > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

### <a name="tables_anyOf_i1"></a>7.2. Property `ExportedCCSDocument > tables > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="bitmaps"></a>8. [Optional] Property ExportedCCSDocument > bitmaps</strong>  

</summary>
<blockquote>

**Title:** Bitmaps

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)              |
| --------------------------- |
| [item 0](#bitmaps_anyOf_i0) |
| [item 1](#bitmaps_anyOf_i1) |

<blockquote>

### <a name="bitmaps_anyOf_i0"></a>8.1. Property `ExportedCCSDocument > bitmaps > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be         | Description    |
| --------------------------------------- | -------------- |
| [BitmapObject](#bitmaps_anyOf_i0_items) | Bitmap object. |

#### <a name="autogenerated_heading_46"></a>8.1.1. ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BitmapObject                                                      |

**Description:** Bitmap object.

<details>
<summary>
<strong> <a name="bitmaps_anyOf_i0_items_type"></a>8.1.1.1. [Required] Property ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="bitmaps_anyOf_i0_items_bounding_box"></a>8.1.1.2. [Required] Property ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > bounding_box</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | Yes                                                                       |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            |                                                                           |

**Description:** Bounding box container.

<details>
<summary>
<strong> <a name="bitmaps_anyOf_i0_items_bounding_box_min"></a>8.1.1.2.1. [Required] Property ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > bounding_box > min</strong>  

</summary>
<blockquote>

**Title:** Min

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                             | Description |
| ----------------------------------------------------------- | ----------- |
| [min items](#bitmaps_anyOf_i0_items_bounding_box_min_items) | -           |

###### <a name="autogenerated_heading_47"></a>8.1.1.2.1.1. ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > bounding_box > min > min items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="bitmaps_anyOf_i0_items_bounding_box_max"></a>8.1.1.2.2. [Required] Property ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > bounding_box > max</strong>  

</summary>
<blockquote>

**Title:** Max

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                             | Description |
| ----------------------------------------------------------- | ----------- |
| [max items](#bitmaps_anyOf_i0_items_bounding_box_max_items) | -           |

###### <a name="autogenerated_heading_48"></a>8.1.1.2.2.1. ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > bounding_box > max > max items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="bitmaps_anyOf_i0_items_prov"></a>8.1.1.3. [Required] Property ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > prov</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | Yes                                                                       |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/Prov                                                              |

**Description:** Provenance.

<details>
<summary>
<strong> <a name="bitmaps_anyOf_i0_items_prov_bbox"></a>8.1.1.3.1. [Required] Property ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > prov > bbox</strong>  

</summary>
<blockquote>

**Title:** Bbox

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                       | Description |
| ----------------------------------------------------- | ----------- |
| [bbox items](#bitmaps_anyOf_i0_items_prov_bbox_items) | -           |

###### <a name="autogenerated_heading_49"></a>8.1.1.3.1.1. ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > prov > bbox > bbox items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="bitmaps_anyOf_i0_items_prov_page"></a>8.1.1.3.2. [Required] Property ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > prov > page</strong>  

</summary>
<blockquote>

**Title:** Page

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="bitmaps_anyOf_i0_items_prov_span"></a>8.1.1.3.3. [Required] Property ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > prov > span</strong>  

</summary>
<blockquote>

**Title:** Span

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | Yes                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                       | Description |
| ----------------------------------------------------- | ----------- |
| [span items](#bitmaps_anyOf_i0_items_prov_span_items) | -           |

###### <a name="autogenerated_heading_50"></a>8.1.1.3.3.1. ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > prov > span > span items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="bitmaps_anyOf_i0_items_prov___ref_s3_data"></a>8.1.1.3.4. [Optional] Property ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > prov > __ref_s3_data</strong>  

</summary>
<blockquote>

**Title:**   Ref S3 Data

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                |
| ------------------------------------------------------------- |
| [item 0](#bitmaps_anyOf_i0_items_prov___ref_s3_data_anyOf_i0) |
| [item 1](#bitmaps_anyOf_i0_items_prov___ref_s3_data_anyOf_i1) |

<blockquote>

###### <a name="bitmaps_anyOf_i0_items_prov___ref_s3_data_anyOf_i0"></a>8.1.1.3.4.1. Property `ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > prov > __ref_s3_data > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="bitmaps_anyOf_i0_items_prov___ref_s3_data_anyOf_i1"></a>8.1.1.3.4.2. Property `ExportedCCSDocument > bitmaps > anyOf > item 0 > BitmapObject > prov > __ref_s3_data > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
</details>

</blockquote>
<blockquote>

### <a name="bitmaps_anyOf_i1"></a>8.2. Property `ExportedCCSDocument > bitmaps > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="equations"></a>9. [Optional] Property ExportedCCSDocument > equations</strong>  

</summary>
<blockquote>

**Title:** Equations

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                |
| ----------------------------- |
| [item 0](#equations_anyOf_i0) |
| [item 1](#equations_anyOf_i1) |

<blockquote>

### <a name="equations_anyOf_i0"></a>9.1. Property `ExportedCCSDocument > equations > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be       | Description |
| ------------------------------------- | ----------- |
| [BaseCell](#equations_anyOf_i0_items) | Base cell.  |

#### <a name="autogenerated_heading_51"></a>9.1.1. ExportedCCSDocument > equations > anyOf > item 0 > BaseCell

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BaseCell                                                          |

**Description:** Base cell.

<details>
<summary>
<strong> <a name="equations_anyOf_i0_items_bounding-box"></a>9.1.1.1. [Optional] Property ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > bounding-box</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                          |
| ----------------------------------------------------------------------- |
| [BoundingBoxContainer](#equations_anyOf_i0_items_bounding-box_anyOf_i0) |
| [item 1](#equations_anyOf_i0_items_bounding-box_anyOf_i1)               |

<blockquote>

###### <a name="equations_anyOf_i0_items_bounding-box_anyOf_i0"></a>9.1.1.1.1. Property `ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > bounding-box > anyOf > BoundingBoxContainer`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BoundingBoxContainer                                              |

**Description:** Bounding box container.

<details>
<summary>
<strong> <a name="equations_anyOf_i0_items_bounding-box_anyOf_i0_min"></a>9.1.1.1.1.1. [Required] Property ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > bounding-box > anyOf > BoundingBoxContainer > min</strong>  

</summary>
<blockquote>

**Title:** Min

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                        | Description |
| ---------------------------------------------------------------------- | ----------- |
| [min items](#equations_anyOf_i0_items_bounding-box_anyOf_i0_min_items) | -           |

###### <a name="autogenerated_heading_52"></a>9.1.1.1.1.1.1. ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > bounding-box > anyOf > BoundingBoxContainer > min > min items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="equations_anyOf_i0_items_bounding-box_anyOf_i0_max"></a>9.1.1.1.1.2. [Required] Property ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > bounding-box > anyOf > BoundingBoxContainer > max</strong>  

</summary>
<blockquote>

**Title:** Max

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                        | Description |
| ---------------------------------------------------------------------- | ----------- |
| [max items](#equations_anyOf_i0_items_bounding-box_anyOf_i0_max_items) | -           |

###### <a name="autogenerated_heading_53"></a>9.1.1.1.1.2.1. ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > bounding-box > anyOf > BoundingBoxContainer > max > max items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="equations_anyOf_i0_items_bounding-box_anyOf_i1"></a>9.1.1.1.2. Property `ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > bounding-box > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="equations_anyOf_i0_items_prov"></a>9.1.1.2. [Optional] Property ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov</strong>  

</summary>
<blockquote>

**Title:** Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                    |
| ------------------------------------------------- |
| [item 0](#equations_anyOf_i0_items_prov_anyOf_i0) |
| [item 1](#equations_anyOf_i0_items_prov_anyOf_i1) |

<blockquote>

###### <a name="equations_anyOf_i0_items_prov_anyOf_i0"></a>9.1.1.2.1. Property `ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                       | Description |
| ----------------------------------------------------- | ----------- |
| [Prov](#equations_anyOf_i0_items_prov_anyOf_i0_items) | Provenance. |

###### <a name="autogenerated_heading_54"></a>9.1.1.2.1.1. ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/Prov                                                              |

**Description:** Provenance.

<details>
<summary>
<strong> <a name="equations_anyOf_i0_items_prov_anyOf_i0_items_bbox"></a>9.1.1.2.1.1.1. [Required] Property ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > bbox</strong>  

</summary>
<blockquote>

**Title:** Bbox

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                        | Description |
| ---------------------------------------------------------------------- | ----------- |
| [bbox items](#equations_anyOf_i0_items_prov_anyOf_i0_items_bbox_items) | -           |

###### <a name="autogenerated_heading_55"></a>9.1.1.2.1.1.1.1. ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > bbox > bbox items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="equations_anyOf_i0_items_prov_anyOf_i0_items_page"></a>9.1.1.2.1.1.2. [Required] Property ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > page</strong>  

</summary>
<blockquote>

**Title:** Page

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="equations_anyOf_i0_items_prov_anyOf_i0_items_span"></a>9.1.1.2.1.1.3. [Required] Property ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > span</strong>  

</summary>
<blockquote>

**Title:** Span

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | Yes                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                        | Description |
| ---------------------------------------------------------------------- | ----------- |
| [span items](#equations_anyOf_i0_items_prov_anyOf_i0_items_span_items) | -           |

###### <a name="autogenerated_heading_56"></a>9.1.1.2.1.1.3.1. ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > span > span items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="equations_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data"></a>9.1.1.2.1.1.4. [Optional] Property ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > __ref_s3_data</strong>  

</summary>
<blockquote>

**Title:**   Ref S3 Data

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                 |
| ------------------------------------------------------------------------------ |
| [item 0](#equations_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0) |
| [item 1](#equations_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1) |

<blockquote>

###### <a name="equations_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0"></a>9.1.1.2.1.1.4.1. Property `ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="equations_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1"></a>9.1.1.2.1.1.4.2. Property `ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="equations_anyOf_i0_items_prov_anyOf_i1"></a>9.1.1.2.2. Property `ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > prov > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="equations_anyOf_i0_items_text"></a>9.1.1.3. [Optional] Property ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > text</strong>  

</summary>
<blockquote>

**Title:** Text

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                    |
| ------------------------------------------------- |
| [item 0](#equations_anyOf_i0_items_text_anyOf_i0) |
| [item 1](#equations_anyOf_i0_items_text_anyOf_i1) |

<blockquote>

###### <a name="equations_anyOf_i0_items_text_anyOf_i0"></a>9.1.1.3.1. Property `ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > text > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="equations_anyOf_i0_items_text_anyOf_i1"></a>9.1.1.3.2. Property `ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > text > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="equations_anyOf_i0_items_type"></a>9.1.1.4. [Required] Property ExportedCCSDocument > equations > anyOf > item 0 > BaseCell > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

</blockquote>
<blockquote>

### <a name="equations_anyOf_i1"></a>9.2. Property `ExportedCCSDocument > equations > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="footnotes"></a>10. [Optional] Property ExportedCCSDocument > footnotes</strong>  

</summary>
<blockquote>

**Title:** Footnotes

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                |
| ----------------------------- |
| [item 0](#footnotes_anyOf_i0) |
| [item 1](#footnotes_anyOf_i1) |

<blockquote>

### <a name="footnotes_anyOf_i0"></a>10.1. Property `ExportedCCSDocument > footnotes > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be       | Description                  |
| ------------------------------------- | ---------------------------- |
| [BaseText](#footnotes_anyOf_i0_items) | Base model for text objects. |

#### <a name="autogenerated_heading_57"></a>10.1.1. ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BaseText                                                          |

**Description:** Base model for text objects.

<details>
<summary>
<strong> <a name="footnotes_anyOf_i0_items_text"></a>10.1.1.1. [Required] Property ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > text</strong>  

</summary>
<blockquote>

**Title:** Text

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="footnotes_anyOf_i0_items_type"></a>10.1.1.2. [Required] Property ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="footnotes_anyOf_i0_items_name"></a>10.1.1.3. [Optional] Property ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > name</strong>  

</summary>
<blockquote>

**Title:** Name

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                    |
| ------------------------------------------------- |
| [item 0](#footnotes_anyOf_i0_items_name_anyOf_i0) |
| [item 1](#footnotes_anyOf_i0_items_name_anyOf_i1) |

<blockquote>

###### <a name="footnotes_anyOf_i0_items_name_anyOf_i0"></a>10.1.1.3.1. Property `ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > name > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="footnotes_anyOf_i0_items_name_anyOf_i1"></a>10.1.1.3.2. Property `ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > name > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="footnotes_anyOf_i0_items_font"></a>10.1.1.4. [Optional] Property ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > font</strong>  

</summary>
<blockquote>

**Title:** Font

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                    |
| ------------------------------------------------- |
| [item 0](#footnotes_anyOf_i0_items_font_anyOf_i0) |
| [item 1](#footnotes_anyOf_i0_items_font_anyOf_i1) |

<blockquote>

###### <a name="footnotes_anyOf_i0_items_font_anyOf_i0"></a>10.1.1.4.1. Property `ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > font > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="footnotes_anyOf_i0_items_font_anyOf_i1"></a>10.1.1.4.2. Property `ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > font > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="footnotes_anyOf_i0_items_prov"></a>10.1.1.5. [Optional] Property ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov</strong>  

</summary>
<blockquote>

**Title:** Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                    |
| ------------------------------------------------- |
| [item 0](#footnotes_anyOf_i0_items_prov_anyOf_i0) |
| [item 1](#footnotes_anyOf_i0_items_prov_anyOf_i1) |

<blockquote>

###### <a name="footnotes_anyOf_i0_items_prov_anyOf_i0"></a>10.1.1.5.1. Property `ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                       | Description |
| ----------------------------------------------------- | ----------- |
| [Prov](#footnotes_anyOf_i0_items_prov_anyOf_i0_items) | Provenance. |

###### <a name="autogenerated_heading_58"></a>10.1.1.5.1.1. ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/Prov                                                              |

**Description:** Provenance.

<details>
<summary>
<strong> <a name="footnotes_anyOf_i0_items_prov_anyOf_i0_items_bbox"></a>10.1.1.5.1.1.1. [Required] Property ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > bbox</strong>  

</summary>
<blockquote>

**Title:** Bbox

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                        | Description |
| ---------------------------------------------------------------------- | ----------- |
| [bbox items](#footnotes_anyOf_i0_items_prov_anyOf_i0_items_bbox_items) | -           |

###### <a name="autogenerated_heading_59"></a>10.1.1.5.1.1.1.1. ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > bbox > bbox items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="footnotes_anyOf_i0_items_prov_anyOf_i0_items_page"></a>10.1.1.5.1.1.2. [Required] Property ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > page</strong>  

</summary>
<blockquote>

**Title:** Page

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="footnotes_anyOf_i0_items_prov_anyOf_i0_items_span"></a>10.1.1.5.1.1.3. [Required] Property ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > span</strong>  

</summary>
<blockquote>

**Title:** Span

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | Yes                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                        | Description |
| ---------------------------------------------------------------------- | ----------- |
| [span items](#footnotes_anyOf_i0_items_prov_anyOf_i0_items_span_items) | -           |

###### <a name="autogenerated_heading_60"></a>10.1.1.5.1.1.3.1. ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > span > span items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="footnotes_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data"></a>10.1.1.5.1.1.4. [Optional] Property ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data</strong>  

</summary>
<blockquote>

**Title:**   Ref S3 Data

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                 |
| ------------------------------------------------------------------------------ |
| [item 0](#footnotes_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0) |
| [item 1](#footnotes_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1) |

<blockquote>

###### <a name="footnotes_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0"></a>10.1.1.5.1.1.4.1. Property `ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="footnotes_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1"></a>10.1.1.5.1.1.4.2. Property `ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="footnotes_anyOf_i0_items_prov_anyOf_i1"></a>10.1.1.5.2. Property `ExportedCCSDocument > footnotes > anyOf > item 0 > BaseText > prov > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

### <a name="footnotes_anyOf_i1"></a>10.2. Property `ExportedCCSDocument > footnotes > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-dimensions"></a>11. [Optional] Property ExportedCCSDocument > page-dimensions</strong>  

</summary>
<blockquote>

**Title:** Page-Dimensions

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#page-dimensions_anyOf_i0) |
| [item 1](#page-dimensions_anyOf_i1) |

<blockquote>

### <a name="page-dimensions_anyOf_i0"></a>11.1. Property `ExportedCCSDocument > page-dimensions > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                   | Description      |
| ------------------------------------------------- | ---------------- |
| [PageDimensions](#page-dimensions_anyOf_i0_items) | Page dimensions. |

#### <a name="autogenerated_heading_61"></a>11.1.1. ExportedCCSDocument > page-dimensions > anyOf > item 0 > PageDimensions

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/PageDimensions                                                    |

**Description:** Page dimensions.

<details>
<summary>
<strong> <a name="page-dimensions_anyOf_i0_items_height"></a>11.1.1.1. [Required] Property ExportedCCSDocument > page-dimensions > anyOf > item 0 > PageDimensions > height</strong>  

</summary>
<blockquote>

**Title:** Height

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-dimensions_anyOf_i0_items_page"></a>11.1.1.2. [Required] Property ExportedCCSDocument > page-dimensions > anyOf > item 0 > PageDimensions > page</strong>  

</summary>
<blockquote>

**Title:** Page

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-dimensions_anyOf_i0_items_width"></a>11.1.1.3. [Required] Property ExportedCCSDocument > page-dimensions > anyOf > item 0 > PageDimensions > width</strong>  

</summary>
<blockquote>

**Title:** Width

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

</blockquote>
</details>

</blockquote>
<blockquote>

### <a name="page-dimensions_anyOf_i1"></a>11.2. Property `ExportedCCSDocument > page-dimensions > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-footers"></a>12. [Optional] Property ExportedCCSDocument > page-footers</strong>  

</summary>
<blockquote>

**Title:** Page-Footers

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                   |
| -------------------------------- |
| [item 0](#page-footers_anyOf_i0) |
| [item 1](#page-footers_anyOf_i1) |

<blockquote>

### <a name="page-footers_anyOf_i0"></a>12.1. Property `ExportedCCSDocument > page-footers > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be          | Description                  |
| ---------------------------------------- | ---------------------------- |
| [BaseText](#page-footers_anyOf_i0_items) | Base model for text objects. |

#### <a name="autogenerated_heading_62"></a>12.1.1. ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BaseText                                                          |

**Description:** Base model for text objects.

<details>
<summary>
<strong> <a name="page-footers_anyOf_i0_items_text"></a>12.1.1.1. [Required] Property ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > text</strong>  

</summary>
<blockquote>

**Title:** Text

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-footers_anyOf_i0_items_type"></a>12.1.1.2. [Required] Property ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-footers_anyOf_i0_items_name"></a>12.1.1.3. [Optional] Property ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > name</strong>  

</summary>
<blockquote>

**Title:** Name

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                       |
| ---------------------------------------------------- |
| [item 0](#page-footers_anyOf_i0_items_name_anyOf_i0) |
| [item 1](#page-footers_anyOf_i0_items_name_anyOf_i1) |

<blockquote>

###### <a name="page-footers_anyOf_i0_items_name_anyOf_i0"></a>12.1.1.3.1. Property `ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > name > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="page-footers_anyOf_i0_items_name_anyOf_i1"></a>12.1.1.3.2. Property `ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > name > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-footers_anyOf_i0_items_font"></a>12.1.1.4. [Optional] Property ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > font</strong>  

</summary>
<blockquote>

**Title:** Font

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                       |
| ---------------------------------------------------- |
| [item 0](#page-footers_anyOf_i0_items_font_anyOf_i0) |
| [item 1](#page-footers_anyOf_i0_items_font_anyOf_i1) |

<blockquote>

###### <a name="page-footers_anyOf_i0_items_font_anyOf_i0"></a>12.1.1.4.1. Property `ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > font > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="page-footers_anyOf_i0_items_font_anyOf_i1"></a>12.1.1.4.2. Property `ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > font > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-footers_anyOf_i0_items_prov"></a>12.1.1.5. [Optional] Property ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov</strong>  

</summary>
<blockquote>

**Title:** Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                       |
| ---------------------------------------------------- |
| [item 0](#page-footers_anyOf_i0_items_prov_anyOf_i0) |
| [item 1](#page-footers_anyOf_i0_items_prov_anyOf_i1) |

<blockquote>

###### <a name="page-footers_anyOf_i0_items_prov_anyOf_i0"></a>12.1.1.5.1. Property `ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                          | Description |
| -------------------------------------------------------- | ----------- |
| [Prov](#page-footers_anyOf_i0_items_prov_anyOf_i0_items) | Provenance. |

###### <a name="autogenerated_heading_63"></a>12.1.1.5.1.1. ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/Prov                                                              |

**Description:** Provenance.

<details>
<summary>
<strong> <a name="page-footers_anyOf_i0_items_prov_anyOf_i0_items_bbox"></a>12.1.1.5.1.1.1. [Required] Property ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > bbox</strong>  

</summary>
<blockquote>

**Title:** Bbox

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                           | Description |
| ------------------------------------------------------------------------- | ----------- |
| [bbox items](#page-footers_anyOf_i0_items_prov_anyOf_i0_items_bbox_items) | -           |

###### <a name="autogenerated_heading_64"></a>12.1.1.5.1.1.1.1. ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > bbox > bbox items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-footers_anyOf_i0_items_prov_anyOf_i0_items_page"></a>12.1.1.5.1.1.2. [Required] Property ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > page</strong>  

</summary>
<blockquote>

**Title:** Page

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-footers_anyOf_i0_items_prov_anyOf_i0_items_span"></a>12.1.1.5.1.1.3. [Required] Property ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > span</strong>  

</summary>
<blockquote>

**Title:** Span

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | Yes                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                           | Description |
| ------------------------------------------------------------------------- | ----------- |
| [span items](#page-footers_anyOf_i0_items_prov_anyOf_i0_items_span_items) | -           |

###### <a name="autogenerated_heading_65"></a>12.1.1.5.1.1.3.1. ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > span > span items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-footers_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data"></a>12.1.1.5.1.1.4. [Optional] Property ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data</strong>  

</summary>
<blockquote>

**Title:**   Ref S3 Data

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                    |
| --------------------------------------------------------------------------------- |
| [item 0](#page-footers_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0) |
| [item 1](#page-footers_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1) |

<blockquote>

###### <a name="page-footers_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0"></a>12.1.1.5.1.1.4.1. Property `ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="page-footers_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1"></a>12.1.1.5.1.1.4.2. Property `ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="page-footers_anyOf_i0_items_prov_anyOf_i1"></a>12.1.1.5.2. Property `ExportedCCSDocument > page-footers > anyOf > item 0 > BaseText > prov > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

### <a name="page-footers_anyOf_i1"></a>12.2. Property `ExportedCCSDocument > page-footers > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-headers"></a>13. [Optional] Property ExportedCCSDocument > page-headers</strong>  

</summary>
<blockquote>

**Title:** Page-Headers

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                   |
| -------------------------------- |
| [item 0](#page-headers_anyOf_i0) |
| [item 1](#page-headers_anyOf_i1) |

<blockquote>

### <a name="page-headers_anyOf_i0"></a>13.1. Property `ExportedCCSDocument > page-headers > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be          | Description                  |
| ---------------------------------------- | ---------------------------- |
| [BaseText](#page-headers_anyOf_i0_items) | Base model for text objects. |

#### <a name="autogenerated_heading_66"></a>13.1.1. ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/BaseText                                                          |

**Description:** Base model for text objects.

<details>
<summary>
<strong> <a name="page-headers_anyOf_i0_items_text"></a>13.1.1.1. [Required] Property ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > text</strong>  

</summary>
<blockquote>

**Title:** Text

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-headers_anyOf_i0_items_type"></a>13.1.1.2. [Required] Property ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-headers_anyOf_i0_items_name"></a>13.1.1.3. [Optional] Property ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > name</strong>  

</summary>
<blockquote>

**Title:** Name

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                       |
| ---------------------------------------------------- |
| [item 0](#page-headers_anyOf_i0_items_name_anyOf_i0) |
| [item 1](#page-headers_anyOf_i0_items_name_anyOf_i1) |

<blockquote>

###### <a name="page-headers_anyOf_i0_items_name_anyOf_i0"></a>13.1.1.3.1. Property `ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > name > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="page-headers_anyOf_i0_items_name_anyOf_i1"></a>13.1.1.3.2. Property `ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > name > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-headers_anyOf_i0_items_font"></a>13.1.1.4. [Optional] Property ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > font</strong>  

</summary>
<blockquote>

**Title:** Font

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                       |
| ---------------------------------------------------- |
| [item 0](#page-headers_anyOf_i0_items_font_anyOf_i0) |
| [item 1](#page-headers_anyOf_i0_items_font_anyOf_i1) |

<blockquote>

###### <a name="page-headers_anyOf_i0_items_font_anyOf_i0"></a>13.1.1.4.1. Property `ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > font > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="page-headers_anyOf_i0_items_font_anyOf_i1"></a>13.1.1.4.2. Property `ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > font > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-headers_anyOf_i0_items_prov"></a>13.1.1.5. [Optional] Property ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov</strong>  

</summary>
<blockquote>

**Title:** Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                       |
| ---------------------------------------------------- |
| [item 0](#page-headers_anyOf_i0_items_prov_anyOf_i0) |
| [item 1](#page-headers_anyOf_i0_items_prov_anyOf_i1) |

<blockquote>

###### <a name="page-headers_anyOf_i0_items_prov_anyOf_i0"></a>13.1.1.5.1. Property `ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                          | Description |
| -------------------------------------------------------- | ----------- |
| [Prov](#page-headers_anyOf_i0_items_prov_anyOf_i0_items) | Provenance. |

###### <a name="autogenerated_heading_67"></a>13.1.1.5.1.1. ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/Prov                                                              |

**Description:** Provenance.

<details>
<summary>
<strong> <a name="page-headers_anyOf_i0_items_prov_anyOf_i0_items_bbox"></a>13.1.1.5.1.1.1. [Required] Property ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > bbox</strong>  

</summary>
<blockquote>

**Title:** Bbox

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 4                  |
| **Max items**        | 4                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                           | Description |
| ------------------------------------------------------------------------- | ----------- |
| [bbox items](#page-headers_anyOf_i0_items_prov_anyOf_i0_items_bbox_items) | -           |

###### <a name="autogenerated_heading_68"></a>13.1.1.5.1.1.1.1. ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > bbox > bbox items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-headers_anyOf_i0_items_prov_anyOf_i0_items_page"></a>13.1.1.5.1.1.2. [Required] Property ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > page</strong>  

</summary>
<blockquote>

**Title:** Page

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-headers_anyOf_i0_items_prov_anyOf_i0_items_span"></a>13.1.1.5.1.1.3. [Required] Property ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > span</strong>  

</summary>
<blockquote>

**Title:** Span

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | Yes                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                           | Description |
| ------------------------------------------------------------------------- | ----------- |
| [span items](#page-headers_anyOf_i0_items_prov_anyOf_i0_items_span_items) | -           |

###### <a name="autogenerated_heading_69"></a>13.1.1.5.1.1.3.1. ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > span > span items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="page-headers_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data"></a>13.1.1.5.1.1.4. [Optional] Property ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data</strong>  

</summary>
<blockquote>

**Title:**   Ref S3 Data

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                                    |
| --------------------------------------------------------------------------------- |
| [item 0](#page-headers_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0) |
| [item 1](#page-headers_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1) |

<blockquote>

###### <a name="page-headers_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i0"></a>13.1.1.5.1.1.4.1. Property `ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

###### <a name="page-headers_anyOf_i0_items_prov_anyOf_i0_items___ref_s3_data_anyOf_i1"></a>13.1.1.5.1.1.4.2. Property `ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov > anyOf > item 0 > Prov > __ref_s3_data > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

###### <a name="page-headers_anyOf_i0_items_prov_anyOf_i1"></a>13.1.1.5.2. Property `ExportedCCSDocument > page-headers > anyOf > item 0 > BaseText > prov > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

### <a name="page-headers_anyOf_i1"></a>13.2. Property `ExportedCCSDocument > page-headers > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data"></a>14. [Optional] Property ExportedCCSDocument > _s3_data</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)              |
| --------------------------- |
| [S3Data](#s3_data_anyOf_i0) |
| [item 1](#s3_data_anyOf_i1) |

<blockquote>

### <a name="s3_data_anyOf_i0"></a>14.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/S3Data                                                            |

**Description:** Data object in a cloud object storage.

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-document"></a>14.1.1. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-document</strong>  

</summary>
<blockquote>

**Title:** Pdf-Document

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                    |
| ------------------------------------------------- |
| [item 0](#s3_data_anyOf_i0_pdf-document_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_pdf-document_anyOf_i1) |

<blockquote>

##### <a name="s3_data_anyOf_i0_pdf-document_anyOf_i0"></a>14.1.1.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-document > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                             | Description                         |
| ----------------------------------------------------------- | ----------------------------------- |
| [S3Resource](#s3_data_anyOf_i0_pdf-document_anyOf_i0_items) | Resource in a cloud object storage. |

###### <a name="autogenerated_heading_70"></a>14.1.1.1.1. ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-document > anyOf > item 0 > S3Resource

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/S3Resource                                                        |

**Description:** Resource in a cloud object storage.

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-document_anyOf_i0_items_mime"></a>14.1.1.1.1.1. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-document > anyOf > item 0 > S3Resource > mime</strong>  

</summary>
<blockquote>

**Title:** Mime

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-document_anyOf_i0_items_path"></a>14.1.1.1.1.2. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-document > anyOf > item 0 > S3Resource > path</strong>  

</summary>
<blockquote>

**Title:** Path

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-document_anyOf_i0_items_page"></a>14.1.1.1.1.3. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-document > anyOf > item 0 > S3Resource > page</strong>  

</summary>
<blockquote>

**Title:** Page

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                        |
| --------------------------------------------------------------------- |
| [item 0](#s3_data_anyOf_i0_pdf-document_anyOf_i0_items_page_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_pdf-document_anyOf_i0_items_page_anyOf_i1) |

<blockquote>

###### <a name="s3_data_anyOf_i0_pdf-document_anyOf_i0_items_page_anyOf_i0"></a>14.1.1.1.1.3.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-document > anyOf > item 0 > S3Resource > page > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
<blockquote>

###### <a name="s3_data_anyOf_i0_pdf-document_anyOf_i0_items_page_anyOf_i1"></a>14.1.1.1.1.3.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-document > anyOf > item 0 > S3Resource > page > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

##### <a name="s3_data_anyOf_i0_pdf-document_anyOf_i1"></a>14.1.1.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-document > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-pages"></a>14.1.2. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-pages</strong>  

</summary>
<blockquote>

**Title:** Pdf-Pages

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                 |
| ---------------------------------------------- |
| [item 0](#s3_data_anyOf_i0_pdf-pages_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_pdf-pages_anyOf_i1) |

<blockquote>

##### <a name="s3_data_anyOf_i0_pdf-pages_anyOf_i0"></a>14.1.2.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-pages > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                          | Description                         |
| -------------------------------------------------------- | ----------------------------------- |
| [S3Resource](#s3_data_anyOf_i0_pdf-pages_anyOf_i0_items) | Resource in a cloud object storage. |

###### <a name="autogenerated_heading_71"></a>14.1.2.1.1. ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-pages > anyOf > item 0 > S3Resource

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/S3Resource                                                        |

**Description:** Resource in a cloud object storage.

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-pages_anyOf_i0_items_mime"></a>14.1.2.1.1.1. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-pages > anyOf > item 0 > S3Resource > mime</strong>  

</summary>
<blockquote>

**Title:** Mime

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-pages_anyOf_i0_items_path"></a>14.1.2.1.1.2. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-pages > anyOf > item 0 > S3Resource > path</strong>  

</summary>
<blockquote>

**Title:** Path

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-pages_anyOf_i0_items_page"></a>14.1.2.1.1.3. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-pages > anyOf > item 0 > S3Resource > page</strong>  

</summary>
<blockquote>

**Title:** Page

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                     |
| ------------------------------------------------------------------ |
| [item 0](#s3_data_anyOf_i0_pdf-pages_anyOf_i0_items_page_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_pdf-pages_anyOf_i0_items_page_anyOf_i1) |

<blockquote>

###### <a name="s3_data_anyOf_i0_pdf-pages_anyOf_i0_items_page_anyOf_i0"></a>14.1.2.1.1.3.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-pages > anyOf > item 0 > S3Resource > page > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
<blockquote>

###### <a name="s3_data_anyOf_i0_pdf-pages_anyOf_i0_items_page_anyOf_i1"></a>14.1.2.1.1.3.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-pages > anyOf > item 0 > S3Resource > page > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

##### <a name="s3_data_anyOf_i0_pdf-pages_anyOf_i1"></a>14.1.2.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-pages > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-images"></a>14.1.3. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-images</strong>  

</summary>
<blockquote>

**Title:** Pdf-Images

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                  |
| ----------------------------------------------- |
| [item 0](#s3_data_anyOf_i0_pdf-images_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_pdf-images_anyOf_i1) |

<blockquote>

##### <a name="s3_data_anyOf_i0_pdf-images_anyOf_i0"></a>14.1.3.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-images > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                           | Description                         |
| --------------------------------------------------------- | ----------------------------------- |
| [S3Resource](#s3_data_anyOf_i0_pdf-images_anyOf_i0_items) | Resource in a cloud object storage. |

###### <a name="autogenerated_heading_72"></a>14.1.3.1.1. ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-images > anyOf > item 0 > S3Resource

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/S3Resource                                                        |

**Description:** Resource in a cloud object storage.

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-images_anyOf_i0_items_mime"></a>14.1.3.1.1.1. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-images > anyOf > item 0 > S3Resource > mime</strong>  

</summary>
<blockquote>

**Title:** Mime

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-images_anyOf_i0_items_path"></a>14.1.3.1.1.2. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-images > anyOf > item 0 > S3Resource > path</strong>  

</summary>
<blockquote>

**Title:** Path

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_pdf-images_anyOf_i0_items_page"></a>14.1.3.1.1.3. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-images > anyOf > item 0 > S3Resource > page</strong>  

</summary>
<blockquote>

**Title:** Page

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                      |
| ------------------------------------------------------------------- |
| [item 0](#s3_data_anyOf_i0_pdf-images_anyOf_i0_items_page_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_pdf-images_anyOf_i0_items_page_anyOf_i1) |

<blockquote>

###### <a name="s3_data_anyOf_i0_pdf-images_anyOf_i0_items_page_anyOf_i0"></a>14.1.3.1.1.3.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-images > anyOf > item 0 > S3Resource > page > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
<blockquote>

###### <a name="s3_data_anyOf_i0_pdf-images_anyOf_i0_items_page_anyOf_i1"></a>14.1.3.1.1.3.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-images > anyOf > item 0 > S3Resource > page > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

##### <a name="s3_data_anyOf_i0_pdf-images_anyOf_i1"></a>14.1.3.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > pdf-images > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_json-document"></a>14.1.4. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > json-document</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                         |
| ------------------------------------------------------ |
| [S3Resource](#s3_data_anyOf_i0_json-document_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_json-document_anyOf_i1)     |

<blockquote>

##### <a name="s3_data_anyOf_i0_json-document_anyOf_i0"></a>14.1.4.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > json-document > anyOf > S3Resource`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/S3Resource                                                        |

**Description:** Resource in a cloud object storage.

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_json-document_anyOf_i0_mime"></a>14.1.4.1.1. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > json-document > anyOf > S3Resource > mime</strong>  

</summary>
<blockquote>

**Title:** Mime

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_json-document_anyOf_i0_path"></a>14.1.4.1.2. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > json-document > anyOf > S3Resource > path</strong>  

</summary>
<blockquote>

**Title:** Path

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_json-document_anyOf_i0_page"></a>14.1.4.1.3. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > json-document > anyOf > S3Resource > page</strong>  

</summary>
<blockquote>

**Title:** Page

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                   |
| ---------------------------------------------------------------- |
| [item 0](#s3_data_anyOf_i0_json-document_anyOf_i0_page_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_json-document_anyOf_i0_page_anyOf_i1) |

<blockquote>

###### <a name="s3_data_anyOf_i0_json-document_anyOf_i0_page_anyOf_i0"></a>14.1.4.1.3.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > json-document > anyOf > S3Resource > page > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
<blockquote>

###### <a name="s3_data_anyOf_i0_json-document_anyOf_i0_page_anyOf_i1"></a>14.1.4.1.3.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > json-document > anyOf > S3Resource > page > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

##### <a name="s3_data_anyOf_i0_json-document_anyOf_i1"></a>14.1.4.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > json-document > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_json-meta"></a>14.1.5. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > json-meta</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                     |
| -------------------------------------------------- |
| [S3Resource](#s3_data_anyOf_i0_json-meta_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_json-meta_anyOf_i1)     |

<blockquote>

##### <a name="s3_data_anyOf_i0_json-meta_anyOf_i0"></a>14.1.5.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > json-meta > anyOf > S3Resource`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/S3Resource                                                        |

**Description:** Resource in a cloud object storage.

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_json-meta_anyOf_i0_mime"></a>14.1.5.1.1. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > json-meta > anyOf > S3Resource > mime</strong>  

</summary>
<blockquote>

**Title:** Mime

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_json-meta_anyOf_i0_path"></a>14.1.5.1.2. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > json-meta > anyOf > S3Resource > path</strong>  

</summary>
<blockquote>

**Title:** Path

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_json-meta_anyOf_i0_page"></a>14.1.5.1.3. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > json-meta > anyOf > S3Resource > page</strong>  

</summary>
<blockquote>

**Title:** Page

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                               |
| ------------------------------------------------------------ |
| [item 0](#s3_data_anyOf_i0_json-meta_anyOf_i0_page_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_json-meta_anyOf_i0_page_anyOf_i1) |

<blockquote>

###### <a name="s3_data_anyOf_i0_json-meta_anyOf_i0_page_anyOf_i0"></a>14.1.5.1.3.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > json-meta > anyOf > S3Resource > page > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
<blockquote>

###### <a name="s3_data_anyOf_i0_json-meta_anyOf_i0_page_anyOf_i1"></a>14.1.5.1.3.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > json-meta > anyOf > S3Resource > page > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

##### <a name="s3_data_anyOf_i0_json-meta_anyOf_i1"></a>14.1.5.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > json-meta > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_glm-json-document"></a>14.1.6. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > glm-json-document</strong>  

</summary>
<blockquote>

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                             |
| ---------------------------------------------------------- |
| [S3Resource](#s3_data_anyOf_i0_glm-json-document_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_glm-json-document_anyOf_i1)     |

<blockquote>

##### <a name="s3_data_anyOf_i0_glm-json-document_anyOf_i0"></a>14.1.6.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > glm-json-document > anyOf > S3Resource`

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/S3Resource                                                        |

**Description:** Resource in a cloud object storage.

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_glm-json-document_anyOf_i0_mime"></a>14.1.6.1.1. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > glm-json-document > anyOf > S3Resource > mime</strong>  

</summary>
<blockquote>

**Title:** Mime

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_glm-json-document_anyOf_i0_path"></a>14.1.6.1.2. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > glm-json-document > anyOf > S3Resource > path</strong>  

</summary>
<blockquote>

**Title:** Path

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_glm-json-document_anyOf_i0_page"></a>14.1.6.1.3. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > glm-json-document > anyOf > S3Resource > page</strong>  

</summary>
<blockquote>

**Title:** Page

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                       |
| -------------------------------------------------------------------- |
| [item 0](#s3_data_anyOf_i0_glm-json-document_anyOf_i0_page_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_glm-json-document_anyOf_i0_page_anyOf_i1) |

<blockquote>

###### <a name="s3_data_anyOf_i0_glm-json-document_anyOf_i0_page_anyOf_i0"></a>14.1.6.1.3.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > glm-json-document > anyOf > S3Resource > page > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
<blockquote>

###### <a name="s3_data_anyOf_i0_glm-json-document_anyOf_i0_page_anyOf_i1"></a>14.1.6.1.3.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > glm-json-document > anyOf > S3Resource > page > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

##### <a name="s3_data_anyOf_i0_glm-json-document_anyOf_i1"></a>14.1.6.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > glm-json-document > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_figures"></a>14.1.7. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > figures</strong>  

</summary>
<blockquote>

**Title:** Figures

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                               |
| -------------------------------------------- |
| [item 0](#s3_data_anyOf_i0_figures_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_figures_anyOf_i1) |

<blockquote>

##### <a name="s3_data_anyOf_i0_figures_anyOf_i0"></a>14.1.7.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > figures > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                        | Description                         |
| ------------------------------------------------------ | ----------------------------------- |
| [S3Resource](#s3_data_anyOf_i0_figures_anyOf_i0_items) | Resource in a cloud object storage. |

###### <a name="autogenerated_heading_73"></a>14.1.7.1.1. ExportedCCSDocument > _s3_data > anyOf > S3Data > figures > anyOf > item 0 > S3Resource

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            | #/$defs/S3Resource                                                        |

**Description:** Resource in a cloud object storage.

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_figures_anyOf_i0_items_mime"></a>14.1.7.1.1.1. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > figures > anyOf > item 0 > S3Resource > mime</strong>  

</summary>
<blockquote>

**Title:** Mime

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_figures_anyOf_i0_items_path"></a>14.1.7.1.1.2. [Required] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > figures > anyOf > item 0 > S3Resource > path</strong>  

</summary>
<blockquote>

**Title:** Path

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

</blockquote>
</details>

<details>
<summary>
<strong> <a name="s3_data_anyOf_i0_figures_anyOf_i0_items_page"></a>14.1.7.1.1.3. [Optional] Property ExportedCCSDocument > _s3_data > anyOf > S3Data > figures > anyOf > item 0 > S3Resource > page</strong>  

</summary>
<blockquote>

**Title:** Page

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                                                   |
| ---------------------------------------------------------------- |
| [item 0](#s3_data_anyOf_i0_figures_anyOf_i0_items_page_anyOf_i0) |
| [item 1](#s3_data_anyOf_i0_figures_anyOf_i0_items_page_anyOf_i1) |

<blockquote>

###### <a name="s3_data_anyOf_i0_figures_anyOf_i0_items_page_anyOf_i0"></a>14.1.7.1.1.3.1. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > figures > anyOf > item 0 > S3Resource > page > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

</blockquote>
<blockquote>

###### <a name="s3_data_anyOf_i0_figures_anyOf_i0_items_page_anyOf_i1"></a>14.1.7.1.1.3.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > figures > anyOf > item 0 > S3Resource > page > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

##### <a name="s3_data_anyOf_i0_figures_anyOf_i1"></a>14.1.7.2. Property `ExportedCCSDocument > _s3_data > anyOf > S3Data > figures > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

</blockquote>
<blockquote>

### <a name="s3_data_anyOf_i1"></a>14.2. Property `ExportedCCSDocument > _s3_data > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

<details>
<summary>
<strong> <a name="identifiers"></a>15. [Optional] Property ExportedCCSDocument > identifiers</strong>  

</summary>
<blockquote>

**Title:** Identifiers

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

<blockquote>

| Any of(Option)                  |
| ------------------------------- |
| [item 0](#identifiers_anyOf_i0) |
| [item 1](#identifiers_anyOf_i1) |

<blockquote>

### <a name="identifiers_anyOf_i0"></a>15.1. Property `ExportedCCSDocument > identifiers > anyOf > item 0`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be           | Description                                 |
| ----------------------------------------- | ------------------------------------------- |
| [Identifier](#identifiers_anyOf_i0_items) | Unique identifier of a Docling data object. |

#### <a name="autogenerated_heading_74"></a>15.1.1. ExportedCCSDocument > identifiers > anyOf > item 0 > Identifier

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| **Type**                  | `object`                                                |
| **Required**              | No                                                      |
| **Additional properties** | [[Not allowed]](# "Additional Properties not allowed.") |
| **Defined in**            | #/$defs/Identifier                                      |

**Description:** Unique identifier of a Docling data object.

<details>
<summary>
<strong> <a name="identifiers_anyOf_i0_items_type"></a>15.1.1.1. [Required] Property ExportedCCSDocument > identifiers > anyOf > item 0 > Identifier > type</strong>  

</summary>
<blockquote>

**Title:** Type

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** A string representing a collection or database that contains this data object.

</blockquote>
</details>

<details>
<summary>
<strong> <a name="identifiers_anyOf_i0_items_value"></a>15.1.1.2. [Required] Property ExportedCCSDocument > identifiers > anyOf > item 0 > Identifier > value</strong>  

</summary>
<blockquote>

**Title:** Value

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** The identifier value of the data object within a collection or database.

</blockquote>
</details>

<details>
<summary>
<strong> <a name="identifiers_anyOf_i0_items__name"></a>15.1.1.3. [Required] Property ExportedCCSDocument > identifiers > anyOf > item 0 > Identifier > _name</strong>  

</summary>
<blockquote>

**Title:** _Name

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** A unique identifier of the data object across Docling, consisting of the concatenation of type and value in lower case, separated by hash (#).

| Restrictions                      |                                                                     |
| --------------------------------- | ------------------------------------------------------------------- |
| **Must match regular expression** | ```^.+#.+$``` [Test](https://regex101.com/?regex=%5E.%2B%23.%2B%24) |

</blockquote>
</details>

</blockquote>
<blockquote>

### <a name="identifiers_anyOf_i1"></a>15.2. Property `ExportedCCSDocument > identifiers > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

</blockquote>

</blockquote>

</blockquote>
</details>

----------------------------------------------------------------------------------------------------------------------------
