# Generic

**Title:** Generic

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |

**Description:** A representation of a generic document.

<details>
<summary>
<strong> <a name="name"></a>1. [Optional] Property Generic > _name</strong>  

</summary>
<blockquote>

**Title:**  Name

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `combining`                                                               |
| **Required**              | No                                                                        |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Default**               | `null`                                                                    |

**Description:** A short description or summary of the document.

<blockquote>

| Any of(Option)           |
| ------------------------ |
| [item 0](#name_anyOf_i0) |
| [item 1](#name_anyOf_i1) |

<blockquote>

### <a name="name_anyOf_i0"></a>1.1. Property `Generic > _name > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

### <a name="name_anyOf_i1"></a>1.2. Property `Generic > _name > anyOf > item 1`

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
<strong> <a name="file-info"></a>2. [Required] Property Generic > file-info</strong>  

</summary>
<blockquote>

**Title:** Document information

|                           |                                                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                  |
| **Required**              | Yes                                                                       |
| **Additional properties** | [[Any type: allowed]](# "Additional Properties of any type are allowed.") |
| **Defined in**            |                                                                           |

**Description:** Minimal identification information of the document within a collection.

<details>
<summary>
<strong> <a name="file-info_filename"></a>2.1. [Required] Property Generic > file-info > filename</strong>  

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
<strong> <a name="file-info_filename-prov"></a>2.2. [Optional] Property Generic > file-info > filename-prov</strong>  

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

#### <a name="file-info_filename-prov_anyOf_i0"></a>2.2.1. Property `Generic > file-info > filename-prov > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

</blockquote>
<blockquote>

#### <a name="file-info_filename-prov_anyOf_i1"></a>2.2.2. Property `Generic > file-info > filename-prov > anyOf > item 1`

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
<strong> <a name="file-info_document-hash"></a>2.3. [Required] Property Generic > file-info > document-hash</strong>  

</summary>
<blockquote>

**Title:** Document-Hash

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** A unique identifier of this data object within a collection of a Deep Search database

</blockquote>
</details>

</blockquote>
</details>

----------------------------------------------------------------------------------------------------------------------------
