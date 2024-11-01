## [v2.3.1](https://github.com/DS4SD/docling-core/releases/tag/v2.3.1) - 2024-11-01

### Fix

* Include titles to chunk heading metadata ([#62](https://github.com/DS4SD/docling-core/issues/62)) ([`bfeb2db`](https://github.com/DS4SD/docling-core/commit/bfeb2db24b70550693911af6aee01db8c74d464a))

## [v2.3.0](https://github.com/DS4SD/docling-core/releases/tag/v2.3.0) - 2024-10-29

### Feature

* Added pydantic models to store charts data (pie, bar, stacked bar, line, scatter) ([#52](https://github.com/DS4SD/docling-core/issues/52)) ([`36b7bea`](https://github.com/DS4SD/docling-core/commit/36b7bea53a291fdd8ffa5fc6cdbe4256d16c8710))

## [v2.2.3](https://github.com/DS4SD/docling-core/releases/tag/v2.2.3) - 2024-10-29

### Fix

* Str representation of enum across python versions ([#60](https://github.com/DS4SD/docling-core/issues/60)) ([`8528918`](https://github.com/DS4SD/docling-core/commit/8528918ae24eb1e50b97935b1136e8e8e2d9717b))
* Title for export to markdown and add text_width parameter ([#59](https://github.com/DS4SD/docling-core/issues/59)) ([`4993c34`](https://github.com/DS4SD/docling-core/commit/4993c3403d0dac869033c38349b785e5c78ac1d9))

## [v2.2.2](https://github.com/DS4SD/docling-core/releases/tag/v2.2.2) - 2024-10-26

### Fix

* Fix non-string table cell handling in chunker ([#58](https://github.com/DS4SD/docling-core/issues/58)) ([`b5d07b2`](https://github.com/DS4SD/docling-core/commit/b5d07b2fa5e865939b9b93a8582eb3207bc48249))

## [v2.2.1](https://github.com/DS4SD/docling-core/releases/tag/v2.2.1) - 2024-10-25

### Fix

* Escaping underscore characters in md export ([#57](https://github.com/DS4SD/docling-core/issues/57)) ([`c344d0f`](https://github.com/DS4SD/docling-core/commit/c344d0fc63ac55068dac5846772e3369f71c771b))

## [v2.2.0](https://github.com/DS4SD/docling-core/releases/tag/v2.2.0) - 2024-10-24

### Feature

* Add headers argument and a custom user-agents for http requests ([#53](https://github.com/DS4SD/docling-core/issues/53)) ([`44941b5`](https://github.com/DS4SD/docling-core/commit/44941b5591fe7db5a2b987f5d2b35785fab386db))

### Fix

* Fix resolution in case of URL without path ([#55](https://github.com/DS4SD/docling-core/issues/55)) ([`2c88e56`](https://github.com/DS4SD/docling-core/commit/2c88e56f8ccb5902457c1749c64c8de6cc963739))

## [v2.1.0](https://github.com/DS4SD/docling-core/releases/tag/v2.1.0) - 2024-10-22

### Feature

* Improve markdown export of DoclingDocument ([#50](https://github.com/DS4SD/docling-core/issues/50)) ([`328778e`](https://github.com/DS4SD/docling-core/commit/328778ed036540ee5fdf0bb16a1b656e5122c5f0))
* Extend chunk meta with schema, version, origin ([#49](https://github.com/DS4SD/docling-core/issues/49)) ([`d09fe7e`](https://github.com/DS4SD/docling-core/commit/d09fe7ed44282b286f9c2588482e515bf40e0fca))

## [v2.0.1](https://github.com/DS4SD/docling-core/releases/tag/v2.0.1) - 2024-10-18

### Fix

* Fix legacy doc ref ([#48](https://github.com/DS4SD/docling-core/issues/48)) ([`e12d6a7`](https://github.com/DS4SD/docling-core/commit/e12d6a70c383a8d3f0c3d73aa6c5eec62a0c3251))
* Add-mimetype-for-asciidocs-markdown ([#47](https://github.com/DS4SD/docling-core/issues/47)) ([`0aab007`](https://github.com/DS4SD/docling-core/commit/0aab0073bbeabe62b2a19e872e108438c199f6c0))

## [v2.0.0](https://github.com/DS4SD/docling-core/releases/tag/v2.0.0) - 2024-10-16

### Feature

* Expose DoclingDocument as main type, move old typing to legacy ([#41](https://github.com/DS4SD/docling-core/issues/41)) ([`03df97f`](https://github.com/DS4SD/docling-core/commit/03df97fa73db2499682613ff17dff9fe996a1bdc))

### Breaking

* Expose DoclingDocument as main type, move old typing to legacy ([#41](https://github.com/DS4SD/docling-core/issues/41)) ([`03df97f`](https://github.com/DS4SD/docling-core/commit/03df97fa73db2499682613ff17dff9fe996a1bdc))

## [v1.7.2](https://github.com/DS4SD/docling-core/releases/tag/v1.7.2) - 2024-10-09

### Fix

* Loosen pandas version ([#40](https://github.com/DS4SD/docling-core/issues/40)) ([`aec1a41`](https://github.com/DS4SD/docling-core/commit/aec1a41402d64c2c216923b40d4521f8d46540b7))

## [v1.7.1](https://github.com/DS4SD/docling-core/releases/tag/v1.7.1) - 2024-10-07

### Fix

* Make doc metadata keys pure strings ([#38](https://github.com/DS4SD/docling-core/issues/38)) ([`246627f`](https://github.com/DS4SD/docling-core/commit/246627f4f6aef1121dd4211cc223f356a133c60e))
* Align chunk ref format with one used in Document ([#37](https://github.com/DS4SD/docling-core/issues/37)) ([`b5592ad`](https://github.com/DS4SD/docling-core/commit/b5592ad747a061cb6b47c86228063371a80a9b44))

## [v1.7.0](https://github.com/DS4SD/docling-core/releases/tag/v1.7.0) - 2024-10-01

### Feature

* (experimental) introduce new document format ([#21](https://github.com/DS4SD/docling-core/issues/21)) ([`688789e`](https://github.com/DS4SD/docling-core/commit/688789ea751d75c15a6957dba4ba496b899e9d11))
* Add doc metadata extractor and ID generator classes ([#34](https://github.com/DS4SD/docling-core/issues/34)) ([`b76780c`](https://github.com/DS4SD/docling-core/commit/b76780c3b21a89d407b6afb5e72cd4f46dbcf569))
* Support heading as chunk metadata ([#36](https://github.com/DS4SD/docling-core/issues/36)) ([`4bde515`](https://github.com/DS4SD/docling-core/commit/4bde51528d23be9bed797030a75991f6acdb241f))

## [v1.6.3](https://github.com/DS4SD/docling-core/releases/tag/v1.6.3) - 2024-09-26

### Fix

* Change order of JSON Schema to search mapper transformations ([#32](https://github.com/DS4SD/docling-core/issues/32)) ([`a4ddd14`](https://github.com/DS4SD/docling-core/commit/a4ddd142eef864c55b62c8815d38dbff14f4caa7))

## [v1.6.2](https://github.com/DS4SD/docling-core/releases/tag/v1.6.2) - 2024-09-24

### Fix

* Remove duplicate captions in markdown ([#31](https://github.com/DS4SD/docling-core/issues/31)) ([`a334b9f`](https://github.com/DS4SD/docling-core/commit/a334b9fc721a2e1efc9f12b585cff17363875d57))

## [v1.6.1](https://github.com/DS4SD/docling-core/releases/tag/v1.6.1) - 2024-09-24

### Fix

* Remove unnecessary package dependency ([#30](https://github.com/DS4SD/docling-core/issues/30)) ([`e706d68`](https://github.com/DS4SD/docling-core/commit/e706d686db159f6480439d214c85b1664f38e28f))

## [v1.6.0](https://github.com/DS4SD/docling-core/releases/tag/v1.6.0) - 2024-09-23

### Feature

* Add figures in markdown export ([#27](https://github.com/DS4SD/docling-core/issues/27)) ([`b843ae6`](https://github.com/DS4SD/docling-core/commit/b843ae6688a20e68e2da59b2f68fd61f8d4beacb))

## [v1.5.0](https://github.com/DS4SD/docling-core/releases/tag/v1.5.0) - 2024-09-20

### Feature

* Add export to doctags for document components ([#25](https://github.com/DS4SD/docling-core/issues/25)) ([`891530f`](https://github.com/DS4SD/docling-core/commit/891530f595dbf656bbc2708fb25a05aa1ec65afa))
* Add file source resolution utility ([#22](https://github.com/DS4SD/docling-core/issues/22)) ([`752cbc3`](https://github.com/DS4SD/docling-core/commit/752cbc3e89461fa633277cfe3887bc5a6fa5c2b0))

## [v1.4.1](https://github.com/DS4SD/docling-core/releases/tag/v1.4.1) - 2024-09-18

### Fix

* Add export to xml and html ([#17](https://github.com/DS4SD/docling-core/issues/17)) ([`9bc256e`](https://github.com/DS4SD/docling-core/commit/9bc256e5bbbe02cc0a317bc2920c8e0becb3090c))

## [v1.4.0](https://github.com/DS4SD/docling-core/releases/tag/v1.4.0) - 2024-09-18

### Feature

* Add table exporters ([#20](https://github.com/DS4SD/docling-core/issues/20)) ([`2cc2429`](https://github.com/DS4SD/docling-core/commit/2cc2429e2731998c3282ba133995439450f08574))

## [v1.3.0](https://github.com/DS4SD/docling-core/releases/tag/v1.3.0) - 2024-09-11

### Feature

* Add hierarchical chunker ([#18](https://github.com/DS4SD/docling-core/issues/18)) ([`9698d30`](https://github.com/DS4SD/docling-core/commit/9698d30288df17ecde67f170848f1be47cd97d33))

## [v1.2.0](https://github.com/DS4SD/docling-core/releases/tag/v1.2.0) - 2024-09-10

### Feature

* Added the XML export ([#16](https://github.com/DS4SD/docling-core/issues/16)) ([`acdf816`](https://github.com/DS4SD/docling-core/commit/acdf81608134c23969c9e620085f4fff4f42a12f))

## [v1.1.4](https://github.com/DS4SD/docling-core/releases/tag/v1.1.4) - 2024-09-06

### Fix

* Validate_model() could be called with other types rather than dict ([#14](https://github.com/DS4SD/docling-core/issues/14)) ([`235b2cd`](https://github.com/DS4SD/docling-core/commit/235b2cd10b595c813c03db5b4effbc7cc2feaaf0))

### Documentation

* Update link to Docling ([#12](https://github.com/DS4SD/docling-core/issues/12)) ([`aaf17fe`](https://github.com/DS4SD/docling-core/commit/aaf17fe0f6eae7ee21c54c56fded05f24ec936b1))

## [v1.1.3](https://github.com/DS4SD/docling-core/releases/tag/v1.1.3) - 2024-08-28

### Fix

* Use same base type for all components ([#10](https://github.com/DS4SD/docling-core/issues/10)) ([`f450c8c`](https://github.com/DS4SD/docling-core/commit/f450c8cbfd623bf5c7013bae956d23618004f43d))

## [v1.1.2](https://github.com/DS4SD/docling-core/releases/tag/v1.1.2) - 2024-07-31

### Fix

* Make page number strictly positive ([#8](https://github.com/DS4SD/docling-core/issues/8)) ([`ec3cff9`](https://github.com/DS4SD/docling-core/commit/ec3cff97e5079251087cd7b4b42e8c509cd244f3))

## [v1.1.1](https://github.com/DS4SD/docling-core/releases/tag/v1.1.1) - 2024-07-23

### Fix

* Set type to optional ([#7](https://github.com/DS4SD/docling-core/issues/7)) ([`faf472c`](https://github.com/DS4SD/docling-core/commit/faf472c1689746adc43e0ae8ef6d6e3fcf87c023))

### Documentation

* Revamp installation instructions ([#6](https://github.com/DS4SD/docling-core/issues/6)) ([`3f77b2e`](https://github.com/DS4SD/docling-core/commit/3f77b2e92c415c7290df8c4d534ba3455dbe62bd))

## [v1.1.0](https://github.com/DS4SD/docling-core/releases/tag/v1.1.0) - 2024-07-18

### Feature

* Add document Markdown export ([#4](https://github.com/DS4SD/docling-core/issues/4)) ([`d0ffc85`](https://github.com/DS4SD/docling-core/commit/d0ffc85e0c2b49d201f5359c4dc4efb5cd5716b0))

## [v1.0.0](https://github.com/DS4SD/docling-core/releases/tag/v1.0.0) - 2024-07-17

### Feature

* Trigger v1.0.0 release ([#3](https://github.com/DS4SD/docling-core/issues/3)) ([`daece4c`](https://github.com/DS4SD/docling-core/commit/daece4ceae363351072aa7e0adb91037e0dd7b66))

### Breaking

* trigger v1.0.0 release ([#3](https://github.com/DS4SD/docling-core/issues/3)) ([`daece4c`](https://github.com/DS4SD/docling-core/commit/daece4ceae363351072aa7e0adb91037e0dd7b66))

## [v0.0.1](https://github.com/DS4SD/docling-core/releases/tag/v0.0.1) - 2024-07-17

### Fix

* Fix definition issues in record type ([#2](https://github.com/DS4SD/docling-core/issues/2)) ([`656f563`](https://github.com/DS4SD/docling-core/commit/656f56380f603c3de125f6c59554f26ac8cd0a78))
