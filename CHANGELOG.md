## [v2.34.0](https://github.com/docling-project/docling-core/releases/tag/v2.34.0) - 2025-06-06

### Feature

* **doctags:** Add enclosing bbox to inline ([#302](https://github.com/docling-project/docling-core/issues/302)) ([`dcc198f`](https://github.com/docling-project/docling-core/commit/dcc198f7c6231fe8c781abe4a83194be2ee8d23b))
* Add subscript & superscript formatting ([#319](https://github.com/docling-project/docling-core/issues/319)) ([`ae96129`](https://github.com/docling-project/docling-core/commit/ae961299a5f729acecf1b2b346113ac23e8b97f0))
* Add table annotations ([#304](https://github.com/docling-project/docling-core/issues/304)) ([`d8a5256`](https://github.com/docling-project/docling-core/commit/d8a5256b2cb654ceb35a70be1b656ac7463ad335))

### Fix

* Fix misplaced list items ([#317](https://github.com/docling-project/docling-core/issues/317)) ([`c383f64`](https://github.com/docling-project/docling-core/commit/c383f64c44b4e1eb760d19d9422948fea127331c))

## [v2.33.1](https://github.com/docling-project/docling-core/releases/tag/v2.33.1) - 2025-06-04

### Fix

* New typer version with new click ([#315](https://github.com/docling-project/docling-core/issues/315)) ([`e17eabf`](https://github.com/docling-project/docling-core/commit/e17eabf0f92c3e3fde9f47fea083e44081bb9669))
* Support section_header levels in doctags deserialization ([#313](https://github.com/docling-project/docling-core/issues/313)) ([`defd49e`](https://github.com/docling-project/docling-core/commit/defd49efae08c0cc2ce002f847724b6d65bd1407))

## [v2.33.0](https://github.com/docling-project/docling-core/releases/tag/v2.33.0) - 2025-06-02

### Feature

* Add BoundingBox methods for overlap and union calculations ([#311](https://github.com/docling-project/docling-core/issues/311)) ([`c521766`](https://github.com/docling-project/docling-core/commit/c521766cb7eb0dfb7d2e1eb5befe3e7e13453e9a))

## [v2.32.0](https://github.com/docling-project/docling-core/releases/tag/v2.32.0) - 2025-05-27

### Feature

* Add annotations in MD & HTML serialization ([#295](https://github.com/docling-project/docling-core/issues/295)) ([`f067c51`](https://github.com/docling-project/docling-core/commit/f067c51c48d3ff69438ef9cc301f45a565b341ee))

### Fix

* **HybridChunker:** Refine `max_tokens` auto-detection ([#306](https://github.com/docling-project/docling-core/issues/306)) ([`87b72d6`](https://github.com/docling-project/docling-core/commit/87b72d653789847ae170110fbb551f34277f0399))

## [v2.31.2](https://github.com/docling-project/docling-core/releases/tag/v2.31.2) - 2025-05-22

### Fix

* Fix hybrid chunker legacy patching ([#300](https://github.com/docling-project/docling-core/issues/300)) ([`ebc356a`](https://github.com/docling-project/docling-core/commit/ebc356a787a404ca73dbe473d7d5e17fa4951954))

## [v2.31.1](https://github.com/docling-project/docling-core/releases/tag/v2.31.1) - 2025-05-20

### Fix

* **markdown:** Fix case of empty page break string ([#298](https://github.com/docling-project/docling-core/issues/298)) ([`c49a50e`](https://github.com/docling-project/docling-core/commit/c49a50e76ba74579350eb874c2b731490c00ea1f))

## [v2.31.0](https://github.com/docling-project/docling-core/releases/tag/v2.31.0) - 2025-05-18

### Feature

* Provide visualizer option in HTML split view ([#294](https://github.com/docling-project/docling-core/issues/294)) ([`6a7eb53`](https://github.com/docling-project/docling-core/commit/6a7eb537eb6a9f35c3ffa3d43ebbad4e5ce5ab9c))

## [v2.30.1](https://github.com/docling-project/docling-core/releases/tag/v2.30.1) - 2025-05-14

### Fix

* Updates for labels and methods to support document GT annotation ([#293](https://github.com/docling-project/docling-core/issues/293)) ([`aa957cf`](https://github.com/docling-project/docling-core/commit/aa957cf4b6bb083886dc51b716189d046c38005c))

## [v2.30.0](https://github.com/docling-project/docling-core/releases/tag/v2.30.0) - 2025-05-06

### Feature

* Add image group serialization in html ([#284](https://github.com/docling-project/docling-core/issues/284)) ([`7f83f1c`](https://github.com/docling-project/docling-core/commit/7f83f1ce84dd4a0ae479430187c193d96d545ba9))
* Adding the label picture_group ([#283](https://github.com/docling-project/docling-core/issues/283)) ([`2f0f121`](https://github.com/docling-project/docling-core/commit/2f0f12160b56808a068dd157a8acf3462fcb95c9))

### Fix

* Add unit flags to SegmentedPage ([#286](https://github.com/docling-project/docling-core/issues/286)) ([`ad88ecf`](https://github.com/docling-project/docling-core/commit/ad88ecf845e297a6a743fd8bf880ad7c0c9c502e))
* Update deserialization for better recovery ([#282](https://github.com/docling-project/docling-core/issues/282)) ([`511fb98`](https://github.com/docling-project/docling-core/commit/511fb98a03e60232c5bf2fdc629c936969b05e95))
* Include captions regardless of `traverse_pictures` flag ([#278](https://github.com/docling-project/docling-core/issues/278)) ([`7eb9fa9`](https://github.com/docling-project/docling-core/commit/7eb9fa96e632529dd5881e78121ad898b82c8bf3))
* Hashlib usage for FIPS ([#280](https://github.com/docling-project/docling-core/issues/280)) ([`4b967ab`](https://github.com/docling-project/docling-core/commit/4b967ab55c646719ac8b7194592c46e3acc2ef76))

## [v2.29.0](https://github.com/docling-project/docling-core/releases/tag/v2.29.0) - 2025-05-01

### Feature

* Promote serializers to stable API ([#276](https://github.com/docling-project/docling-core/issues/276)) ([`d05fe08`](https://github.com/docling-project/docling-core/commit/d05fe085468fa2548b46d90967038facbdb166b6))

### Fix

* Fix multi-provenance item visualization ([#277](https://github.com/docling-project/docling-core/issues/277)) ([`8677d6e`](https://github.com/docling-project/docling-core/commit/8677d6e9c67c1bac72c8f078fdcdc8891148f32a))
* Added return value for crop_text method in segmentedPdfPage Class ([#275](https://github.com/docling-project/docling-core/issues/275)) ([`591fe59`](https://github.com/docling-project/docling-core/commit/591fe59357192a05d0ad24310c149edda00c0af4))
* Make load_from_doctags method static ([#273](https://github.com/docling-project/docling-core/issues/273)) ([`8f85d05`](https://github.com/docling-project/docling-core/commit/8f85d056e8d58964180756572b37138cffef028a))

## [v2.28.1](https://github.com/docling-project/docling-core/releases/tag/v2.28.1) - 2025-04-25

### Fix

* Visualization of document pages without items ([#271](https://github.com/docling-project/docling-core/issues/271)) ([`a947440`](https://github.com/docling-project/docling-core/commit/a94744054509df8ba450629e8276c335b3a4d902))
* UnboundLocal variable ([#269](https://github.com/docling-project/docling-core/issues/269)) ([`d9709d0`](https://github.com/docling-project/docling-core/commit/d9709d0b124427e414abe8d7f7a3a89e649a7a44))

## [v2.28.0](https://github.com/docling-project/docling-core/releases/tag/v2.28.0) - 2025-04-23

### Feature

* Add tiktoken tokenizers support to HybridChunker ([#240](https://github.com/docling-project/docling-core/issues/240)) ([`763e136`](https://github.com/docling-project/docling-core/commit/763e1364ff0b95388696ccd3d69f150718012a3a))
* Add visualizers ([#263](https://github.com/docling-project/docling-core/issues/263)) ([`a258d52`](https://github.com/docling-project/docling-core/commit/a258d525e13fe1112ff1f0590fe5f4462feab7ff))

## [v2.27.0](https://github.com/docling-project/docling-core/releases/tag/v2.27.0) - 2025-04-16

### Feature

* Chart tabular data serialization for HTML serializer ([#258](https://github.com/docling-project/docling-core/issues/258)) ([`caa8aee`](https://github.com/docling-project/docling-core/commit/caa8aeefae81b893cd38bcc4349fe7112bd0eb24))

### Fix

* HTML serialization for single image documents ([#261](https://github.com/docling-project/docling-core/issues/261)) ([`d0a49da`](https://github.com/docling-project/docling-core/commit/d0a49da6de0da65275c709316bcdecaebdb6b883))
* **codecov:** Fix codecov argument and yaml file ([#260](https://github.com/docling-project/docling-core/issues/260)) ([`1af0721`](https://github.com/docling-project/docling-core/commit/1af07218e1496029693eb7dcebd6f4ac4451003d))
* Safer label color API ([#259](https://github.com/docling-project/docling-core/issues/259)) ([`159f61d`](https://github.com/docling-project/docling-core/commit/159f61d6d8ce79d5dad1d47af729019c73c68d62))

## [v2.26.4](https://github.com/docling-project/docling-core/releases/tag/v2.26.4) - 2025-04-14

### Fix

* Fix page breaking in case page starts with group ([#253](https://github.com/docling-project/docling-core/issues/253)) ([`928e5c5`](https://github.com/docling-project/docling-core/commit/928e5c504612f13a7ec5bc3381f81c581cb9691c))

## [v2.26.3](https://github.com/docling-project/docling-core/releases/tag/v2.26.3) - 2025-04-14

### Fix

* **deps:** Widen typer upper bound ([#252](https://github.com/docling-project/docling-core/issues/252)) ([`437aef0`](https://github.com/docling-project/docling-core/commit/437aef0a410dc3f501beafe02feb2857613f7b4f))

## [v2.26.2](https://github.com/docling-project/docling-core/releases/tag/v2.26.2) - 2025-04-14

### Fix

* Fix code handling in HTML serialization ([#251](https://github.com/docling-project/docling-core/issues/251)) ([`15d2f2c`](https://github.com/docling-project/docling-core/commit/15d2f2c1d5df9c73bbe3d616a17f5e5000393c46))

## [v2.26.1](https://github.com/docling-project/docling-core/releases/tag/v2.26.1) - 2025-04-11

### Performance

* Fix serialization performance ([#249](https://github.com/docling-project/docling-core/issues/249)) ([`98c60bb`](https://github.com/docling-project/docling-core/commit/98c60bb65b080d460c9d74c1332eb781f174ce5f))

## [v2.26.0](https://github.com/docling-project/docling-core/releases/tag/v2.26.0) - 2025-04-11

### Feature

* Add HTML serializer ([#232](https://github.com/docling-project/docling-core/issues/232)) ([`5d40600`](https://github.com/docling-project/docling-core/commit/5d406008f2f8afe3dbb70e97f3a70681ddeca057))
* Add serializer provider to chunkers ([#239](https://github.com/docling-project/docling-core/issues/239)) ([`23036e1`](https://github.com/docling-project/docling-core/commit/23036e17fae03f99d123781dfe0c5229b1daf59c))
* Integrate serialization API into chunkers ([#221](https://github.com/docling-project/docling-core/issues/221)) ([`5e4c0fd`](https://github.com/docling-project/docling-core/commit/5e4c0fd51fc410c4cd0f91e7566bfbdfc62b3695))
* Expose page number in Serialization API ([#238](https://github.com/docling-project/docling-core/issues/238)) ([`73b9941`](https://github.com/docling-project/docling-core/commit/73b99410a9bc0c039cd557a76657c56b9342eb0b))
* Markdown chart serializer (picture+table) ([#235](https://github.com/docling-project/docling-core/issues/235)) ([`0482bac`](https://github.com/docling-project/docling-core/commit/0482bacb5bd34051ea9f743ae968c2d12dba51fc))
* Support of DocTags charts (serialization and deserialization) ([#229](https://github.com/docling-project/docling-core/issues/229)) ([`e9259a5`](https://github.com/docling-project/docling-core/commit/e9259a5f8778a4a8ab814a4f75b64e41a1b52946))
* Added initial delete and insert methods in DoclingDocument ([#220](https://github.com/docling-project/docling-core/issues/220)) ([`f2fe1c1`](https://github.com/docling-project/docling-core/commit/f2fe1c1bc61159854af01e3dbad05690cf5a7a35))

### Fix

* Fix page filtering issue ([#247](https://github.com/docling-project/docling-core/issues/247)) ([`ab78e0b`](https://github.com/docling-project/docling-core/commit/ab78e0bd9ce2352c175c63f69e79ef9dbf0281b7))
* Propagate HTMLOutputStyle properly through ([#246](https://github.com/docling-project/docling-core/issues/246)) ([`587e67f`](https://github.com/docling-project/docling-core/commit/587e67f423c2c7ed432bccd0e9b5c9dce8747f95))
* Better `BoundingRectangle.angle` and `BoundingRectangle.angle_360` computation ([#237](https://github.com/docling-project/docling-core/issues/237)) ([`055742c`](https://github.com/docling-project/docling-core/commit/055742c2e809087562c39130345e33af10686169))
* DocTags import location fix for tables, pictures, captions ([#227](https://github.com/docling-project/docling-core/issues/227)) ([`a055e1a`](https://github.com/docling-project/docling-core/commit/a055e1a9f0a80ccf78e070f0b2179fede73ccf71))

### Performance

* Accelerate span deduplication ([#248](https://github.com/docling-project/docling-core/issues/248)) ([`26f639d`](https://github.com/docling-project/docling-core/commit/26f639d8b833a049cb33490876698b669345ac44))

## [v2.25.0](https://github.com/docling-project/docling-core/releases/tag/v2.25.0) - 2025-03-31

### Feature

* Allow images in doctags deserializer to be optional and support multipage ([#225](https://github.com/docling-project/docling-core/issues/225)) ([`e0943d2`](https://github.com/docling-project/docling-core/commit/e0943d24ec80d3973a1a30e6755e35695f49b432))

### Fix

* DivisionByZero in intersection_over_self ([#224](https://github.com/docling-project/docling-core/issues/224)) ([`2f380ab`](https://github.com/docling-project/docling-core/commit/2f380abd174624a1ac36c47257eba0a659413f01))
* Fix hyperlink deserialization ([#223](https://github.com/docling-project/docling-core/issues/223)) ([`57d26ee`](https://github.com/docling-project/docling-core/commit/57d26eed112815af39b5a89a71f2a28ffb6aad67))

## [v2.24.1](https://github.com/docling-project/docling-core/releases/tag/v2.24.1) - 2025-03-28

### Fix

* Automatic transformation of output cells bbox coord origin defined by input in get_cells_in_bbox ([#219](https://github.com/docling-project/docling-core/issues/219)) ([`8e0e9b7`](https://github.com/docling-project/docling-core/commit/8e0e9b72bf04fe4605ccfcbb1ec463e5fce87694))

## [v2.24.0](https://github.com/docling-project/docling-core/releases/tag/v2.24.0) - 2025-03-25

### Feature

* Expose MD page break & DocTags minification ([#213](https://github.com/docling-project/docling-core/issues/213)) ([`ff13a93`](https://github.com/docling-project/docling-core/commit/ff13a9385d91b51f3c01b6cfbde72a1d5ce7c4c9))
* Add document tokens from key value items ([#170](https://github.com/docling-project/docling-core/issues/170)) ([`db119f4`](https://github.com/docling-project/docling-core/commit/db119f46655f9b75bc55dd068b2c219ed2ff4332))
* Add DocTags serializers ([#192](https://github.com/docling-project/docling-core/issues/192)) ([`1f4d57e`](https://github.com/docling-project/docling-core/commit/1f4d57ed187c2b5e0a66e96e192ff985ab6e5c05))
* Add kv_item support for doctag to docling_document ([#188](https://github.com/docling-project/docling-core/issues/188)) ([`2371c11`](https://github.com/docling-project/docling-core/commit/2371c11b8f74628169a9bb377036511235070af0))

### Fix

* Enable caption serialization for all floating items ([#216](https://github.com/docling-project/docling-core/issues/216)) ([`e1d0597`](https://github.com/docling-project/docling-core/commit/e1d0597338aa2a9f2c2500484556b31ff649198d))
* Allow captions without holding item ([#215](https://github.com/docling-project/docling-core/issues/215)) ([`2efb71a`](https://github.com/docling-project/docling-core/commit/2efb71a0ca16d408ec6bc817d967688444c13a7e))
* Add 'text/csv' mimetype to _extra_mimetypes type list ([#210](https://github.com/docling-project/docling-core/issues/210)) ([`bc3f5d5`](https://github.com/docling-project/docling-core/commit/bc3f5d57c39a47007beca2c11008de058a65243d))
* Add handling for str filenames in save/load methods ([#205](https://github.com/docling-project/docling-core/issues/205)) ([`75d94ab`](https://github.com/docling-project/docling-core/commit/75d94abc3189278679a7661926218c19b00345f8))
* Markdown picture item export ([#207](https://github.com/docling-project/docling-core/issues/207)) ([`510649e`](https://github.com/docling-project/docling-core/commit/510649e51a7b3eec2640958f4217fac4c6da7514))
* DocTags support of furniture ([#209](https://github.com/docling-project/docling-core/issues/209)) ([`337ff74`](https://github.com/docling-project/docling-core/commit/337ff74dae59bab8742abc4fe449462a42f42d5b))

### Performance

* **serialization:** Cache excluded references ([#214](https://github.com/docling-project/docling-core/issues/214)) ([`bcace5d`](https://github.com/docling-project/docling-core/commit/bcace5df9f0fea901580c65c9eb5f80affb51b75))

## [v2.23.3](https://github.com/docling-project/docling-core/releases/tag/v2.23.3) - 2025-03-19

### Fix

* **markdown:** Fix ordered list numbering ([#200](https://github.com/docling-project/docling-core/issues/200)) ([`7ed4d22`](https://github.com/docling-project/docling-core/commit/7ed4d225b67dd41aa2c3e7c0d4b2b96f9e95114e))

## [v2.23.2](https://github.com/docling-project/docling-core/releases/tag/v2.23.2) - 2025-03-18

### Fix

* Add caption to the table in load_from_doctags ([#197](https://github.com/docling-project/docling-core/issues/197)) ([`5cee486`](https://github.com/docling-project/docling-core/commit/5cee4866143dd2c770cac5a44d525f1ce5b2c94d))

## [v2.23.1](https://github.com/docling-project/docling-core/releases/tag/v2.23.1) - 2025-03-17

### Fix

* New favicon url ([#191](https://github.com/docling-project/docling-core/issues/191)) ([`11eb1dc`](https://github.com/docling-project/docling-core/commit/11eb1dc5aa317c91f6558e93b44d3015b3cb4e98))

## [v2.23.0](https://github.com/docling-project/docling-core/releases/tag/v2.23.0) - 2025-03-13

### Feature

* Add serializers, text formatting, update Markdown export ([#182](https://github.com/docling-project/docling-core/issues/182)) ([`a7cdc87`](https://github.com/docling-project/docling-core/commit/a7cdc87411102b15da5459e0b5d00fbe966aea9d))
* Add data model types from docling-parse ([#186](https://github.com/docling-project/docling-core/issues/186)) ([`a86a4a3`](https://github.com/docling-project/docling-core/commit/a86a4a3aa36ad0e69835af3aeba381be39a99cb9))

## [v2.22.0](https://github.com/docling-project/docling-core/releases/tag/v2.22.0) - 2025-03-12

### Feature

* Add DoclingDocument.load_from_doctags method and DocTags data models ([#187](https://github.com/docling-project/docling-core/issues/187)) ([`c065c4c`](https://github.com/docling-project/docling-core/commit/c065c4c7c62f126879fdf78661521400779602af))
* Add document tokens for SMILES ([#176](https://github.com/docling-project/docling-core/issues/176)) ([`32398b8`](https://github.com/docling-project/docling-core/commit/32398b8ac0c909933b93fd6abea6979acbef8992))

## [v2.21.2](https://github.com/docling-project/docling-core/releases/tag/v2.21.2) - 2025-03-06

### Fix

* Suppress warning for missing fallback case ([#184](https://github.com/docling-project/docling-core/issues/184)) ([`ccde54a`](https://github.com/docling-project/docling-core/commit/ccde54aa2926281644e5c1f0c96b79db18f6bbc7))
* **doctags:** Fix code export ([#181](https://github.com/docling-project/docling-core/issues/181)) ([`53f6d09`](https://github.com/docling-project/docling-core/commit/53f6d099b05f295fea546010dc2faadc5b2c7ee2))
* **markdown:** Fix escaping in case of nesting ([#180](https://github.com/docling-project/docling-core/issues/180)) ([`834db4b`](https://github.com/docling-project/docling-core/commit/834db4bc664e010e10c4503e60be576ed7819e2c))
* **HybridChunker:** Remove `max_length` from tokenization ([#178](https://github.com/docling-project/docling-core/issues/178)) ([`419252c`](https://github.com/docling-project/docling-core/commit/419252c39b856c45e50326b4eff3c4a183ac8437))

## [v2.21.1](https://github.com/docling-project/docling-core/releases/tag/v2.21.1) - 2025-02-28

### Fix

* **markdown:** Fix handling of ordered lists ([#175](https://github.com/docling-project/docling-core/issues/175)) ([`349f7da`](https://github.com/docling-project/docling-core/commit/349f7daa0c20c861134cffb28177eaaf48b27ae5))

## [v2.21.0](https://github.com/docling-project/docling-core/releases/tag/v2.21.0) - 2025-02-27

### Feature

* Add inline groups, revamp Markdown export incl. list groups ([#156](https://github.com/docling-project/docling-core/issues/156)) ([`2abaf9b`](https://github.com/docling-project/docling-core/commit/2abaf9b53736187adec0266c5ed8b9acff008f6e))

### Fix

* **markdown:** Fix case of leading list ([#174](https://github.com/docling-project/docling-core/issues/174)) ([`c77c59b`](https://github.com/docling-project/docling-core/commit/c77c59bec09d4b8093771935393f558cf319ec29))
* Properly handle missing page image case for export_to_html ([#166](https://github.com/docling-project/docling-core/issues/166)) ([`4708f93`](https://github.com/docling-project/docling-core/commit/4708f933a7ef87e4637f5bea07e6e4f296abc51a))

## [v2.20.0](https://github.com/docling-project/docling-core/releases/tag/v2.20.0) - 2025-02-19

### Feature

* Introduce Key-Value and Forms items ([#158](https://github.com/docling-project/docling-core/issues/158)) ([`d622800`](https://github.com/docling-project/docling-core/commit/d6228007502fc1f27400059eae7bb768209c0a6f))

## [v2.19.1](https://github.com/docling-project/docling-core/releases/tag/v2.19.1) - 2025-02-17

### Fix

* Expose included_content_layers arg in export/save methods for MD+HTML ([#164](https://github.com/docling-project/docling-core/issues/164)) ([`c46995b`](https://github.com/docling-project/docling-core/commit/c46995bca39fbaa2a9d1fb68c5c9cb5beb6d6722))

## [v2.19.0](https://github.com/docling-project/docling-core/releases/tag/v2.19.0) - 2025-02-17

### Feature

* Redefine CodeItem as floating object with captions ([#160](https://github.com/docling-project/docling-core/issues/160)) ([`916323f`](https://github.com/docling-project/docling-core/commit/916323fb55274753aa1d6a4928388a35417f94b6))
* Implementation of doc tags ([#138](https://github.com/docling-project/docling-core/issues/138)) ([`f751b45`](https://github.com/docling-project/docling-core/commit/f751b45b62fb318929f8131ab82fa17db98e8e44))

### Fix

* Document Tokens (doc tags) clean up, fix iterate_items for content_layer ([#161](https://github.com/docling-project/docling-core/issues/161)) ([`58ed6c8`](https://github.com/docling-project/docling-core/commit/58ed6c8ab75ba179faf1598b9877662cdcc4c1d3))
* Fix inheritance of CodeItem for backward compatibility ([#162](https://github.com/docling-project/docling-core/issues/162)) ([`7267c3f`](https://github.com/docling-project/docling-core/commit/7267c3f5716d3f292592d3b11ddd2b0db4392c20))

## [v2.18.1](https://github.com/docling-project/docling-core/releases/tag/v2.18.1) - 2025-02-13

### Fix

* Update Pillow constraints ([#157](https://github.com/docling-project/docling-core/issues/157)) ([`a9afeda`](https://github.com/docling-project/docling-core/commit/a9afeda6d1251900142571f7bff3d00d871d5915))

## [v2.18.0](https://github.com/docling-project/docling-core/releases/tag/v2.18.0) - 2025-02-10

### Feature

* Add ContentLayer attribute to designate items to body or furniture ([#148](https://github.com/docling-project/docling-core/issues/148)) ([`786f0c6`](https://github.com/docling-project/docling-core/commit/786f0c68336a7b9cced5fb0cb66427b050955e32))

## [v2.17.2](https://github.com/docling-project/docling-core/releases/tag/v2.17.2) - 2025-02-06

### Fix

* Define LTR/RTL text direction in HTML export ([#152](https://github.com/docling-project/docling-core/issues/152)) ([`3cf31cb`](https://github.com/docling-project/docling-core/commit/3cf31cbe384e3f77a375aa057ef61d156d990b23))

## [v2.17.1](https://github.com/docling-project/docling-core/releases/tag/v2.17.1) - 2025-02-03

### Fix

* Image fallback for malformed equations ([#149](https://github.com/docling-project/docling-core/issues/149)) ([`eb9b4b3`](https://github.com/docling-project/docling-core/commit/eb9b4b39a1a2f81baf72d3fa3bbc7cd8ed594c1c))

## [v2.17.0](https://github.com/docling-project/docling-core/releases/tag/v2.17.0) - 2025-02-03

### Feature

* **HTML:** Fallback showing formulas as images ([#146](https://github.com/docling-project/docling-core/issues/146)) ([`23477f7`](https://github.com/docling-project/docling-core/commit/23477f76741b3593734287776fdf5e0761558c2d))
* **HTML:** Export formulas with mathml ([#144](https://github.com/docling-project/docling-core/issues/144)) ([`ed36437`](https://github.com/docling-project/docling-core/commit/ed36437346177b9249c98df3eb5ddeadef004c59))

### Fix

* Add html escape in md export and fix formula escapes ([#143](https://github.com/docling-project/docling-core/issues/143)) ([`c6590e8`](https://github.com/docling-project/docling-core/commit/c6590e83e28626e4a6b62fdbd270cb794bf10918))

## [v2.16.1](https://github.com/docling-project/docling-core/releases/tag/v2.16.1) - 2025-01-30

### Fix

* Add newline to md formula export ([#142](https://github.com/docling-project/docling-core/issues/142)) ([`d07a87e`](https://github.com/docling-project/docling-core/commit/d07a87e1fbc777cd6d01c7646d714a44a69bc123))

## [v2.16.0](https://github.com/docling-project/docling-core/releases/tag/v2.16.0) - 2025-01-29

### Feature

* Escape underscores that are within latex equations ([#137](https://github.com/docling-project/docling-core/issues/137)) ([`0d5cd11`](https://github.com/docling-project/docling-core/commit/0d5cd11326d8521360add6ffaa3de845bf72abe2))
* Add escaping_underscores option to markdown export ([#135](https://github.com/docling-project/docling-core/issues/135)) ([`c9739b2`](https://github.com/docling-project/docling-core/commit/c9739b2c6cf0686747fbda5331e1fd1a174bb91f))
* Added the geometric operations to BoundingBox ([#136](https://github.com/docling-project/docling-core/issues/136)) ([`f02bbae`](https://github.com/docling-project/docling-core/commit/f02bbaea47ebbfe98265f530b0b62dd2a6ac1ecd))

## [v2.15.1](https://github.com/docling-project/docling-core/releases/tag/v2.15.1) - 2025-01-21

### Fix

* Backward compatible add_text() ([#132](https://github.com/docling-project/docling-core/issues/132)) ([`7e45817`](https://github.com/docling-project/docling-core/commit/7e458179d8ec46017fd90114a55360daf419f926))

## [v2.15.0](https://github.com/docling-project/docling-core/releases/tag/v2.15.0) - 2025-01-21

### Feature

* Add CodeItem as pydantic type, update export methods and APIs ([#129](https://github.com/docling-project/docling-core/issues/129)) ([`c940aa5`](https://github.com/docling-project/docling-core/commit/c940aa5ca9b345333e3e95d8c0ec32ddfa227385))

### Fix

* Fix hybrid chunker token constraint ([#131](https://github.com/docling-project/docling-core/issues/131)) ([`b741eea`](https://github.com/docling-project/docling-core/commit/b741eeaab437781e36f9d356478ef525ef54867b))
* Always return a new bbox when changing origin ([#128](https://github.com/docling-project/docling-core/issues/128)) ([`841668f`](https://github.com/docling-project/docling-core/commit/841668f416f2079afc6f8ab07e5507aacce59de3))

## [v2.14.0](https://github.com/docling-project/docling-core/releases/tag/v2.14.0) - 2025-01-10

### Feature

* Dev/add labels for pictures-classes ([#127](https://github.com/docling-project/docling-core/issues/127)) ([`078cd61`](https://github.com/docling-project/docling-core/commit/078cd61b31c36bec553f64c411012e361683bd35))

## [v2.13.1](https://github.com/docling-project/docling-core/releases/tag/v2.13.1) - 2025-01-08

### Fix

* Restore proper string serialization of DocItemLabel ([#124](https://github.com/docling-project/docling-core/issues/124)) ([`a52bb88`](https://github.com/docling-project/docling-core/commit/a52bb88f78146a5777246d3fc04b04d0db1c1631))

## [v2.13.0](https://github.com/docling-project/docling-core/releases/tag/v2.13.0) - 2025-01-08

### Feature

* Add mapping to colors into DocItemLabel ([#123](https://github.com/docling-project/docling-core/issues/123)) ([`639f122`](https://github.com/docling-project/docling-core/commit/639f12226d4d413c5f95dc4989391a209cca1ae6))

### Fix

* Quote referenced URIs in markdown and html ([#122](https://github.com/docling-project/docling-core/issues/122)) ([`127dd2f`](https://github.com/docling-project/docling-core/commit/127dd2f6f8862e2c74f821cdb3a1995ee0a243cc))

## [v2.12.1](https://github.com/docling-project/docling-core/releases/tag/v2.12.1) - 2024-12-17

### Fix

* Fixes for legacy-doc handling ([#115](https://github.com/docling-project/docling-core/issues/115)) ([`b116c46`](https://github.com/docling-project/docling-core/commit/b116c465a2af6327cffa95de0745506404cb39f9))

## [v2.12.0](https://github.com/docling-project/docling-core/releases/tag/v2.12.0) - 2024-12-17

### Feature

* Added the new label comment_section in the groups ([#114](https://github.com/docling-project/docling-core/issues/114)) ([`5101dd8`](https://github.com/docling-project/docling-core/commit/5101dd8845dcfc098c7009556e7468478393ea5e))

### Fix

* Skip labels not included in the allow-list ([#113](https://github.com/docling-project/docling-core/issues/113)) ([`d147c25`](https://github.com/docling-project/docling-core/commit/d147c2565f635e851b73cd6b97cc78617372b57f))
* Always write with utf8 encoding ([#111](https://github.com/docling-project/docling-core/issues/111)) ([`268c294`](https://github.com/docling-project/docling-core/commit/268c294cc95abb36fc491521e39c7bf6e6a45abc))

## [v2.11.0](https://github.com/docling-project/docling-core/releases/tag/v2.11.0) - 2024-12-16

### Feature

* Add group labels for form and key-value areas ([#110](https://github.com/docling-project/docling-core/issues/110)) ([`aeaf89d`](https://github.com/docling-project/docling-core/commit/aeaf89de106201c53066c16963a9d6ba4467e51c))

## [v2.10.0](https://github.com/docling-project/docling-core/releases/tag/v2.10.0) - 2024-12-13

### Feature

* Add legacy to DoclingDocument utility ([#108](https://github.com/docling-project/docling-core/issues/108)) ([`b31e0a3`](https://github.com/docling-project/docling-core/commit/b31e0a3d05cbcf450a4d287484761fc447d3e2ec))
* Add DoclingDocument viewer to CLI ([#99](https://github.com/docling-project/docling-core/issues/99)) ([`9628d19`](https://github.com/docling-project/docling-core/commit/9628d19c24ed92834636973c9e272c9a21864604))
* Add default tokenizer to HybridChunker ([#107](https://github.com/docling-project/docling-core/issues/107)) ([`2591c70`](https://github.com/docling-project/docling-core/commit/2591c70c66d1615050f47d045c4bc6092f99ebad))

### Fix

* Improve doc item typing ([#105](https://github.com/docling-project/docling-core/issues/105)) ([`047a196`](https://github.com/docling-project/docling-core/commit/047a1960afbaed4613b8d305d8dff4988a97c2d9))
* Set origin when merging chunks ([#109](https://github.com/docling-project/docling-core/issues/109)) ([`b546c0a`](https://github.com/docling-project/docling-core/commit/b546c0a50d11152f0ad65a1bc59e33478bc11052))
* Add REFERENCE to exported labels and remove CAPTION ([#106](https://github.com/docling-project/docling-core/issues/106)) ([`a66b0bb`](https://github.com/docling-project/docling-core/commit/a66b0bb6f8d821bbad738c1bd9fc52317304579f))

## [v2.9.0](https://github.com/docling-project/docling-core/releases/tag/v2.9.0) - 2024-12-09

### Feature

* Utilities converting document formats ([#91](https://github.com/docling-project/docling-core/issues/91)) ([`437c498`](https://github.com/docling-project/docling-core/commit/437c498f77c71cce49c139a25e0803acde429b90))

### Fix

* **markdown:** Preserve underscores in image URLs during markdown export ([#98](https://github.com/docling-project/docling-core/issues/98)) ([`fd7529f`](https://github.com/docling-project/docling-core/commit/fd7529f4096ecc5c7809b884c5fef2df9818801f))

## [v2.8.0](https://github.com/docling-project/docling-core/releases/tag/v2.8.0) - 2024-12-06

### Feature

* Add hybrid chunker ([#68](https://github.com/docling-project/docling-core/issues/68)) ([`628ab67`](https://github.com/docling-project/docling-core/commit/628ab679cbcbf4d708619111cd391ff62dc9d080))

## [v2.7.1](https://github.com/docling-project/docling-core/releases/tag/v2.7.1) - 2024-12-06

### Fix

* Multimodal output ([#96](https://github.com/docling-project/docling-core/issues/96)) ([`2133af6`](https://github.com/docling-project/docling-core/commit/2133af61121d202a4df04ef9a28308b82d7c87cb))

## [v2.7.0](https://github.com/docling-project/docling-core/releases/tag/v2.7.0) - 2024-12-04

### Feature

* Export to OTSL method for docling doc tables ([#86](https://github.com/docling-project/docling-core/issues/86)) ([`180e294`](https://github.com/docling-project/docling-core/commit/180e294aada4d97ceeb61556b5f7f310bd078c5f))

## [v2.6.1](https://github.com/docling-project/docling-core/releases/tag/v2.6.1) - 2024-12-02

### Fix

* Fix circular import ([#87](https://github.com/docling-project/docling-core/issues/87)) ([`63e6c01`](https://github.com/docling-project/docling-core/commit/63e6c01863cbf8d71e1636f3dca8226f6bdf63e2))

## [v2.6.0](https://github.com/docling-project/docling-core/releases/tag/v2.6.0) - 2024-12-02

### Feature

* Extend source resolution with streams and workdir ([#79](https://github.com/docling-project/docling-core/issues/79)) ([`9a74d13`](https://github.com/docling-project/docling-core/commit/9a74d13fd60334bd0a4b4687fd5deaaf79b89001))
* Simple method to load DoclingDocument from .json files ([#71](https://github.com/docling-project/docling-core/issues/71)) ([`fc1cfb0`](https://github.com/docling-project/docling-core/commit/fc1cfb0fe02914f7c86e357909221ab143d74d4c))

### Fix

* Allow all url types in referenced exports ([#82](https://github.com/docling-project/docling-core/issues/82)) ([`3bd83bc`](https://github.com/docling-project/docling-core/commit/3bd83bcfc401f09ed07e7c7f20c51403bf6484d6))
* Even better style for HTML export ([#78](https://github.com/docling-project/docling-core/issues/78)) ([`8422ad4`](https://github.com/docling-project/docling-core/commit/8422ad4fbf6ed8372815e0fa9393951f04a759d5))

## [v2.5.1](https://github.com/docling-project/docling-core/releases/tag/v2.5.1) - 2024-11-27

### Fix

* Hotfix for TableItem.export_to_html args ([#76](https://github.com/docling-project/docling-core/issues/76)) ([`ae2f131`](https://github.com/docling-project/docling-core/commit/ae2f1317255938d102c716efaf2db6adbc724bd1))
* Artifacts dir double stem ([#75](https://github.com/docling-project/docling-core/issues/75)) ([`f93332b`](https://github.com/docling-project/docling-core/commit/f93332b402c1e175a2df3cad7c254b23591eea34))

## [v2.5.0](https://github.com/docling-project/docling-core/releases/tag/v2.5.0) - 2024-11-27

### Feature

* Adding HTML export to DoclingDocument, adding export of images in png with links to Markdown & HTML ([#69](https://github.com/docling-project/docling-core/issues/69)) ([`ef49fd3`](https://github.com/docling-project/docling-core/commit/ef49fd3f34ce140af20dff8cf48676df20a9502e))

## [v2.4.1](https://github.com/docling-project/docling-core/releases/tag/v2.4.1) - 2024-11-21

### Fix

* Temporarily force pydantic < 2.10 ([#70](https://github.com/docling-project/docling-core/issues/70)) ([`289b629`](https://github.com/docling-project/docling-core/commit/289b629dc9451678885ae30bbcc286290bed5d87))

## [v2.4.0](https://github.com/docling-project/docling-core/releases/tag/v2.4.0) - 2024-11-18

### Feature

* Add get_image for all DocItem ([#67](https://github.com/docling-project/docling-core/issues/67)) ([`9d7e831`](https://github.com/docling-project/docling-core/commit/9d7e831fb23c5069361bcb6be8d562f36393398b))
* Allow exporting a specific page to md. ([#63](https://github.com/docling-project/docling-core/issues/63)) ([`1a201bc`](https://github.com/docling-project/docling-core/commit/1a201bc65a32c88bae705b65998ab486b8a4302d))

## [v2.3.2](https://github.com/docling-project/docling-core/releases/tag/v2.3.2) - 2024-11-11

### Fix

* Fixed selection logic for a slice of the document ([#66](https://github.com/docling-project/docling-core/issues/66)) ([`dfdc76b`](https://github.com/docling-project/docling-core/commit/dfdc76bf55a1442d4321b577904c0b4748158b55))

## [v2.3.1](https://github.com/docling-project/docling-core/releases/tag/v2.3.1) - 2024-11-01

### Fix

* Include titles to chunk heading metadata ([#62](https://github.com/docling-project/docling-core/issues/62)) ([`bfeb2db`](https://github.com/docling-project/docling-core/commit/bfeb2db24b70550693911af6aee01db8c74d464a))

## [v2.3.0](https://github.com/docling-project/docling-core/releases/tag/v2.3.0) - 2024-10-29

### Feature

* Added pydantic models to store charts data (pie, bar, stacked bar, line, scatter) ([#52](https://github.com/docling-project/docling-core/issues/52)) ([`36b7bea`](https://github.com/docling-project/docling-core/commit/36b7bea53a291fdd8ffa5fc6cdbe4256d16c8710))

## [v2.2.3](https://github.com/docling-project/docling-core/releases/tag/v2.2.3) - 2024-10-29

### Fix

* Str representation of enum across python versions ([#60](https://github.com/docling-project/docling-core/issues/60)) ([`8528918`](https://github.com/docling-project/docling-core/commit/8528918ae24eb1e50b97935b1136e8e8e2d9717b))
* Title for export to markdown and add text_width parameter ([#59](https://github.com/docling-project/docling-core/issues/59)) ([`4993c34`](https://github.com/docling-project/docling-core/commit/4993c3403d0dac869033c38349b785e5c78ac1d9))

## [v2.2.2](https://github.com/docling-project/docling-core/releases/tag/v2.2.2) - 2024-10-26

### Fix

* Fix non-string table cell handling in chunker ([#58](https://github.com/docling-project/docling-core/issues/58)) ([`b5d07b2`](https://github.com/docling-project/docling-core/commit/b5d07b2fa5e865939b9b93a8582eb3207bc48249))

## [v2.2.1](https://github.com/docling-project/docling-core/releases/tag/v2.2.1) - 2024-10-25

### Fix

* Escaping underscore characters in md export ([#57](https://github.com/docling-project/docling-core/issues/57)) ([`c344d0f`](https://github.com/docling-project/docling-core/commit/c344d0fc63ac55068dac5846772e3369f71c771b))

## [v2.2.0](https://github.com/docling-project/docling-core/releases/tag/v2.2.0) - 2024-10-24

### Feature

* Add headers argument and a custom user-agents for http requests ([#53](https://github.com/docling-project/docling-core/issues/53)) ([`44941b5`](https://github.com/docling-project/docling-core/commit/44941b5591fe7db5a2b987f5d2b35785fab386db))

### Fix

* Fix resolution in case of URL without path ([#55](https://github.com/docling-project/docling-core/issues/55)) ([`2c88e56`](https://github.com/docling-project/docling-core/commit/2c88e56f8ccb5902457c1749c64c8de6cc963739))

## [v2.1.0](https://github.com/docling-project/docling-core/releases/tag/v2.1.0) - 2024-10-22

### Feature

* Improve markdown export of DoclingDocument ([#50](https://github.com/docling-project/docling-core/issues/50)) ([`328778e`](https://github.com/docling-project/docling-core/commit/328778ed036540ee5fdf0bb16a1b656e5122c5f0))
* Extend chunk meta with schema, version, origin ([#49](https://github.com/docling-project/docling-core/issues/49)) ([`d09fe7e`](https://github.com/docling-project/docling-core/commit/d09fe7ed44282b286f9c2588482e515bf40e0fca))

## [v2.0.1](https://github.com/docling-project/docling-core/releases/tag/v2.0.1) - 2024-10-18

### Fix

* Fix legacy doc ref ([#48](https://github.com/docling-project/docling-core/issues/48)) ([`e12d6a7`](https://github.com/docling-project/docling-core/commit/e12d6a70c383a8d3f0c3d73aa6c5eec62a0c3251))
* Add-mimetype-for-asciidocs-markdown ([#47](https://github.com/docling-project/docling-core/issues/47)) ([`0aab007`](https://github.com/docling-project/docling-core/commit/0aab0073bbeabe62b2a19e872e108438c199f6c0))

## [v2.0.0](https://github.com/docling-project/docling-core/releases/tag/v2.0.0) - 2024-10-16

### Feature

* Expose DoclingDocument as main type, move old typing to legacy ([#41](https://github.com/docling-project/docling-core/issues/41)) ([`03df97f`](https://github.com/docling-project/docling-core/commit/03df97fa73db2499682613ff17dff9fe996a1bdc))

### Breaking

* Expose DoclingDocument as main type, move old typing to legacy ([#41](https://github.com/docling-project/docling-core/issues/41)) ([`03df97f`](https://github.com/docling-project/docling-core/commit/03df97fa73db2499682613ff17dff9fe996a1bdc))

## [v1.7.2](https://github.com/docling-project/docling-core/releases/tag/v1.7.2) - 2024-10-09

### Fix

* Loosen pandas version ([#40](https://github.com/docling-project/docling-core/issues/40)) ([`aec1a41`](https://github.com/docling-project/docling-core/commit/aec1a41402d64c2c216923b40d4521f8d46540b7))

## [v1.7.1](https://github.com/docling-project/docling-core/releases/tag/v1.7.1) - 2024-10-07

### Fix

* Make doc metadata keys pure strings ([#38](https://github.com/docling-project/docling-core/issues/38)) ([`246627f`](https://github.com/docling-project/docling-core/commit/246627f4f6aef1121dd4211cc223f356a133c60e))
* Align chunk ref format with one used in Document ([#37](https://github.com/docling-project/docling-core/issues/37)) ([`b5592ad`](https://github.com/docling-project/docling-core/commit/b5592ad747a061cb6b47c86228063371a80a9b44))

## [v1.7.0](https://github.com/docling-project/docling-core/releases/tag/v1.7.0) - 2024-10-01

### Feature

* (experimental) introduce new document format ([#21](https://github.com/docling-project/docling-core/issues/21)) ([`688789e`](https://github.com/docling-project/docling-core/commit/688789ea751d75c15a6957dba4ba496b899e9d11))
* Add doc metadata extractor and ID generator classes ([#34](https://github.com/docling-project/docling-core/issues/34)) ([`b76780c`](https://github.com/docling-project/docling-core/commit/b76780c3b21a89d407b6afb5e72cd4f46dbcf569))
* Support heading as chunk metadata ([#36](https://github.com/docling-project/docling-core/issues/36)) ([`4bde515`](https://github.com/docling-project/docling-core/commit/4bde51528d23be9bed797030a75991f6acdb241f))

## [v1.6.3](https://github.com/docling-project/docling-core/releases/tag/v1.6.3) - 2024-09-26

### Fix

* Change order of JSON Schema to search mapper transformations ([#32](https://github.com/docling-project/docling-core/issues/32)) ([`a4ddd14`](https://github.com/docling-project/docling-core/commit/a4ddd142eef864c55b62c8815d38dbff14f4caa7))

## [v1.6.2](https://github.com/docling-project/docling-core/releases/tag/v1.6.2) - 2024-09-24

### Fix

* Remove duplicate captions in markdown ([#31](https://github.com/docling-project/docling-core/issues/31)) ([`a334b9f`](https://github.com/docling-project/docling-core/commit/a334b9fc721a2e1efc9f12b585cff17363875d57))

## [v1.6.1](https://github.com/docling-project/docling-core/releases/tag/v1.6.1) - 2024-09-24

### Fix

* Remove unnecessary package dependency ([#30](https://github.com/docling-project/docling-core/issues/30)) ([`e706d68`](https://github.com/docling-project/docling-core/commit/e706d686db159f6480439d214c85b1664f38e28f))

## [v1.6.0](https://github.com/docling-project/docling-core/releases/tag/v1.6.0) - 2024-09-23

### Feature

* Add figures in markdown export ([#27](https://github.com/docling-project/docling-core/issues/27)) ([`b843ae6`](https://github.com/docling-project/docling-core/commit/b843ae6688a20e68e2da59b2f68fd61f8d4beacb))

## [v1.5.0](https://github.com/docling-project/docling-core/releases/tag/v1.5.0) - 2024-09-20

### Feature

* Add export to doctags for document components ([#25](https://github.com/docling-project/docling-core/issues/25)) ([`891530f`](https://github.com/docling-project/docling-core/commit/891530f595dbf656bbc2708fb25a05aa1ec65afa))
* Add file source resolution utility ([#22](https://github.com/docling-project/docling-core/issues/22)) ([`752cbc3`](https://github.com/docling-project/docling-core/commit/752cbc3e89461fa633277cfe3887bc5a6fa5c2b0))

## [v1.4.1](https://github.com/docling-project/docling-core/releases/tag/v1.4.1) - 2024-09-18

### Fix

* Add export to xml and html ([#17](https://github.com/docling-project/docling-core/issues/17)) ([`9bc256e`](https://github.com/docling-project/docling-core/commit/9bc256e5bbbe02cc0a317bc2920c8e0becb3090c))

## [v1.4.0](https://github.com/docling-project/docling-core/releases/tag/v1.4.0) - 2024-09-18

### Feature

* Add table exporters ([#20](https://github.com/docling-project/docling-core/issues/20)) ([`2cc2429`](https://github.com/docling-project/docling-core/commit/2cc2429e2731998c3282ba133995439450f08574))

## [v1.3.0](https://github.com/docling-project/docling-core/releases/tag/v1.3.0) - 2024-09-11

### Feature

* Add hierarchical chunker ([#18](https://github.com/docling-project/docling-core/issues/18)) ([`9698d30`](https://github.com/docling-project/docling-core/commit/9698d30288df17ecde67f170848f1be47cd97d33))

## [v1.2.0](https://github.com/docling-project/docling-core/releases/tag/v1.2.0) - 2024-09-10

### Feature

* Added the XML export ([#16](https://github.com/docling-project/docling-core/issues/16)) ([`acdf816`](https://github.com/docling-project/docling-core/commit/acdf81608134c23969c9e620085f4fff4f42a12f))

## [v1.1.4](https://github.com/docling-project/docling-core/releases/tag/v1.1.4) - 2024-09-06

### Fix

* Validate_model() could be called with other types rather than dict ([#14](https://github.com/docling-project/docling-core/issues/14)) ([`235b2cd`](https://github.com/docling-project/docling-core/commit/235b2cd10b595c813c03db5b4effbc7cc2feaaf0))

### Documentation

* Update link to Docling ([#12](https://github.com/docling-project/docling-core/issues/12)) ([`aaf17fe`](https://github.com/docling-project/docling-core/commit/aaf17fe0f6eae7ee21c54c56fded05f24ec936b1))

## [v1.1.3](https://github.com/docling-project/docling-core/releases/tag/v1.1.3) - 2024-08-28

### Fix

* Use same base type for all components ([#10](https://github.com/docling-project/docling-core/issues/10)) ([`f450c8c`](https://github.com/docling-project/docling-core/commit/f450c8cbfd623bf5c7013bae956d23618004f43d))

## [v1.1.2](https://github.com/docling-project/docling-core/releases/tag/v1.1.2) - 2024-07-31

### Fix

* Make page number strictly positive ([#8](https://github.com/docling-project/docling-core/issues/8)) ([`ec3cff9`](https://github.com/docling-project/docling-core/commit/ec3cff97e5079251087cd7b4b42e8c509cd244f3))

## [v1.1.1](https://github.com/docling-project/docling-core/releases/tag/v1.1.1) - 2024-07-23

### Fix

* Set type to optional ([#7](https://github.com/docling-project/docling-core/issues/7)) ([`faf472c`](https://github.com/docling-project/docling-core/commit/faf472c1689746adc43e0ae8ef6d6e3fcf87c023))

### Documentation

* Revamp installation instructions ([#6](https://github.com/docling-project/docling-core/issues/6)) ([`3f77b2e`](https://github.com/docling-project/docling-core/commit/3f77b2e92c415c7290df8c4d534ba3455dbe62bd))

## [v1.1.0](https://github.com/docling-project/docling-core/releases/tag/v1.1.0) - 2024-07-18

### Feature

* Add document Markdown export ([#4](https://github.com/docling-project/docling-core/issues/4)) ([`d0ffc85`](https://github.com/docling-project/docling-core/commit/d0ffc85e0c2b49d201f5359c4dc4efb5cd5716b0))

## [v1.0.0](https://github.com/docling-project/docling-core/releases/tag/v1.0.0) - 2024-07-17

### Feature

* Trigger v1.0.0 release ([#3](https://github.com/docling-project/docling-core/issues/3)) ([`daece4c`](https://github.com/docling-project/docling-core/commit/daece4ceae363351072aa7e0adb91037e0dd7b66))

### Breaking

* trigger v1.0.0 release ([#3](https://github.com/docling-project/docling-core/issues/3)) ([`daece4c`](https://github.com/docling-project/docling-core/commit/daece4ceae363351072aa7e0adb91037e0dd7b66))

## [v0.0.1](https://github.com/docling-project/docling-core/releases/tag/v0.0.1) - 2024-07-17

### Fix

* Fix definition issues in record type ([#2](https://github.com/docling-project/docling-core/issues/2)) ([`656f563`](https://github.com/docling-project/docling-core/commit/656f56380f603c3de125f6c59554f26ac8cd0a78))
