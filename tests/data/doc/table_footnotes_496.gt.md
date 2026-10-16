## Structured Document Parsing Test

The following inventory report is designed to test how document parsers handle multi-line annotations that are geographically tied to a table structure.

| Record ID   | Asset Category   |   Unit Count | Status      |
|-------------|------------------|--------------|-------------|
| 440-A       | Hardware         |          142 | Operational |
| 440-B       | Software License |           12 | Pending*    |
| 441-C       | Cloud Instance   |            3 | Operational |
| 442-D       | Legacy Support   |            1 | Archived**  |

*Note: The 'Pending' status indicates that the asset is awaiting final security validation.

**Note: 'Archived' items are scheduled for decommissioning at the end of the current fiscal year.

This final paragraph provides a clear end-of-section marker. Parsers should correctly associate the two footnotes with the table above rather than this trailing body text.
