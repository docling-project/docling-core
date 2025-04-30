"""HTML styles for different export modes."""


def _get_css_with_no_styling() -> str:
    """Return default CSS styles for the HTML document."""
    return "<style></style>"


def _get_css_for_split_page() -> str:
    """Return default CSS styles for the HTML document."""
    return """<style>
    html {
        background-color: #e1e1e1;
        font-family: Arial, sans-serif;
        line-height: 1.6;
    }
    img {
        min-width: 500px;
        max-width: 100%;
    }
    table {
        border-collapse: collapse;
        border: 0px solid #fff;
        width: 100%;
    }
    td {
        vertical-align: top;
    }
    .page {
        background-color: white;
        margin-top:15px;
        padding: 30px;
        border: 1px solid black;
        width:100%;
        max-width:1000px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    .page figure {
        text-align: center;
    }
    .page img {
        max-width: 900px;
        min-width: auto;
    }
    .page table {
        border-collapse: collapse;
        margin: 1em 0;
        width: 100%;
    }
    .page table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .page table th {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
        background-color: #f2f2f2;
        font-weight: bold;
    }
    .page table caption {
        color: #666;
        font-style: italic;
        margin-top: 0.5em;
        padding: 8px;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .page figcaption {
        color: #666;
        font-style: italic;
        margin-top: 0.5em;
        padding: 8px;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    code {
        background-color: rgb(228, 228, 228);
        border: 1px solid darkgray;
        padding: 10px;
        display: inline-block;
        font-family: monospace;
        max-width:980px;
        word-wrap: normal;
        white-space: pre-wrap;
        word-wrap: break-word;
        /*overflow-wrap: break-word;*/
    }
</style>
"""


def _get_css_for_single_column() -> str:
    """Return CSS styles for the single-column HTML document."""
    return """<style>
    html {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
        line-height: 1.6;
    }
    body {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        background-color: white;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    h1 {
        font-size: 2em;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.3em;
    }
    table {
        border-collapse: collapse;
        margin: 1em 0;
        width: 100%;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    figure {
        margin: 1.5em 0;
        text-align: center;
    }
    figcaption {
        color: #666;
        font-style: italic;
        margin-top: 0.5em;
    }
    img {
        max-width: 100%;
        height: auto;
    }
    pre {
        background-color: #f6f8fa;
        border-radius: 3px;
        padding: 1em;
        overflow: auto;
    }
    code {
        font-family: monospace;
        background-color: #f6f8fa;
        padding: 0.2em 0.4em;
        border-radius: 3px;
    }
    pre code {
        background-color: transparent;
        padding: 0;
    }
    .formula {
        text-align: center;
        padding: 0.5em;
        margin: 1em 0;
        background-color: #f9f9f9;
    }
    .formula-not-decoded {
        text-align: center;
        padding: 0.5em;
        margin: 1em 0;
        background: repeating-linear-gradient(
            45deg,
            #f0f0f0,
            #f0f0f0 10px,
            #f9f9f9 10px,
            #f9f9f9 20px
        );
    }
    .page-break {
        page-break-after: always;
        border-top: 1px dashed #ccc;
        margin: 2em 0;
    }
    .key-value-region {
        background-color: #f9f9f9;
        padding: 1em;
        border-radius: 4px;
        margin: 1em 0;
    }
    .key-value-region dt {
        font-weight: bold;
    }
    .key-value-region dd {
        margin-left: 1em;
        margin-bottom: 0.5em;
    }
    .form-container {
        border: 1px solid #ddd;
        padding: 1em;
        border-radius: 4px;
        margin: 1em 0;
    }
    .form-item {
        margin-bottom: 0.5em;
    }
    .image-classification {
        font-size: 0.9em;
        color: #666;
        margin-top: 0.5em;
    }
</style>"""
