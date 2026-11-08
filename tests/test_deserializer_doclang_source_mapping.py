from docling_core.transforms.deserializer import DocLangSourceMap
from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer


def test_deserialize_collects_source_bindings_without_changing_document() -> None:
    xml = """<doclang>
  <heading level="2">Section</heading>
  <text><thread thread_id="7"/>first</text>
  <text><thread thread_id="7"/>second</text>
  <list><ldiv/><content>item</content></list>
  <table><ched/>Head<fcel/>Value<nl/></table>
</doclang>"""
    source_map = DocLangSourceMap()
    doc = DocLangDocDeserializer().deserialize_str(xml, source_map=source_map)

    assert doc == DocLangDocDeserializer().deserialize_str(xml)
    first_thread = source_map.targets_by_xpath["/d:doclang/d:text[1]"]
    assert first_thread == source_map.targets_by_xpath["/d:doclang/d:text[2]"]
    assert source_map.targets_by_xpath["/d:doclang/d:list[1]/d:ldiv[1]"].kind == "item"
    assert source_map.targets_by_xpath["/d:doclang/d:table[1]/d:ched[1]"].row == 0
    assert source_map.targets_by_xpath["/d:doclang/d:table[1]/d:fcel[1]"].col == 1

    bindings = dict(source_map.targets_by_xpath)
    doc._hierarchize()
    assert source_map.targets_by_xpath == bindings
