from lxml import etree

# Parse the XML file
tree = etree.parse("input/SIKOR/SIKOR_sme_20151010/SIKOR_free_sme_20151010.xml")
root = tree.getroot()

# Count tags
subcorpus_count = len(root.xpath("//subcorpus"))
text_count = len(root.xpath("//text"))
sentence_count = len(root.xpath("//sentence"))

print(f"Subcorpus tags: {subcorpus_count}")
print(f"Text tags: {text_count}")
print(f"Sentence tags: {sentence_count}")

sentence_words = {
    sentence.get("id"): sentence.xpath(".//w/@word") for sentence in root.xpath("//sentence")
}
print(sentence_words[list(sentence_words)[0]])