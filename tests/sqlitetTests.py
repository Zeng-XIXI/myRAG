import chromadb

db = chromadb.PersistentClient(path="../storage")
collection = db.get_collection("knowledge", embedding_function=None)

data = collection.get(include=["documents", "metadatas"])

print("记录总数：", len(data["ids"]),"查看数据库里的全部文本片段")

for record_id, document, metadata in zip(
    data["ids"], data["documents"], data["metadatas"]
):
    print("\n" + "=" * 60)
    print("ID：", record_id)
    print("来源：", metadata.get("source"))
    print("片段：", metadata.get("chunk"))
    print("内容：\n", document)


print("查看集合和模型信息")
for collection in db.list_collections():
    print("集合名称：", collection.name)
    print("记录数量：", collection.count())
    print("元数据：", collection.metadata)


print("查看实际向量")
print("向量一般有上千个浮点数，不适合完整打印。可以只查看维度和前几个数值：")
data = collection.get(
    limit=3,
    include=["documents", "metadatas", "embeddings"],
)
for document, metadata, embedding in zip(
    data["documents"], data["metadatas"], data["embeddings"]
):
    print("\n来源：", metadata)
    print("文本：", document[:100])
    print("向量维度：", len(embedding))
    print("向量前10项：", embedding[:10])
print(collection)