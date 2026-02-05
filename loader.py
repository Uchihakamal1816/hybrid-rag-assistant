from langchain_core.documents import Document
import os

ACT_DIR = "/home/uchihakamal/Desktop/ai and ml/project/agora/fulltext/"
def load_all_acts():
    documents = []

    for filename in os.listdir(ACT_DIR):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(ACT_DIR, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        agora_id = filename.replace(".txt", "")

        documents.append(
            Document(
                page_content=text,
                metadata={"agora_id": agora_id}
            )
        )

    return documents


docs = load_all_acts()
print("Total acts loaded:", len(docs))
print(docs[0].metadata)

