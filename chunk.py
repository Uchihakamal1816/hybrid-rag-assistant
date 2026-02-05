from langchain_text_splitters import RecursiveCharacterTextSplitter

from loader import load_all_acts

docs = load_all_acts()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,     
    chunk_overlap=150  
)

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))
print("\nSample chunk:\n", chunks[0].page_content[:300])
print("\nMetadata:", chunks[0].metadata)
