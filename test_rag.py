import asyncio
from services.ragservice import RagService
from database.vectorstore import vector_store
from langchain_core.documents import Document

# 1. Add some mock documents to vector_store with specific user_id
doc1 = Document(page_content="Apples are good for health.", metadata={"user_id": "alice"})
doc2 = Document(page_content="Bananas are rich in potassium.", metadata={"user_id": "bob"})

vector_store.add_documents([doc1, doc2])

service = RagService()

# 2. Retrieve for alice
docs_alice = service.retrieve_documents("What is good for health?", user_id="alice")
print(f"Alice's docs: {[d.page_content for d in docs_alice]}")

# 3. Retrieve for bob
docs_bob = service.retrieve_documents("What is rich in potassium?", user_id="bob")
print(f"Bob's docs: {[d.page_content for d in docs_bob]}")
