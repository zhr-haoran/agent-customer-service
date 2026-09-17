import uuid
import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, collection_name="knowledge_base",
                 model_path=r'D:\LLM\Local_model\models\BAAI--bge-large-zh-v1.5'):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model = SentenceTransformer(model_path)

    def add_documents(self, texts):
        """把文本列表转成向量并存入向量库"""
        embeddings = self.model.encode(texts).tolist()
        # 修复：用 uuid 生成唯一 ID，避免重复上传时报 DuplicatedIDError
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        self.collection.add(documents=texts, embeddings=embeddings, ids=ids)

    def search(self, query, n_results=5):
        query_vec = self.model.encode([query]).tolist()
        result = self.collection.query(query_embeddings=query_vec, n_results=n_results)
        return result['documents'][0]