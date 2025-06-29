import numpy as np
import sqlite3
import pickle
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st

class RAGProcessor:
    def __init__(self, db_path: str = "documents.db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.index_path = "faiss_index.pkl"
        self.chunks_path = "document_chunks.pkl"
        self.init_rag_database()
        
    def init_rag_database(self):
        """Initialize RAG-specific database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER,
                page_number INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER,
                embedding_data BLOB,
                FOREIGN KEY (chunk_id) REFERENCES document_chunks (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def process_document_for_rag(self, document_id: int, content: str, page_count: int) -> bool:
        """Process document content for RAG: chunk, embed, and store"""
        try:

            chunks = self.chunk_text(content)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                cursor.execute('''
                    INSERT INTO document_chunks (document_id, chunk_text, chunk_index, page_number)
                    VALUES (?, ?, ?, ?)
                    ''', (document_id, chunk, i, 1))
                chunk_ids.append(cursor.lastrowid)
            
            embeddings = self.model.encode(chunks, show_progress_bar=True)
            
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                cursor.execute('''
                    INSERT INTO embeddings (chunk_id, embedding_data)
                    VALUES (?, ?)
                ''', (chunk_id, pickle.dumps(embedding)))
            
            conn.commit()
            conn.close()
            
            self.update_faiss_index()
            
            return True
            
        except Exception as e:
            st.error(f"Error processing document for RAG: {str(e)}")
            return False
    
    def update_faiss_index(self):
        """Update FAISS index with all embeddings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT e.embedding_data, c.chunk_text, c.id, d.filename
                FROM embeddings e
                JOIN document_chunks c ON e.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
            ''')
            
            results = cursor.fetchall()
            
            if not results:
                return
            
            embeddings = []
            chunk_texts = []
            chunk_ids = []
            filenames = []
            
            for embedding_data, chunk_text, chunk_id, filename in results:
                embedding = pickle.loads(embedding_data)
                embeddings.append(embedding)
                chunk_texts.append(chunk_text)
                chunk_ids.append(chunk_id)
                filenames.append(filename)
            
            embeddings = np.array(embeddings).astype('float32')
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
            faiss.write_index(index, "faiss_index.idx")
            
            metadata = {
                'chunk_texts': chunk_texts,
                'chunk_ids': chunk_ids,
                'filenames': filenames
            }
            
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            conn.close()
            
        except Exception as e:
            st.error(f"Error updating FAISS index: {str(e)}")
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search documents using semantic similarity"""
        try:

            if not os.path.exists("faiss_index.idx") or not os.path.exists(self.chunks_path):
                return []
            
            index = faiss.read_index("faiss_index.idx")
            with open(self.chunks_path, 'rb') as f:
                metadata = pickle.load(f)
            
            query_embedding = self.model.encode([query])
            
            scores, indices = index.search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(metadata['chunk_texts']):
                    results.append({
                        'chunk_text': metadata['chunk_texts'][idx],
                        'filename': metadata['filenames'][idx],
                        'chunk_id': metadata['chunk_ids'][idx],
                        'similarity_score': float(score),
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_rag_context(self, query: str, top_k: int = 3) -> str:
        """Get relevant context for RAG from search results"""
        results = self.search_documents(query, top_k)
        
        if not results:
            return ""
        
        context = "Relevant document excerpts:\n\n"
        for result in results:
            context += f"From {result['filename']} (similarity: {result['similarity_score']:.3f}):\n"
            context += f"{result['chunk_text']}\n\n"
        
        return context 