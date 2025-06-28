import pypdf
import sqlite3
import os
from typing import List, Dict
import streamlit as st

class PDFProcessor:
    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                page_count INTEGER,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_text_from_pdf(self, pdf_file) -> Dict:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = pypdf.PdfReader(pdf_file)
            text_content = ""
            page_count = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                
                # Add extraction information
                if not page_text.strip():
                    page_text = f"[No text found on page {page_num + 1} - may contain only images]"
                
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page_text
                text_content += f"\n[Text length: {len(page_text)} characters]\n"
            
            # Check total content
            if not text_content.strip():
                text_content = "[PDF contains no extractable text - may be a scanned document]"
            
            return {
                "success": True,
                "content": text_content,
                "page_count": page_count,
                "filename": pdf_file.name,
                "total_chars": len(text_content)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "filename": pdf_file.name
            }
    
    def save_document_to_db(self, filename: str, content: str, page_count: int) -> bool:
        """Save extracted text to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO documents (filename, content, page_count)
                VALUES (?, ?, ?)
            ''', (filename, content, page_count))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error saving to database: {str(e)}")
            return False
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, filename, content, page_count, upload_date
                FROM documents
                ORDER BY upload_date DESC
            ''')
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    "id": row[0],
                    "filename": row[1],
                    "content": row[2],
                    "page_count": row[3],
                    "upload_date": row[4]
                })
            
            conn.close()
            return documents
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def search_documents(self, query: str) -> List[Dict]:
        """Search documents by content"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, filename, content, page_count, upload_date
                FROM documents
                WHERE content LIKE ?
                ORDER BY upload_date DESC
            ''', (f'%{query}%',))
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    "id": row[0],
                    "filename": row[1],
                    "content": row[2],
                    "page_count": row[3],
                    "upload_date": row[4]
                })
            
            conn.close()
            return documents
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return [] 