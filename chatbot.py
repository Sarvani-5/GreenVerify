import json
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
import re

class GreenVerifyRAGBot:
    def __init__(self, gemini_api_key: str, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize RAG system with FAISS vector store and Gemini
        
        Args:
            gemini_api_key: Your Google Gemini API key
            model_name: Sentence transformer model for embeddings
        """
        # Initialize Gemini
        gemini_api_key = "AIzaSyA49YQ1ZwIGQp0tMdaouunx9A04F9Qn2O0"
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
        
        # Initialize sentence transformer
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.faiss_index = None
        self.documents = []
        self.chunk_metadata = []
        
    def load_json_documents(self, json_files: List[str]):
        """Load JSON documents from web crawler output"""
        print("Loading JSON documents...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    self._process_document(data)
                elif isinstance(data, list):
                    for doc in data:
                        self._process_document(doc)
                        
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(self.documents)} documents")
    
    def _process_document(self, doc_data: Dict):
        """Process individual document and create chunks"""
        title = doc_data.get('title', 'Untitled')
        url = doc_data.get('url', '')
        text_content = doc_data.get('text_content', '')
        headings = doc_data.get('headings', {})
        meta_description = doc_data.get('meta_description', '')
        
        # Combine headings into text
        heading_text = ""
        for level, heading_list in headings.items():
            if heading_list:
                heading_text += " ".join(heading_list) + " "
        
        # Create metadata-rich chunks
        chunks = self._create_smart_chunks(text_content, title, heading_text, meta_description)
        
        doc_info = {
            'title': title,
            'url': url,
            'chunks': chunks,
            'heading_text': heading_text,
            'meta_description': meta_description,
            'full_text': text_content
        }
        
        self.documents.append(doc_info)
    
    def _create_smart_chunks(self, text: str, title: str, headings: str, meta: str, 
                           chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Create intelligent chunks with context"""
        if not text:
            return []
        
        # Add context to each chunk
        context_header = f"Document: {title}\nKey Topics: {headings}\nSummary: {meta}\n\nContent:\n"
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = context_header
        current_size = len(current_chunk)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence exceeds chunk size
            if current_size + len(sentence) > chunk_size and len(current_chunk) > len(context_header):
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                current_chunk = context_header + sentence + " "
                current_size = len(current_chunk)
            else:
                current_chunk += sentence + " "
                current_size += len(sentence) + 1
        
        # Add remaining content
        if len(current_chunk) > len(context_header):
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def build_faiss_index(self):
        """Build FAISS vector index from document chunks"""
        print("Creating embeddings and building FAISS index...")
        
        all_chunks = []
        
        for doc_idx, doc in enumerate(self.documents):
            for chunk_idx, chunk in enumerate(doc['chunks']):
                all_chunks.append(chunk)
                self.chunk_metadata.append({
                    'doc_idx': doc_idx,
                    'chunk_idx': chunk_idx,
                    'title': doc['title'],
                    'url': doc['url']
                })
        
        # Create embeddings
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Initialize FAISS index (using IndexFlatIP for cosine similarity)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.faiss_index.add(embeddings)
        
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents using FAISS"""
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index() first.")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        similarities, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            metadata = self.chunk_metadata[idx]
            doc = self.documents[metadata['doc_idx']]
            chunk = doc['chunks'][metadata['chunk_idx']]
            
            results.append({
                'content': chunk,
                'similarity': float(similarity),
                'title': metadata['title'],
                'url': metadata['url'],
                'rank': i + 1
            })
        
        return results
    
    def generate_response(self, query: str, max_context_length: int = 3000, 
                         similarity_threshold: float = 0.3) -> str:
        """Generate response using Gemini with retrieved context"""
        # Get relevant documents
        relevant_docs = self.search_documents(query, top_k=3)
        
        # Check if we have good quality matches
        has_relevant_content = any(doc['similarity'] > similarity_threshold for doc in relevant_docs)
        
        if not relevant_docs or not has_relevant_content:
            return f"""I couldn't find specific information about your query in my current knowledge base. 

For detailed and up-to-date information about green building certifications, GRIHA ratings, and sustainable construction practices, please visit the official GRIHA India website: https://www.grihaindia.org/

You can find comprehensive information about:
- GRIHA rating criteria and processes
- Eligibility requirements
- Application procedures
- Certification costs and timelines
- Technical guidelines and standards

Is there anything else I can help you with from the available documentation?"""
        
        # Prepare context
        context = ""
        for doc in relevant_docs:
            if doc['similarity'] > similarity_threshold:
                context += f"Source: {doc['title']}\nContent: {doc['content']}\n\n"
        
        # Truncate context if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Create enhanced prompt for Gemini
        prompt = f"""You are a Green Verify chatbot assistant. Use the provided context to answer the user's question about green building certifications, GRIHA ratings, and sustainable construction practices.

Context from Green Verify documentation:
{context}

User Question: {query}

Instructions:
1. Answer based primarily on the provided context
2. Be specific and accurate
3. If the context doesn't contain complete information, provide what you can and then mention: "For more detailed information, please visit the official GRIHA India website: https://www.grihaindia.org/"
4. Focus on green building certifications, sustainability criteria, and verification processes
5. Provide actionable information when possible
6. If the user needs official forms, applications, or the latest updates, always refer them to https://www.grihaindia.org/
7. Be helpful but acknowledge the limitations of your current knowledge base

Answer:"""

        try:
            # Generate response with Gemini
            response = self.gemini_model.generate_content(prompt)
            
            # Add website reference if response seems incomplete or user might need more info
            response_text = response.text
            if any(keyword in query.lower() for keyword in ['apply', 'application', 'form', 'register', 'contact', 'latest', 'current', 'fee', 'cost', 'price']):
                response_text += f"\n\n💡 For the most current information, official forms, and to begin the application process, please visit: https://www.grihaindia.org/"
            
            return response_text
            
        except Exception as e:
            print(f"Gemini API Error: {str(e)}")
            # Fallback to context-based response if Gemini fails
            if context:
                fallback_response = f"""Based on the available information in our knowledge base:

{context[:1500]}...

Note: This is a direct extract from our documentation. For complete and up-to-date information, please visit the official GRIHA India website: https://www.grihaindia.org/

Error details: {str(e)}"""
                return fallback_response
            else:
                return f"""I encountered an error accessing the AI model: {str(e)}

For reliable information about green building certifications and GRIHA ratings, please visit the official GRIHA India website: https://www.grihaindia.org/

You can also try:
1. Using different keywords in your question
2. Asking more specific questions about GRIHA ratings
3. Checking if your internet connection is stable"""
    
    def chat(self):
        """Interactive chat interface"""
        print("🌱 Green Verify RAG Chatbot")
        print("Ask me anything about green building certifications and GRIHA ratings!")
        print("💡 For official information and applications, visit: https://www.grihaindia.org/")
        print("Type 'quit' to exit\n")
        
        while True:
            query = input("You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! 🌱")
                print("Remember to visit https://www.grihaindia.org/ for official GRIHA information!")
                break
            
            if not query:
                continue
            
            print("\n🤖 Searching knowledge base...")
            response = self.generate_response(query)
            print(f"Bot: {response}\n")
            print("-" * 50)
    
    def save_index(self, filepath: str):
        """Save FAISS index and metadata"""
        if self.faiss_index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, f"{filepath}.faiss")
        
        # Save metadata
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'chunk_metadata': self.chunk_metadata,
                'documents': self.documents
            }, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load FAISS index and metadata"""
        # Load FAISS index
        self.faiss_index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunk_metadata = data['chunk_metadata']
            self.documents = data['documents']
        
        print(f"Index loaded from {filepath}")

# Example usage and setup
def main():
    # Initialize the RAG bot
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # Replace with your actual API key
    
    # Check available models first
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        models = genai.list_models()
        print("Available Gemini models:")
        for model in models:
            print(f"- {model.name}")
        print()
    except Exception as e:
        print(f"Error checking models: {e}")
    
    rag_bot = GreenVerifyRAGBot(
        gemini_api_key=GEMINI_API_KEY,
        model_name='all-MiniLM-L6-v2'
    )
    
    # Load JSON files from web crawler
    json_files = ['crawled_data.json']  # Add your JSON file paths here
    rag_bot.load_json_documents(json_files)
    
    # Build FAISS index
    rag_bot.build_faiss_index()
    
    # Save index for future use
    # rag_bot.save_index('green_verify_index')
    
    # Test queries
    test_queries = [
        "What are GRIHA rating criteria?",
        "How do I get green building certification?",
        "What is the eligibility for GRIHA rating?",
        "Tell me about sustainable building practices",
        "What are the costs involved in green certification?"
    ]
    
    print("Testing the RAG system:")
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        response = rag_bot.generate_response(query)
        print(f"🤖 Response: {response}")
        print("-" * 80)
    
    # Start interactive chat
    rag_bot.chat()

if __name__ == "__main__":
    # Requirements to install:
    # pip install faiss-cpu sentence-transformers google-generativeai numpy
    main()