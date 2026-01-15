import os
import io
import json
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import chromadb
import PyPDF2
from groq import Groq


# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")


# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI(title="PDF RAG API")


# ---------------------------
# Enable CORS for React frontend
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Initialize Groq, ChromaDB, and SentenceTransformer model
# ---------------------------
groq_client = Groq(api_key=GROQ_KEY)

# Create chroma_db directory if it doesn't exist
os.makedirs("./chroma_db", exist_ok=True)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pdf_collection")
model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------
# PDF tracking file
# ---------------------------
TRACKING_FILE = "./chroma_db/indexed_pdfs.json"


def load_indexed_pdfs():
    """Load list of indexed PDFs from tracking file"""
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return []


def save_indexed_pdfs(pdf_list):
    """Save list of indexed PDFs to tracking file"""
    os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
    with open(TRACKING_FILE, 'w') as f:
        json.dump(pdf_list, f)


# ---------------------------
# Root endpoint
# ---------------------------
@app.get("/")
def root():
    """API status and statistics"""
    return {
        "status": "running",
        "total_chunks": collection.count(),
        "indexed_pdfs": len(load_indexed_pdfs())
    }


# ---------------------------
# Upload PDF endpoint
# ---------------------------
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    """Upload and index a PDF file"""
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        indexed_pdfs = load_indexed_pdfs()
        if file.filename in indexed_pdfs:
            return {
                "message": f"{file.filename} is already indexed",
                "status": "skipped"
            }

        # Read PDF content
        content = await file.read()
        pdf_text = ""
        
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"

        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")

        # Split text into chunks
        chunk_size = 400
        chunks = []
        id_counter = collection.count()
        for i in range(0, len(pdf_text), chunk_size):
            chunk = pdf_text[i:i + chunk_size].strip()
            if chunk:
                chunks.append((str(id_counter), chunk))
                id_counter += 1

        # Encode chunks and add to ChromaDB
        for cid, text in chunks:
            emb = model.encode(text).tolist()
            collection.add(
                documents=[text],
                ids=[cid],
                embeddings=[emb],
                metadatas=[{"source": file.filename}]
            )

        # Update tracking
        indexed_pdfs.append(file.filename)
        save_indexed_pdfs(indexed_pdfs)

        return {
            "message": f"{file.filename} uploaded and indexed successfully",
            "chunks_added": len(chunks),
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in upload_pdf: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


# ---------------------------
# Query endpoint - PRODUCTION READY
# ---------------------------
@app.post("/query/")
async def query_pdf(query: str = Form(...)):
    """
    Query the indexed PDFs with strict answer validation.
    Only returns answers from PDF context, otherwise returns 'not available'.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Check if collection has any documents
        if collection.count() == 0:
            return {
                "answer": "No documents have been uploaded yet. Please upload a PDF first.",
                "sources": [],
                "status": "no_documents"
            }

        # Encode query
        query_emb = model.encode(query).tolist()

        # Search in ChromaDB with distances
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=5,  # Get top 5 results
            include=["documents", "metadatas", "distances"]
        )

        matches = results['documents'][0]
        sources = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results else []

        print(f"Matches found: {len(matches)}")
        if distances:
            print(f"Best match distance: {min(distances):.4f}")

        # Check if we have any matches
        if not matches or not any(matches):
            print("No matches found")
            return {
                "answer": "This information is not available in the uploaded PDF documents.",
                "sources": [],
                "status": "not_in_pdf"
            }

        # Build context from matches
        context = "\n\n".join(matches[:3])
        print(f"Context length: {len(context)} chars")

        # STRICT PROMPT - Prevents hallucination
        prompt = f"""You are a PDF document assistant. Your ONLY job is to answer questions using the context provided below.

STRICT RULES YOU MUST FOLLOW:
1. Read the context carefully
2. If the answer EXISTS in the context → Give a clear, concise answer (1-2 sentences)
3. If the answer DOES NOT exist in the context → Reply with EXACTLY these words: "ANSWER_NOT_IN_PDF"
4. DO NOT use your general knowledge or training data
5. DO NOT make assumptions or inferences beyond what is written
6. DO NOT answer questions about things not mentioned in the context

Context from PDF document:
{context}

User Question: {query}

Your Answer (remember: answer from context OR say "ANSWER_NOT_IN_PDF"):"""

        print("Calling Groq API...")
        
        # Call Groq with strict parameters
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a strict PDF document reader. You ONLY answer from the given context. If the answer is not in the context, you must say ANSWER_NOT_IN_PDF. Never use your general knowledge."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=250,
            top_p=0.2  # Focused sampling
        )

        answer = response.choices[0].message.content.strip()
        print(f"LLM Response: {answer[:200]}...")

        # Validate response - Check if LLM says answer not found
        not_found_indicators = [
            "answer_not_in_pdf",
            "not in the pdf",
            "not available in the pdf",
            "not found in the pdf",
            "not mentioned in the",
            "context does not",
            "context doesn't",
            "information is not available",
            "cannot find this information"
        ]
        
        answer_lower = answer.lower()
        if any(indicator in answer_lower for indicator in not_found_indicators):
            print("LLM indicated: Answer not in PDF")
            return {
                "answer": "This information is not available in the uploaded PDF documents.",
                "sources": [],
                "status": "not_in_pdf"
            }

        # Get unique sources
        unique_sources = list(set([s['source'] for s in sources[:3]]))

        print(f"Answer found in: {unique_sources}")
        print(f"{'='*60}\n")
        
        return {
            "answer": answer,
            "sources": unique_sources,
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in query_pdf: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# ---------------------------
# List all indexed PDFs
# ---------------------------
@app.get("/indexed_pdfs/")
def get_indexed_pdfs():
    """Get list of all indexed PDF files"""
    try:
        pdfs = load_indexed_pdfs()
        return {
            "indexed_pdfs": pdfs,
            "total_count": len(pdfs)
        }
    except Exception as e:
        print(f"Error in get_indexed_pdfs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching PDFs: {str(e)}")


# ---------------------------
# Delete individual PDF endpoint
# ---------------------------
@app.delete("/delete_pdf/{filename}")
async def delete_pdf(filename: str):
    """Delete a specific PDF and all its chunks from the index"""
    try:
        print(f"Attempting to delete: {filename}")
        
        # Get all documents from collection
        all_docs = collection.get()
        
        # Find IDs with matching filename in metadata
        ids_to_delete = []
        for idx in range(len(all_docs['ids'])):
            metadata = all_docs['metadatas'][idx]
            if metadata.get('source') == filename:
                ids_to_delete.append(all_docs['ids'][idx])
        
        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found")
        
        print(f"Found {len(ids_to_delete)} chunks to delete")
        
        # Delete documents by IDs
        collection.delete(ids=ids_to_delete)
        
        # Update indexed PDFs list
        indexed_pdfs = load_indexed_pdfs()
        if filename in indexed_pdfs:
            indexed_pdfs.remove(filename)
            save_indexed_pdfs(indexed_pdfs)
        
        print(f"Successfully deleted {filename}")
        
        return {
            "message": f"Successfully deleted {filename}",
            "chunks_deleted": len(ids_to_delete),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in delete_pdf: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")


# ---------------------------
# Clear index endpoint
# ---------------------------
@app.delete("/clear_index/")
def clear_index():
    """Clear all indexed PDFs and reset the database"""
    try:
        client.delete_collection(name="pdf_collection")
        global collection
        collection = client.get_or_create_collection(name="pdf_collection")
        save_indexed_pdfs([])

        return {
            "message": "All indexed data cleared",
            "status": "success"
        }
    except Exception as e:
        print(f"Error in clear_index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing index: {str(e)}")