import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import PyPDF2
from groq import Groq
import json

# Load environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize Groq client
groq_client = Groq(api_key=groq_key)

# Initialize ChromaDB client with persistence
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pdf_collection")

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Track indexed PDFs
TRACKING_FILE = "./chroma_db/indexed_pdfs.json"

def load_indexed_pdfs():
    """Load list of already indexed PDF filenames"""
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return []

def save_indexed_pdfs(pdf_list):
    """Save list of indexed PDF filenames"""
    os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
    with open(TRACKING_FILE, 'w') as f:
        json.dump(pdf_list, f)

# Step 1: Read PDFs & chunk text
pdf_folder = "pdfs"
chunk_size = 400

indexed_pdfs = load_indexed_pdfs()
print(f"📂 Already indexed: {len(indexed_pdfs)} PDFs")

if not os.path.exists(pdf_folder):
    print(f"❌ Folder '{pdf_folder}' not found!")
else:
    # Get current PDF files
    current_pdfs = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    # Find new PDFs
    new_pdfs = [pdf for pdf in current_pdfs if pdf not in indexed_pdfs]
    
    if new_pdfs:
        print(f"🆕 Found {len(new_pdfs)} new PDF(s) to index: {new_pdfs}")
        
        all_chunks = []
        # Get the next ID based on existing collection count
        id_counter = collection.count()
        
        for file in new_pdfs:
            pdf_path = os.path.join(pdf_folder, file)
            print(f"📄 Reading: {file}")
            
            pdf_text = ""
            try:
                with open(pdf_path, "rb") as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    for page in reader.pages:
                        content = page.extract_text()
                        if content:
                            pdf_text += content + "\n"
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
                continue
            
            # Split into chunks and filter empty ones
            for i in range(0, len(pdf_text), chunk_size):
                chunk = pdf_text[i:i + chunk_size].strip()
                if chunk:
                    all_chunks.append((str(id_counter), chunk, file))
                    id_counter += 1
        
        print(f"📚 Total new chunks: {len(all_chunks)}")
        
        # Step 2: Store embeddings in ChromaDB
        if all_chunks:
            successfully_indexed = set()
            for cid, text, source_file in all_chunks:
                try:
                    emb = model.encode(text).tolist()
                    collection.add(
                        documents=[text],
                        ids=[cid],
                        embeddings=[emb],
                        metadatas=[{"source": source_file}]
                    )
                    successfully_indexed.add(source_file)
                except Exception as e:
                    print(f"⚠️ Error adding chunk {cid}: {e}")
            
            # Update tracking file with successfully indexed PDFs
            indexed_pdfs.extend(list(successfully_indexed))
            save_indexed_pdfs(indexed_pdfs)
            
            print(f"✅ {len(successfully_indexed)} new PDF(s) indexed in ChromaDB.")
        else:
            print("❌ No valid chunks found to index.")
    else:
        print("✅ No new PDFs found. All PDFs are already indexed.")

print(f"\n📊 Total chunks in database: {collection.count()}")

# Step 3: Query loop
print("\n" + "="*50)
print("🧠 RAG Query System - Type your questions below")
print("="*50)

while True:
    query = input("\n🔍 Ask something (or type 'exit'): ").strip()
    if query.lower() == "exit":
        print("\n👋 Goodbye!")
        break

    if not query:
        continue

    try:
        query_emb = model.encode(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3,
            include=["documents", "metadatas"]
        )

        # Get matches and filter out None values
        matches = results.get('documents', [[]])[0] if results.get('documents') else []
        sources = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        
        # Filter out None values from both lists
        valid_data = [(doc, meta) for doc, meta in zip(matches, sources) if doc and meta]
        
        if valid_data:
            matches = [item[0] for item in valid_data]
            sources = [item[1] for item in valid_data]
            
            context = "\n".join(matches)
            prompt = (
                f"Use ONLY the following context to answer.\n"
                f"Give a direct answer in 1-2 short sentences maximum. Be concise.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )

            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content
            unique_sources = list(set([s['source'] for s in sources if s]))
            
            print("\n💬 Answer:")
            print(answer)
            print(f"\n📚 Sources: {', '.join(unique_sources)}")
        else:
            print("\n❌ No relevant information found in the indexed documents.")
            
    except Exception as e:
        print(f"\n⚠️ Error processing query: {e}")