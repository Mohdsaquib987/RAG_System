
import os
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=groq_key)

# Permanent Qdrant storage (local folder)
qdrant = QdrantClient(path="qdrant_db")

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Step 1: Read ALL PDFs
# -------------------------
pdf_folder = "pdfs"
all_chunks = []
chunk_size = 400
id_counter = 0

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file)
        print(f"📄 Reading: {file}")

        pdf_text = ""
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    pdf_text += content + "\n"

        # Split into chunks
        for i in range(0, len(pdf_text), chunk_size):
            chunk = pdf_text[i:i + chunk_size]
            all_chunks.append((id_counter, chunk))
            id_counter += 1

print(f"📚 Total chunks created: {len(all_chunks)}")

# -------------------------
# Step 2: Store embeddings
# -------------------------
vectors = []
for pid, text in all_chunks:
    emb = model.encode(text).tolist()
    vectors.append(
        PointStruct(
            id=pid,
            vector=emb,
            payload={"text": text}
        )
    )

# Recreate collection
qdrant.recreate_collection(
    collection_name="docs",
    vectors_config=VectorParams(
        size=len(vectors[0].vector),
        distance=Distance.COSINE
    )
)

qdrant.upsert(
    collection_name="docs",
    points=vectors
)

print("✅ All PDFs indexed permanently in Qdrant.")

# -------------------------
# Step 3: Ask Questions
# -------------------------
while True:
    query = input("\n🧠 Ask something (or type 'exit'): ")

    if query.lower() == "exit":
        break

    query_vec = model.encode(query).tolist()

    # NEW correct Qdrant search
    results = qdrant.query_points(
        collection_name="docs",
        query=query_vec,
        limit=3
    )

    context = "\n".join(
        [match.payload["text"] for match in results.points]
    )

    prompt = (
        f"Use ONLY the following context to answer.\n"
        f"Keep answer in 2 sentences max.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n💬  says:\n", response.choices[0].message.content)
