# Solar Vector Store

A FastAPI server that implements an OpenAI‐style Vector Stores API using:

1. **Upstage Document Digitization** to parse PDFs (or other files) into text  
2. **Upstage Solar Embeddings** to turn text into vectors  
3. **Qdrant** (cloud or local) to store and search the vectors  

---

## Features

- **Vector Store Management**  
  Create, list, retrieve, update (distance metric), and delete named stores (Qdrant collections)

- **File Ingestion**  
  Upload a PDF (or other file), parse it into pages via Upstage, embed each page, and upsert into Qdrant

- **Search**  
  Embed a text query and perform a **k-NN** search over the stored vectors

- **File & Store Cleanup**  
  Delete individual ingested files or entire vector stores

---

## Prerequisites

- **Python 3.10+**  
- **Upstage API Key** (for Document Digitization & Embeddings)  
- **Qdrant** (cloud or self-hosted) URL & API Key (if you use Qdrant Cloud)

---

## Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-org/solar-vector-store.git
   cd solar-vector-store
   ```

2. **Create & activate virtual environment**  
   ```bash
   make venv
   source .venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   make install
   ```

4. **Configure environment**  
   Copy `.env.example` → `.env`, then set:
   ```
   UPSTAGE_API_KEY=up_...
   QDRANT_URL=https://<your-qdrant-cloud>
   QDRANT_API_KEY=<your-qdrant-key>    # optional for self-hosted
   ```

5. **Run the server**  
   ```bash
   make run
   ```
   The API will be available at `http://localhost:8000` and interactive docs at `http://localhost:8000/docs`.

---

## API Endpoints

| Method | Path                                        | Description                                 |
| ------ | ------------------------------------------- | ------------------------------------------- |
| POST   | `/vector_stores`                            | Create a new vector store                   |
| GET    | `/vector_stores`                            | List all vector stores                      |
| GET    | `/vector_stores/{store_id}`                 | Retrieve one vector store’s metadata        |
| PATCH  | `/vector_stores/{store_id}`                 | Update name or distance metric              |
| DELETE | `/vector_stores/{store_id}`                 | Delete a vector store                       |
| POST   | `/vector_stores/{store_id}/files`           | Upload & ingest a file                      |
| GET    | `/vector_stores/{store_id}/files`           | List ingested files                         |
| GET    | `/vector_stores/{store_id}/files/{file_id}` | Retrieve file metadata                      |
| DELETE | `/vector_stores/{store_id}/files/{file_id}` | Delete an ingested file’s vectors & metadata|
| POST   | `/vector_stores/{store_id}/query`           | Query the store with a text embedding       |

---

## Testing

1. Place a sample `test.pdf` in the `tests/` directory.  
2. Run the test suite:
   ```bash
   make test
   ```

---

## License

This project is licensed under the MIT License.
