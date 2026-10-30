# LocalMind — personal document assistant

A local web app that lets you upload PDF, DOCX, TXT, and Markdown documents, then ask questions grounded in their contents. Text is extracted, split into overlapping passages, and indexed using Ollama embeddings. The most relevant passages are sent to Ollama's local chat API with the question.

## Requirements

- Python 3.10+
- Ollama running locally
- Models: `nomic-embed-text` and `qwen3:1.7b` (or another Ollama chat model)

## Install and start

```sh
ollama pull nomic-embed-text
ollama pull qwen3:1.7b
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn assistant:app --reload
```

Open http://127.0.0.1:8000. The first upload can take a little while while Ollama creates embeddings. Uploaded files and their searchable passages are stored in `data/` alongside the app.

Set `OLLAMA_CHAT_MODEL` to change the chat model, `OLLAMA_EMBED_MODEL` to change the embedding model, or `OLLAMA_URL` to use a different local Ollama address. For example:

```sh
OLLAMA_CHAT_MODEL=qwen3:1.7b uvicorn assistant:app --reload
```

The assistant embeds each passage and question with `nomic-embed-text`, ranks passages by cosine similarity, and sends the best matches to the local chat model with filename context. It declines when no passage clears the relevance threshold. Recent conversation turns are included so short follow-up questions can refer to earlier questions and answers. Set `RAG_MIN_RELEVANCE` to adjust that threshold (default `0.15`). Scanned PDFs without a text layer need OCR before upload. Upload size is limited to 25 MB per file.

## Try it

1. Upload a text-based PDF, DOCX, TXT, or Markdown file from the left sidebar. Confirm its filename appears in the library.
2. Ask a question whose answer is in the file. Check that the answer cites the filename.
3. Ask a follow-up such as “Can you explain that in more detail?” or “What year was that?” The assistant uses the recent conversation and searches the uploaded passages again.
4. Upload an unsupported file (for example, `.png`) or an empty file. The interface should show a clear error. A scanned PDF without selectable text is reported as containing no readable text.
5. To confirm Ollama connection errors are surfaced, stop Ollama and try an upload or question; restart Ollama afterward.

The same service can be checked from a terminal while it is running:

```sh
curl -F 'file=@/path/to/notes.txt' http://127.0.0.1:8000/api/upload
curl http://127.0.0.1:8000/api/documents
curl -H 'Content-Type: application/json' \
  -d '{"question":"What are the key points?","history":[]}' \
  http://127.0.0.1:8000/api/chat
```
