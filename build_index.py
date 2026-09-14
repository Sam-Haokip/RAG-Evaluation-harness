import os

CORPUS_DIR = "corpus"
CHUNK_SIZE = 20  # lines per chunk

def chunk_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    for i in range(0, len(lines), CHUNK_SIZE):
        chunk_lines = lines[i:i + CHUNK_SIZE]
        chunk_text = "".join(chunk_lines)
        chunks.append(chunk_text)

    return chunks

def build_all_chunks():
    all_chunks = []
    for filename in os.listdir(CORPUS_DIR):
        if filename.endswith(".py"):
            filepath = os.path.join(CORPUS_DIR, filename)
            file_chunks = chunk_file(filepath)
            for idx, chunk_text in enumerate(file_chunks):
                all_chunks.append({
                    "source_file": filename,
                    "chunk_index": idx,
                    "text": chunk_text
                })
    return all_chunks

# Build chunks from the whole corpus
all_chunks = build_all_chunks()
print(f"Total chunks across whole corpus: {len(all_chunks)}")
print("Example chunk entry:")
print(all_chunks[0])

from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AICREDITS_API_KEY"),
    base_url="https://aicredits.in/v1"
)

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Embed all chunks
print(f"Embedding all {len(all_chunks)} chunks... this may take a minute.")

for i, chunk in enumerate(all_chunks):
    chunk["embedding"] = embed_text(chunk["text"])
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(all_chunks)} done...")

print("All chunks embedded.")

# Save everything to a file so we don't have to redo this
with open("chunks_with_embeddings.json", "w") as f:
    json.dump(all_chunks, f)

print("Saved to chunks_with_embeddings.json")

import ast

def chunk_file_by_function(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source_code = f.read()
        lines = source_code.splitlines()

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    chunks = []
    covered_lines = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno
            chunk_text = "\n".join(lines[start:end])
            chunks.append({
                "name": node.name,
                "type": type(node).__name__,
                "text": chunk_text
            })
            for i in range(start, end):
                covered_lines.add(i)

    # Capture module-level content: any line not part of a function/class
    module_lines = [line for i, line in enumerate(lines) if i not in covered_lines]
    module_text = "\n".join(module_lines).strip()

    if module_text:
        chunks.append({
            "name": "module_level",
            "type": "Module",
            "text": module_text
        })

    return chunks
  
def build_all_chunks_by_function():
    all_chunks = []
    for filename in os.listdir(CORPUS_DIR):
        if filename.endswith(".py"):
            filepath = os.path.join(CORPUS_DIR, filename)
            file_chunks = chunk_file_by_function(filepath)
            for chunk in file_chunks:
                all_chunks.append({
                    "source_file": filename,
                    "name": chunk["name"],
                    "type": chunk["type"],
                    "text": chunk["text"]
                })
    return all_chunks

function_chunks = build_all_chunks_by_function()
print(f"Total function-aware chunks across corpus: {len(function_chunks)}")

for i, chunk in enumerate(function_chunks):
    chunk["embedding"] = embed_text(chunk["text"])
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(function_chunks)} done...")

with open("chunks_function_aware.json", "w") as f:
    json.dump(function_chunks, f)

print("Saved to chunks_function_aware.json")