from fastapi import FastAPI, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from uuid import uuid4
from typing import Dict, List, Optional

import openai
import chromadb
import pathlib

# --- constants ---------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert on PETSc and scientific computing. "
    "Answer clearly and concisely.  When helpful, show short code snippets."
)
LLAMA_ENDPOINT = "http://127.0.0.1:8001/v1"

MAX_TURNS   = 6        # how many previous user/assistant pairs to keep
MAX_TOKENS  = 256

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2] 	# ~/petsc-chat-demo
CHROMA_DIR = ROOT_DIR / "chromadb_petsc" 		# Absolute path.

# --- model & vector store ----------------------------------------------------
client = openai.OpenAI(base_url=LLAMA_ENDPOINT, api_key="not-needed")
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_or_create_collection("petsc-docs")

# --- FastAPI app -------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://demo.local", "http://localhost:8080"],
    allow_methods=["POST"],
    allow_headers=["*"],
    allow_credentials=True,
)

# in-memory conversation buckets  {session_id: [ {role,content}, ... ]}
conversations: Dict[str, List[Dict[str, str]]] = {}


# --------------------------------------------------------------------------- #
@app.post("/chat")
def chat(request: Request, payload: Dict[str, str], session_id: Optional[str] = Cookie(None)):
    user_msg = payload.get("message", "").strip()
    if not user_msg:
        return JSONResponse({"error": "empty message"}, status_code=400)

    # get / create session bucket
    if session_id is None or session_id not in conversations:
        session_id = str(uuid4())
        conversations[session_id] = []

    history = conversations[session_id]

    # --- retrieval -----------------------------------------------------------
    docs = collection.query(
        query_texts=[user_msg],
        n_results=4,
        include=["documents", "metadatas"],
    )
    context_blurb = "\n\n".join(docs["documents"][0]) if docs["documents"] else ""

    # --- build message list --------------------------------------------------
    history.append({"role": "user", "content": user_msg})
    recent_history = history[-MAX_TURNS*2:]          # user+assistant pairs = 2 msgs / turn
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent_history)
    if context_blurb:
        messages.append({"role": "system", "content": f"Relevant documentation:\n{context_blurb}"})

    # --- LLM call ------------------------------------------------------------
    resp = client.chat.completions.create(
        model="local",
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.2,
    )
    assistant_reply = resp.choices[0].message.content.strip()

    history.append({"role": "assistant", "content": assistant_reply})

    # --- return --------------------------------------------------------------
    json_resp = JSONResponse({"reply": assistant_reply})
    json_resp.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    return json_resp

