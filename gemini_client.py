"""
gemini_client.py
----------------
Thin wrapper around the Google Generative AI SDK (google-genai).

Uses the model specified by the GEMINI_MODEL env variable
(default: gemini-2.5-flash-lite) to generate a grounded answer
from retrieved context chunks.

Required environment variables (loaded from .env by main.py before import):
    GEMINI_API_KEY  – your Google AI Studio / Vertex AI API key
    GEMINI_MODEL    – model name (default: gemini-2.5-flash-lite)
"""

import os
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Module-level client singleton
_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Return the cached Gemini client, creating it on first call."""
    global _client
    if _client is None:
        print(f"[gemini] Initialising Gemini client (model: {GEMINI_MODEL}) …")
        _client = genai.Client(api_key=GEMINI_API_KEY)
        print("[gemini] Client ready.")
    return _client


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a drug-drug interaction (DDI) lookup assistant. Your ONLY knowledge source is the CONTEXT provided in each request — retrieved chunks from a medical knowledge base (DrugBank DDI dataset, FDA regulatory data, PubMed abstracts, and DrugBank compound records).

STRICT RULES — follow every one without exception:
1. NEVER use any knowledge outside the provided CONTEXT. If a fact is not in the context, it does not exist for you.
2. If the context does not contain sufficient information to answer the sections below, say exactly: "The knowledge base does not contain enough information about this combination."
3. Do NOT speculate, infer, or fill gaps with general pharmacology knowledge.
4. Do NOT answer questions unrelated to drug-drug interactions (e.g. disease explanations, dosing, lifestyle advice). Respond: "This question is outside the scope of the drug interaction knowledge base."

OUTPUT FORMAT — always use this exact three-section structure when the context is sufficient:

**What happens when both are taken together:**
- List every interaction effect found in the context. Be specific and directional (e.g. "Drug A increases the serum concentration of Drug B", "risk of bleeding is increased").

**What happens in the body:**
- Explain the physiological and clinical effects on the body as described in the context: what organs or systems are affected, what symptoms or measurable changes the patient may experience (e.g. increased bleeding, elevated drug levels in blood, liver enzyme changes, blood pressure changes). Use plain language a patient can understand.
- If the context contains no body-level detail, write: "The knowledge base does not describe the specific body effects for this combination."

**What can be done instead:**
- List only alternatives or safer options explicitly mentioned in the context.
- If the context mentions no alternatives, write: "The knowledge base does not mention specific alternatives for this combination."

End every response with: "⚠️ Consult a licensed healthcare professional before making any medication changes."
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(query: str, context_chunks: list[str]) -> str:
    """
    Generate a grounded natural-language answer using Gemini.

    Args:
        query:          The user's original question.
        context_chunks: List of raw text chunks retrieved from Pinecone.

    Returns:
        The model's text response as a string.
    """
    client = get_client()

    # Build the context block
    numbered_context = "\n\n".join(
        f"[{i + 1}] {chunk.strip()}" for i, chunk in enumerate(context_chunks)
    )

    user_message = (
        f"CONTEXT (your only permitted knowledge source):\n{numbered_context}\n\n"
        f"QUESTION: {query}\n\n"
        "Using ONLY the context above, answer in the two-section format defined in your instructions. "
        "Do not introduce any information that does not appear verbatim or by clear implication in the context."
    )

    print(f"[gemini] Generating answer for: '{query[:80]}{'…' if len(query) > 80 else ''}'")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.2,          # low temperature for factual accuracy
            max_output_tokens=1024,
        ),
        contents=user_message,
    )

    answer = response.text or ""
    print(f"[gemini] Answer generated ({len(answer)} chars).")
    return answer
