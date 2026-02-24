import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

CHAT_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
CHAT_KEY = os.environ["AZURE_OPENAI_API_KEY"]
CHAT_DEPLOYMENT = os.environ["AZURE_DEPLOYMENT_NAME"]
CHAT_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

def _chat_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=CHAT_KEY,
        api_version=CHAT_API_VERSION,
        azure_endpoint=CHAT_ENDPOINT,
    )

SYSTEM_PROMPT = """You are a contract abstraction assistant.

CRITICAL RULES:
- You MUST return valid JSON only.
- Do NOT include markdown, commentary, or explanations outside JSON.
- If you cannot comply with the JSON schema, return the null object.

Return EXACTLY this JSON structure:

{
  "value": <typed value or null>,
  "raw_answer": <short human-readable string or null>,
  "citations": ["c00001", "c00012"]
}

Type rules:
- boolean: true or false
- date: ISO YYYY-MM-DD
- percentage: number only (e.g., 3.5)
- currency: number only (USD)
- number: number only
- identifier / enum / text / long_text: string
- list: JSON array of strings

If the answer is not explicitly stated in the excerpts, return:
{
  "value": null,
  "raw_answer": null,
  "citations": []
}
"""

def answer(question: str, expected_type: str, excerpts: List[Dict]) -> Dict:
    """
    expected_type: boolean | date | percentage | currency | number | identifier | enum | text | long_text | list
    """
    # Build context with chunk ids
    blocks = []
    for ex in excerpts:
        cid = ex.get("chunk_id") or ex.get("id")
        text = (ex.get("content") or "").strip()
        blocks.append(f"[{cid}] {text}")

    context = "\n\n".join(blocks)

    user_prompt = f"""Question: {question}
Expected answer type: {expected_type}

Excerpts:
{context}

Rules for value formatting:
- boolean: true/false
- date: ISO YYYY-MM-DD if possible, else null
- percentage: number only (e.g., 3.5 for 3.5%)
- currency: number only in USD (e.g., 25000)
- number: number only
- identifier/enum/text/long_text: string
- list: JSON array of strings
"""

    chat = _chat_client()
    resp = chat.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=700,
    )

    content = resp.choices[0].message.content.strip()

    # Try to parse JSON. If it fails, wrap as null (safe).
    import json
    try:
        return json.loads(content)
    except Exception:
        return {"value": None, "raw_answer": None, "citations": [], "notes": f"Non-JSON model output: {content[:200]}"}
