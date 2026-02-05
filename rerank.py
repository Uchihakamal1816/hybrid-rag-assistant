import os
from groq import Groq
from dotenv import load_dotenv
import re
load_dotenv()

client = Groq(api_key=os.environ["GROQ_API"])

def rerank_chunks(query, chunks, top_k=3):
    context = "\n\n".join(
        [f"Chunk {i+1}:\n{c.page_content[:800]}" for i, c in enumerate(chunks)]
    )

    prompt = f"""
Select the most relevant chunks to answer the question.

Question:
{query}

Chunks:
{context}

Return ONLY numbers separated by commas.
Example: 2,5,1
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content

    numbers = re.findall(r"\d+", raw)
    selected = [int(n) - 1 for n in numbers[:top_k]]

    return [chunks[i] for i in selected if i < len(chunks)]
