from transformers import pipeline
import numpy as np
import re


judge = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=80
)


def grounding_score(answer, chunks):
    context = " ".join(c.page_content.lower() for c in chunks)
    words = re.findall(r"\w+", answer.lower())

    hits = sum(w in context for w in words)
    return hits / max(len(words), 1)




def evidence_snippets(chunks, max_len=120):
    return [
        c.page_content[:max_len].replace("\n", " ") + "..."
        for c in chunks
    ]



def llm_faithfulness(question, answer, snippets):
    context = "\n".join(snippets)

    prompt = f"""
Question: {question}
Evidence:
{context}
Answer: {answer}

Score faithfulness between 0 and 1 (only number):
"""

    raw = judge(prompt)[0]["generated_text"]
    nums = re.findall(r"\d+\.?\d*", raw)

    return float(nums[0]) if nums else 0.5


def evaluate_turn(question, answer, chunks):
    g = grounding_score(answer, chunks)
    snippets = evidence_snippets(chunks)
    f = llm_faithfulness(question, answer, snippets)

    return {
        "grounding": round(g, 3),
        "faithfulness": round(f, 3)
    }

def evaluate_chat(chat_log):
    results = [
        evaluate_turn(
            t["question"],
            t["answer"],
            t["chunks"]
        )
        for t in chat_log
    ]

    return {
        "avg_grounding": round(np.mean([r["grounding"] for r in results]), 3),
        "avg_faithfulness": round(np.mean([r["faithfulness"] for r in results]), 3),
        "turns": results
    }
