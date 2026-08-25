"""Self-edit ("implications") generation via local Ollama, using SEAL's own prompt
(general-knowledge/src/data_generation/make_squad_data.py::MAKE_SQUAD_DATA_TEMPLATES_BASE["implications"])."""
import requests

IMPLICATIONS_PROMPT = (
    "Let's read the following passage and produce a list of implications derived directly or "
    "indirectly from the content.\n\nPassage:\n{title}\n{context}\n\nImplications:\n"
)


def generate_self_edit(
    title: str,
    context: str,
    model: str = "qwen2.5:1.5b",
    host: str = "http://localhost:11434",
    max_tokens: int = 512,
    temperature: float = 1.0,
    seed: int = 0,
) -> str:
    prompt = IMPLICATIONS_PROMPT.format(title=title, context=context)
    r = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"].strip()
