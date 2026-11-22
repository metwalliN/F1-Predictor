"""Language-model providers. Groq, OpenRouter and Hugging Face expose the OpenAI chat-completions format;
Ollama runs open-weight models locally. Keys come from environment variables or the uncommitted .env file.

Any failure (missing key, network, HTTP error, empty reply) raises LLMError; the caller shows only
"API connection error" and falls back to a direct reading of the retrieved rows.
"""
import re
import time
from dataclasses import dataclass

import requests

from .. import config


class LLMError(RuntimeError):
    pass


@dataclass
class ModelSpec:
    id: str              # shown in the UI and the log
    provider: str        # groq | openrouter | hf | ollama
    model: str           # provider's model name
    family: str
    size: str


# The compared models: different families and sizes, all free. Groq retired its free-tier Llama 3.x and Qwen3-32B
# models in July-August 2026 (console.groq.com/docs/deprecations); these were served when this list was written.
# Preview models can be withdrawn at short notice: check console.groq.com/docs/models and edit this list if needed.
# The two local models run through Ollama (ollama pull gemma3:4b / llama3.2:3b) and need no key.
MODELS = {
    'gpt-oss-120b': ModelSpec('gpt-oss-120b', 'groq', 'openai/gpt-oss-120b', 'OpenAI open-weight', '120B (MoE)'),
    'gpt-oss-20b': ModelSpec('gpt-oss-20b', 'groq', 'openai/gpt-oss-20b', 'OpenAI open-weight', '20B (MoE)'),
    'qwen3.8-27b': ModelSpec('qwen3.8-27b', 'groq', 'qwen/qwen3.8-27b', 'Alibaba Qwen', '27B (preview)'),
    'gemma3-4b-local': ModelSpec('gemma3-4b-local', 'ollama', 'gemma3:4b', 'Google Gemma', '4B'),
    'llama3.2-3b-local': ModelSpec('llama3.2-3b-local', 'ollama', 'llama3.2:3b', 'Meta Llama', '3B'),
}
DEFAULT_PAIR = ('gpt-oss-120b', 'qwen3.8-27b')
COMPARED = ['gpt-oss-120b', 'gpt-oss-20b', 'qwen3.8-27b']

ENDPOINTS = {'groq': 'https://api.groq.com/openai/v1/chat/completions',
             'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
             'hf': 'https://router.huggingface.co/v1/chat/completions'}
KEYS = {'groq': lambda: config.GROQ_API_KEY, 'openrouter': lambda: config.OPENROUTER_API_KEY,
        'hf': lambda: config.HF_TOKEN}


@dataclass
class LLMReply:
    model_id: str
    text: str
    latency_s: float
    input_tokens: int
    output_tokens: int


def chat(model_id, messages, temperature=0.0, max_tokens=400, timeout=45):
    spec = MODELS[model_id]
    t0 = time.perf_counter()
    try:
        if spec.provider == 'ollama':
            r = requests.post(f'{config.OLLAMA_URL}/api/chat', timeout=timeout, json={
                'model': spec.model, 'messages': messages, 'stream': False,
                'options': {'temperature': temperature, 'num_predict': max_tokens}})
            r.raise_for_status()
            j = r.json()
            text, tin, tout = j['message']['content'], j.get('prompt_eval_count', 0), j.get('eval_count', 0)
        else:
            key = KEYS[spec.provider]()
            if not key:
                raise LLMError(f'no API key for {spec.provider}')
            body = {'model': spec.model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens}
            extra = {}
            if spec.model.startswith('qwen/'):
                extra = {'reasoning_format': 'hidden'}        # Groq: return the answer without the thinking trace
                body['max_tokens'] = max_tokens + 1200
            if spec.model.startswith('openai/gpt-oss'):
                extra = {'reasoning_effort': 'low'}
                body['max_tokens'] = max_tokens + 800
            headers = {'Authorization': f'Bearer {key}'}
            r = requests.post(ENDPOINTS[spec.provider], timeout=timeout, json={**body, **extra}, headers=headers)
            if r.status_code == 400 and extra:                # provider rejected the reasoning option: plain request
                r = requests.post(ENDPOINTS[spec.provider], timeout=timeout, json=body, headers=headers)
            r.raise_for_status()
            j = r.json()
            text = j['choices'][0]['message'].get('content') or ''
            usage = j.get('usage') or {}
            tin, tout = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
    except LLMError:
        raise
    except Exception as e:                                    # network, HTTP status, malformed JSON
        raise LLMError(f'{type(e).__name__}: {e}') from e
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    if not text:
        raise LLMError('empty reply')
    return LLMReply(model_id, text, round(time.perf_counter() - t0, 3), int(tin or 0), int(tout or 0))
