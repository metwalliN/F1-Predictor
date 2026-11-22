"""Prediction explanation: live SHAP contributions + Milestone 1 global SHAP findings + pre-race graph facts
(templates restricted to races strictly before the scored race), turned into plain language by an LLM."""
from ..retrieval.templates import T
from ..llm.prompts import explanation_messages
from ..llm.providers import chat, LLMError
from ..llm.answer import API_ERROR


def graph_facts(client, driver_id, constructor_id, circuit_id, year, rnd):
    facts, calls = [], []
    for tid, params in (('prerace_driver_at_circuit', dict(driverId=driver_id, circuitId=circuit_id, year=year,
                                                           round=rnd)),
                        ('prerace_constructor_reliability', dict(constructorId=constructor_id, year=year,
                                                                 round=rnd))):
        if None in params.values():
            continue
        t = T[tid]
        try:
            rows = client.run(t.cypher, {**t.fixed_params, **params})
        except Exception:
            rows = []
        calls.append({'template': tid, 'cypher': t.cypher.strip(), 'params': params, 'rows': rows})
        facts.extend(rows)
    return facts, calls


def explain(predictor, pred, driver_name, race_name, facts, model_id):
    context = predictor.explanation_context(pred, driver_name, race_name, facts)
    msgs = explanation_messages(context)
    try:
        r = chat(model_id, msgs, max_tokens=350)
        return {'ok': True, 'text': r.text, 'model': model_id, 'latency_s': r.latency_s,
                'input_tokens': r.input_tokens, 'output_tokens': r.output_tokens, 'prompt': msgs}
    except LLMError as e:
        return {'ok': False, 'text': predictor.fallback_explanation(pred), 'model': model_id,
                'error': f'{API_ERROR} | {e}', 'prompt': msgs}
