"""Headless run of the Streamlit app (no API keys, so the fallback path is exercised)."""
import os
from streamlit.testing.v1 import AppTest

os.environ.setdefault('INTENT_CLASSIFIER', 'rule')
at = AppTest.from_file(os.path.join(os.path.dirname(__file__), '..', 'app', 'streamlit_app.py'), default_timeout=120)
at.run()
assert not at.exception, at.exception
for msg in ['How many podiums did Leclerc get in 2022?', 'And in 2020?', 'Who won the 2019 German Grand Prix?',
            'Who designed the 2023 Red Bull car?', 'How many points did Schumacher score in 2000?', 'Michael']:
    at.chat_input[0].set_value(msg).run()
    assert not at.exception, at.exception
print('markdown:', [m.value[:90] for m in at.markdown][-12:])
print('errors:', [e.value for e in at.error])
print('cache html present:', any('tt-cache' in str(h.proto) for h in at.get('html')))
at.chat_input[0].set_value('Predict Norris at the 2024 Miami Grand Prix starting from P3').run()
assert not at.exception, at.exception
print('view:', at.session_state.view)
predict = next(b for b in at.button if b.label == 'Predict')
predict.click().run()
assert not at.exception, at.exception
print('metrics:', [(m.label, m.value) for m in at.metric])
print('errors:', [e.value for e in at.error])
