"""Toto: the Graph-RAG Formula 1 assistant (Section 5). Two views, Chat and Prediction, in the approved design
(design-brief.md / f1-opening-page.html): theme in .streamlit/config.toml, styles in app/theme.css,
images and the Titillium Web font in app/static/.

run:  python start.py      (or: streamlit run app/streamlit_app.py from the repository root)
"""
import html
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from f1rag import config                                                         # noqa: E402
from f1rag.graph.client import get_client                                        # noqa: E402
from f1rag.pipeline import Assistant, SessionState                               # noqa: E402
from f1rag.preprocessing.entities import EntityExtractor                         # noqa: E402
from f1rag.preprocessing.classifiers import (RuleClassifier, EmbeddingClassifier,  # noqa: E402
                                             SentenceTransformerEncoder)
from f1rag.llm.providers import MODELS, DEFAULT_PAIR                             # noqa: E402
from f1rag.llm.answer import API_ERROR                                           # noqa: E402
from f1rag.prediction.predictor import PodiumPredictor, ValidationError, PLAIN, UNITS  # noqa: E402
from f1rag.prediction.explain import graph_facts, explain                        # noqa: E402

st.set_page_config(page_title='Toto · F1 assistant', page_icon='🏁', layout='wide',
                   initial_sidebar_state='expanded')
STATIC = 'app/static'
st.html('<style>' + (Path(__file__).parent / 'theme.css').read_text(encoding='utf-8') + '</style>')


# ------------------------------------------------------------------------------------------------ resources
@st.cache_resource
def resources():
    extractor = EntityExtractor()
    client = get_client()
    choice = config.env('INTENT_CLASSIFIER', 'embedding')
    note = ''
    if choice == 'embedding':
        try:
            clf = EmbeddingClassifier(SentenceTransformerEncoder(config.EMBEDDING_MODEL), extractor,
                                      min_score=float(config.env('INTENT_MIN_SCORE', '0.45')),
                                      margin=float(config.env('INTENT_MARGIN', '0.04')))
        except Exception as e:                                      # model not downloadable: keep the app usable
            clf, note = RuleClassifier(), f'embedding model unavailable ({type(e).__name__}); using rules'
    else:
        clf = RuleClassifier()
    return Assistant(client, clf, extractor), PodiumPredictor(), note


bot, predictor, clf_note = resources()
ss = st.session_state
ss.setdefault('state', SessionState())
ss.setdefault('view', 'Chat')
ss.setdefault('prefill', None)
ss.setdefault('pred_result', None)
if ss.get('goto'):                       # a chat message routed to the prediction view (set before the widget exists)
    ss.view = ss.pop('goto')
if ss.view is None:                      # the segmented control was clicked off: keep the last view
    ss.view = ss.get('last_view', 'Chat')
ss.last_view = ss.view
esc = html.escape


# ------------------------------------------------------------------------------------------------ sidebar
CACHE_SLOTS = [('Driver', 'driver'), ('Team', 'constructor'), ('Circuit', 'circuit'), ('Race', 'race'),
               ('Season', 'season')]


def cache_html(state):
    view = state.cache_view()
    if not any(view.get(k) for _, k in CACHE_SLOTS) and not view.get('driver2'):
        return ('<div class="tt-cache-empty">Empty. Ask a question and the driver, team, circuit, race and season '
                'will appear here.</div>')
    rows = []
    for label, key in CACHE_SLOTS:
        v = view.get(key)
        if key == 'driver' and view.get('driver2'):
            v = f"{view.get('driver', '–')} · {view['driver2']}"
        v = str(v).split(' (id ')[0].replace(') · ', ' · ') if v else None
        rows.append(f'<dt>{label}</dt>' + (f'<dd>{esc(v)}</dd>' if v else '<dd class="tt-empty">–</dd>'))
    return f'<dl class="tt-cache">{"".join(rows)}</dl>'


def turn_log():
    logs = [t['meta'].log for t in ss.state.history if t['role'] == 'assistant' and t['meta'].log]
    return '\n'.join(json.dumps(r, ensure_ascii=False, default=str) for r in logs) + ('\n' if logs else '')


with st.sidebar:
    st.html(f'<div class="tt-logo"><img src="{STATIC}/f1-logo-red.png" alt="F1"></div>')
    ids = list(MODELS)
    with st.container(key='side_models'):
        st.html('<div class="tt-side-label">Models</div>')
        m1 = st.selectbox('Model A', ids, index=ids.index(DEFAULT_PAIR[0]), key='model_a')
        m2 = st.selectbox('Model B', ids, index=ids.index(DEFAULT_PAIR[1]), key='model_b')
    with st.container(key='side_cache'):
        st.html('<div class="tt-side-label">Entity cache</div>')
        st.html(cache_html(ss.state))
    with st.container(key='side_bottom'):
        st.download_button('Download turn log', turn_log(), file_name='toto_turn_log.jsonl',
                           mime='application/json', key='dl_log')

# ------------------------------------------------------------------------------------------------ top bar
with st.container(key='topbar'):
    st.segmented_control('View', ['Chat', 'Prediction'], key='view', label_visibility='collapsed')


# ------------------------------------------------------------------------------------------------ chat view
def submit(msg):
    with st.spinner('Thinking...'):
        res = bot.handle(msg, ss.state, models=(m1, m2), few_shot=False)
    if res.kind == 'prediction':
        ss.prefill, ss.pred_result = res.prefill, None
        ss.goto = 'Prediction'
    st.rerun()


LAUNCH_SECONDS = 1.1          # length of the GO animation (theme.css: carGoA / carGoB)


def launch_car():
    """GO pressed on the opening page: remember the question and start the car animation on the next run."""
    msg = (ss.get('ask_open') or '').strip()
    if msg:
        ss.launch_id = ss.get('launch_id', 0) + 1
        ss.launch_msg = msg


def opening_page():
    question = ss.pop('launch_msg', None)
    n = ss.get('launch_id', 0)
    car_class = ('go-a' if n % 2 else 'go-b') if question else ''
    with st.container(key='opening'):
        st.html(f'<div class="tt-hero"><img class="{car_class}" src="{STATIC}/car-hero.png" '
                'alt="Formula 1 car seen from above"></div>')
        st.html('<div class="tt-col tt-intro"><h1>Ask me about '
                f'<img src="{STATIC}/f1-logo-outline.png" alt="F1"></h1>'
                "<p>Hello I'm Toto, your Formula 1 data assistant. Ask about any driver, team, circuit or race from "
                '1950 to 2024 and get answers taken straight from the knowledge graph, or ask for a podium '
                'prediction and see what the model based it on.</p></div>')
        with st.container(key='open_input'):
            st.chat_input('Ask a question', key='ask_open', on_submit=launch_car)
            st.html('<p class="tt-fine">Answers are limited to what the graph contains.</p>')
    if question:
        # The page above is already on screen with the car driving off; answer while it plays, then switch
        # to the chat view once both the animation and the answer are done.
        t0 = time.perf_counter()
        res = bot.handle(question, ss.state, models=(m1, m2), few_shot=False)
        if res.kind == 'prediction':
            ss.prefill, ss.pred_result = res.prefill, None
            ss.goto = 'Prediction'
        time.sleep(max(0.0, LAUNCH_SECONDS - (time.perf_counter() - t0)))
        st.rerun()


def dot_graph(sg):
    dot = ['digraph G { rankdir=LR; bgcolor="transparent"; pad=0.2;',
           'node [shape=box, style="rounded,filled", fillcolor="#0B0B0C", color="#26262A", fontcolor="#F5F5F5", '
           'fontname="Titillium Web", fontsize=11, penwidth=1];',
           'edge [color="#6B6F74", fontcolor="#8B8F94", fontname="Titillium Web", fontsize=9, arrowsize=0.6];']
    for n in sg['nodes']:
        lab = f"{n['label']}\\n{n['caption']}".replace('"', "'")
        accent = ', color="#00D2BE", penwidth=1.4' if n['label'] == 'Result' else ''
        dot.append(f'"{n["id"]}" [label="{lab}"{accent}];')
    for e in sg['edges']:
        dot.append(f'"{e["source"]}" -> "{e["target"]}" [label="{e["type"]}"];')
    return '\n'.join(dot) + '}'


def trace_label(res):
    rv = res.retrieval
    bits = [res.intent or '']
    if rv:
        bits += [', '.join(c['template'] for c in rv.calls), f"{len(rv.rows)} row{'s' if len(rv.rows) != 1 else ''}"]
    return '**Retrieval trace** &nbsp; :gray[' + ' · '.join(b for b in bits if b) + ']'


def show_trace(res, idx):
    with st.expander(trace_label(res), expanded=False):
        d = res.decision or {}
        st.html('<div class="tt-trace-h">Intent and entities</div>'
                f'<p class="tt-trace-p"><code>{esc(str(res.intent))}</code> · {esc(str(d.get("status")))} by '
                f'{esc(str(d.get("method")))}'
                + (f' · confidence {d["confidence"]}' if d.get('confidence') is not None else '')
                + (f' · {esc(d["reason"])}' if d.get('reason') else '') + '</p>'
                + '<p class="tt-trace-p">' + ' · '.join(f'{esc(k)}: <code>{esc(str(v))}</code>'
                                                        for k, v in (res.slots or {}).items()) + '</p>')
        rv = res.retrieval
        if rv and rv.subgraph:
            sg = rv.subgraph
            st.html('<div class="tt-trace-h">Retrieved subgraph</div>')
            st.graphviz_chart(dot_graph(sg))
            c1, c2 = st.columns(2)
            with c1:
                st.html('<div class="tt-trace-h">Nodes</div>')
                st.dataframe(pd.DataFrame(sg['nodes']), hide_index=True, width='stretch')
            with c2:
                st.html('<div class="tt-trace-h">Edges</div>')
                st.dataframe(pd.DataFrame(sg['edges']), hide_index=True, width='stretch')
            st.html('<div class="tt-trace-h">Relationship statements</div><p class="tt-trace-p tt-statements">'
                    + ' · '.join(f'<code>{esc(s)}</code>' for s in sg['statements']) + '</p>')
        elif rv:
            st.html('<div class="tt-trace-h">Retrieved values</div>')
            for c in rv.calls:
                if len(c['rows']) == 1:
                    cols = st.columns(min(4, len(c['rows'][0])))
                    for i, (k, v) in enumerate(c['rows'][0].items()):
                        cols[i % len(cols)].metric(k.replace('_', ' '), ', '.join(map(str, v)) if isinstance(v, list)
                                                   else str(v))
                elif c['rows']:
                    st.dataframe(pd.DataFrame(c['rows']), hide_index=True, width='stretch')
        if rv:
            for c in rv.calls:
                st.html(f'<div class="tt-trace-h">Template <code>{esc(c["template"])}</code> · '
                        f'{esc(c["description"])} ({esc(c["hops"])}) · {len(c["rows"])} row(s) in {c["seconds"]} s'
                        '</div>')
                st.code(c['cypher'], language='cypher')
                st.html('<p class="tt-trace-p">Parameters: <code>' + esc(json.dumps(c['params'])) + '</code>' +
                        (f" + {esc(', '.join(c['fixed_params']))}" if c['fixed_params'] else '') + '</p>')
                st.json(c['rows'], expanded=False)


def show_assistant(res, idx):
    if res.kind == 'answer' and res.answers:
        cols = st.columns(len(res.answers))
        for j, (col, a) in enumerate(zip(cols, res.answers)):
            with col, st.container(key=f'card_{idx}_{j}'):
                meta = (f' <span>· {a.latency_s:.2f} s · {a.input_tokens}→{a.output_tokens} tokens</span>'
                        if a.ok else '')
                st.html(f'<div class="tt-model">Model {"AB"[j] if j < 2 else j + 1} · {esc(a.model_id)}{meta}</div>'
                        + ('' if a.ok else f'<div class="tt-api-error">{API_ERROR}</div>'))
                st.markdown(a.text)
        show_trace(res, idx)
    else:
        with st.container(key=f'card_{idx}_toto'):
            st.html('<div class="tt-model">Toto</div>')
            st.markdown(res.message)
        if res.retrieval and (res.kind == 'answer' or res.kind == 'no_data'):
            show_trace(res, idx)


def chat_view():
    if not ss.state.history:
        opening_page()
        return
    with st.container(key='thread'):
        for i, turn in enumerate(ss.state.history):
            if turn['role'] == 'user':
                st.html(f'<div class="tt-user"><p>{esc(turn["content"])}</p></div>')
            else:
                show_assistant(turn['meta'], i)
    msg = st.chat_input('Ask a question', key='ask_chat')
    if msg:
        submit(msg)


# ------------------------------------------------------------------------------------------------ prediction view
def prediction_view():
    with st.container(key='predict'):
        _prediction_body()


def _prediction_body():
    st.html('<h1 class="tt-page-h">Podium prediction</h1>')
    rows = predictor.rows
    pre = ss.prefill or {}
    races = rows[['raceId', 'year', 'round', 'race_name', 'circuitId']].drop_duplicates('raceId')
    races = races.sort_values(['year', 'round'], ascending=False)
    race_labels = {r.raceId: f'{r.year} {r.race_name}' for r in races.itertuples()}
    race_ids = list(race_labels)
    default_race = pre['race'].id if pre.get('race') and pre['race'].id in race_labels else race_ids[0]
    c1, c2 = st.columns(2)
    race_id = c1.selectbox('Race (2003-2024, as in the Milestone 1 data)', race_ids,
                           index=race_ids.index(default_race), format_func=race_labels.get)
    field = rows[rows.raceId == race_id].sort_values('driver_surname')
    drv_labels = {r.driverId: f'{r.driver_forename} {r.driver_surname} ({r.constructor_name})'
                  for r in field.itertuples()}
    drv_ids = list(drv_labels)
    want = pre['driver'].id if pre.get('driver') else None
    if want is not None and want not in drv_labels:
        st.warning(f"{pre['driver'].name} has no pre-race row for this race (not a starter in the data). "
                   'Pick another driver or enter values manually.')
    driver_id = c2.selectbox('Driver', drv_ids, index=drv_ids.index(want) if want in drv_ids else 0,
                             format_func=drv_labels.get)
    base = predictor.lookup(driver_id, race_id) or {}
    user_vals = dict(pre.get('user_values', {})) if pre.get('driver') and pre['driver'].id == driver_id and \
        pre.get('race') and pre['race'].id == race_id else {}
    if 'quali_position' in user_vals and 'grid_position' in user_vals:
        user_vals['grid_minus_quali'] = user_vals['grid_position'] - user_vals.pop('quali_position')
    user_vals.pop('quali_position', None)
    if user_vals.get('pit_lane_start') == 1 and base:
        user_vals['grid_position'] = base['field_size'] + 1

    st.html('<p class="tt-page-p">Values from your message are shown normally. Gray labels are defaults: the '
            'pre-race value computed in Milestone 1 for this driver and race (form and standings from earlier races '
            'only). Edit any value, then press <b>Predict</b>.</p>')
    values, allow_missing = {}, set()
    groups = {}
    for f in predictor.features:
        groups.setdefault(predictor.ranges[f]['group'], []).append(f)
    with st.form('features'):
        for g, feats in groups.items():
            st.html(f'<div class="tt-group">{esc(g.replace("_", " ").title())}</div>')
            cols = st.columns(3)
            for i, f in enumerate(feats):
                spec = predictor.ranges[f]
                from_user = f in user_vals
                dv = user_vals.get(f, base.get(f, spec['train_median']))
                missing_default = dv is None or (isinstance(dv, float) and pd.isna(dv))
                label = PLAIN[f] if from_user else f':gray[{PLAIN[f]} · default]'
                helptext = (f"{spec['description']}. Valid range {spec['min']:g} to {spec['max']:g} "
                            f"(observed in 2003-2024 data)." + (f" Unit: {UNITS[f]}." if f in UNITS else '') +
                            (f" Can be not available: {spec['why_missing']}." if spec['missing_allowed'] else ''))
                with cols[i % 3]:
                    if spec['kind'] == 'binary':
                        v = st.selectbox(label, [0, 1], index=int(dv or 0), key=f'in_{f}_{driver_id}_{race_id}',
                                         format_func=lambda x: 'yes' if x else 'no', help=helptext)
                    else:
                        step = 1.0 if spec['kind'] == 'int' else 0.01
                        v = st.number_input(label, value=None if missing_default else float(dv), step=step,
                                            format='%.0f' if spec['kind'] == 'int' else '%.3f',
                                            key=f'in_{f}_{driver_id}_{race_id}', help=helptext,
                                            placeholder='required')
                    if spec['missing_allowed']:
                        na = st.checkbox('not available', value=missing_default, key=f'na_{f}_{driver_id}_{race_id}',
                                         help=spec['why_missing'])
                        if na:
                            allow_missing.add(f)
                            v = None
                    values[f] = v
        go = st.form_submit_button('Predict', type='primary')
    if go:
        try:
            clean = predictor.validate(values, allow_missing)
        except ValidationError as e:
            st.error('No prediction: please fix these inputs.\n\n' + '\n'.join(f'- {p}' for p in e.args[0]))
            ss.pred_result = None
            return
        info = rows[(rows.raceId == race_id) & (rows.driverId == driver_id)].iloc[0]
        pred = predictor.predict(clean, race={'id': race_id, 'name': race_labels[race_id]},
                                 actual_finish=int(info.actual_finish) if pd.notna(info.actual_finish) else None)
        facts, fact_calls = graph_facts(bot.client, int(driver_id), int(info.constructorId), int(info.circuitId),
                                        int(info.year), int(info['round']))
        exp = explain(predictor, pred, drv_labels[driver_id], race_labels[race_id], facts, m1)
        ss.pred_result = {'pred': pred, 'exp': exp, 'facts': fact_calls, 'driver': drv_labels[driver_id]}
    r = ss.pred_result
    if r:
        pred = r['pred']
        with st.container(key='pred_result'):
            a, b, c = st.columns(3)
            a.metric('Podium probability', f'{pred.probability * 100:.1f}%')
            b.metric('Predicted class', 'Podium' if pred.podium else 'Not podium',
                     help=f'threshold {pred.threshold:.3f}, chosen on validation in Milestone 1')
            c.metric('Actual finish', f'P{pred.actual_finish}' if pred.actual_finish else 'unknown')
            left, right = st.columns([1.1, 1])
            with left:
                st.html('<div class="tt-group">Local explanation (SHAP, computed live for this row)</div>')
                st.pyplot(shap_plot(pred))
            with right, st.container(key='pred_explain'):
                st.html(f'<div class="tt-model">Explanation · {esc(r["exp"]["model"])}</div>' +
                        ('' if r['exp']['ok'] else f'<div class="tt-api-error">{API_ERROR}</div>'))
                st.markdown(r['exp']['text'])
                with st.expander('Pre-race graph facts used'):
                    for c in r['facts']:
                        st.code(c['cypher'], language='cypher')
                        st.json({'params': c['params'], 'rows': c['rows']}, expanded=False)


def shap_plot(pred):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for w in ('Regular', 'SemiBold', 'Bold'):
        p = Path(__file__).parent / 'static' / 'fonts' / f'TitilliumWeb-{w}.ttf'
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    items = sorted(pred.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10][::-1]
    with plt.rc_context({'font.family': ['Titillium Web', 'DejaVu Sans'], 'text.color': '#F5F5F5',
                         'axes.labelcolor': '#8B8F94', 'xtick.color': '#8B8F94', 'ytick.color': '#F5F5F5',
                         'axes.edgecolor': '#26262A'}):
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        ax.barh([PLAIN[f] for f, _ in items], [v * 100 for _, v in items],
                color=['#00D2BE' if v > 0 else '#6B6F74' for _, v in items])
        ax.axvline(0, color='#8B8F94', lw=0.8)
        ax.set_xlabel('contribution to podium probability (percentage points); teal pushes up, gray down')
        ax.set_title(f'From the average {pred.base_value * 100:.1f}% to {pred.probability * 100:.1f}% '
                     f'(top 10 features)', fontsize=10.5, color='#F5F5F5')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8.5)
        fig.tight_layout()
    return fig


if ss.view == 'Prediction':
    prediction_view()
else:
    chat_view()
