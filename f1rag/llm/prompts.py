"""Prompts. The grounded-answer prompt has the three required parts: persona, context, task.
The few-shot variant adds three worked examples whose rows were retrieved from the graph with the same templates
(none of them is in the QA test set)."""
import json

PERSONA = ('You are Toto, a Formula 1 data analyst. You report only what the race data in front of you '
           'shows, in a friendly, conversational tone, the way a TV data analyst would explain a stat to a fan.')

TASK = """Task:
- Answer the user's question using ONLY the results in the context. Do not use your own knowledge of Formula 1.
- Quote every number exactly as it appears in the results (no rounding, no recalculating, no new totals).
- Do not add any count, name, race, team or year that is not in the results.
- If the results do not contain what the question asks, say so plainly and give what the results do show.
- State the scope of the figures in a short clause (what the data covers and what it excludes), using the scope note.
- Keep it to 2-4 sentences. No lists unless the question asks for a ranking."""


def context_block(question, intent, entities, calls):
    parts = [f'Question: {question}', f'Detected intent: {intent}',
             'Extracted entities: ' + json.dumps(entities, ensure_ascii=False)]
    for c in calls:
        parts.append(f"Template: {c['template']} ({c['description']})")
        parts.append(f"Scope note: {c['scope']}")
        parts.append('Results: ' + json.dumps(c['rows'], ensure_ascii=False, default=str))
    return '\n'.join(parts)


FEW_SHOT = [
    {
        'context': context_block(
            'How did Fernando Alonso do in 2006?', 'driver_stats',
            {'driver': 'Fernando Alonso', 'season': [2006, 2006]},
            [{'template': 'driver_season_stats', 'description': 'Wins, podiums, points, poles and average finish for '
              'one driver over a season or a range of seasons',
              'scope': 'Grand Prix results only: sprint races are not in the graph; points are race points as '
                       'recorded. A podium is a finishing position of 1 to 3; average finish uses classified finishes '
                       'only.',
              'rows': [{'driver': 'Fernando Alonso', 'first_season': 2006, 'last_season': 2006, 'wins': 7,
                        'podiums': 14, 'points': 134.0, 'starts_from_pole': 6, 'not_classified': 2,
                        'avg_classified_finish': 1.94, 'races_entered': 18}]}]),
        'answer': 'In 2006 Fernando Alonso won 7 of his 18 races and finished on the podium 14 times, scoring 134.0 '
                  'points. He started from pole 6 times, was not classified twice (2), and averaged 1.94 in the races '
                  'he was classified in. These figures cover Grand Prix results only.',
    },
    {
        'context': context_block(
            'Who won the 2010 Bahrain Grand Prix?', 'race_result', {'race': '2010 Bahrain Grand Prix'},
            [{'template': 'race_podium', 'description': 'Top finishers of one race (winner and podium by default)',
              'scope': 'The official classification of this one race (positionOrder).',
              'rows': [{'race': 'Bahrain Grand Prix', 'year': 2010, 'circuit': 'Bahrain International Circuit',
                        'finish': 1, 'driver': 'Fernando Alonso', 'constructor': 'Ferrari', 'grid': 3,
                        'points': 25.0, 'laps': 49, 'status': 'Finished'},
                       {'race': 'Bahrain Grand Prix', 'year': 2010, 'circuit': 'Bahrain International Circuit',
                        'finish': 2, 'driver': 'Felipe Massa', 'constructor': 'Ferrari', 'grid': 2, 'points': 18.0,
                        'laps': 49, 'status': 'Finished'},
                       {'race': 'Bahrain Grand Prix', 'year': 2010, 'circuit': 'Bahrain International Circuit',
                        'finish': 3, 'driver': 'Lewis Hamilton', 'constructor': 'McLaren', 'grid': 4,
                        'points': 15.0, 'laps': 49, 'status': 'Finished'}]}]),
        'answer': 'Fernando Alonso won the 2010 Bahrain Grand Prix for Ferrari, starting from 3rd on the grid. Felipe '
                  'Massa (Ferrari) was second and Lewis Hamilton (McLaren) third. This is the official classification '
                  'of that race.',
    },
    {
        'context': context_block(
            'Why did Ferrari struggle in 2020?', 'constructor_stats', {'constructor': 'Ferrari',
                                                                      'season': [2020, 2020]},
            [{'template': 'constructor_stats', 'description': 'Points, wins, podiums and reliability for a '
              'constructor over seasons',
              'scope': 'Every car entry of this constructor in Grand Prix results for these seasons. Reliability = '
                       'classified finishes / car entries. Points are summed race points (sprints not included).',
              'rows': [{'constructor': 'Ferrari', 'first_season': 2020, 'last_season': 2020, 'car_entries': 34,
                        'wins': 0, 'podiums': 3, 'points': 131.0, 'classified_finishes': 28,
                        'classified_rate': 0.824, 'mechanical_retirements': 2, 'races': 17}]}]),
        'answer': "The data doesn't say why Ferrari struggled, only how they did: in 2020 Ferrari had 0 wins and 3 "
                  'podiums from 34 car entries in 17 races, scoring 131.0 points. 28 entries were classified (a rate '
                  'of 0.824) and 2 ended in mechanical retirements. These are Grand Prix race results only.',
    },
]


def grounded_messages(question, intent, entities, calls, few_shot=False):
    system = PERSONA + '\n\n' + TASK
    msgs = [{'role': 'system', 'content': system}]
    if few_shot:
        for ex in FEW_SHOT:
            msgs.append({'role': 'user', 'content': 'Context:\n' + ex['context'] + '\n\nAnswer the question.'})
            msgs.append({'role': 'assistant', 'content': ex['answer']})
    msgs.append({'role': 'user', 'content': 'Context:\n' + context_block(question, intent, entities, calls) +
                 '\n\nAnswer the question.'})
    return msgs


# ------------------------------------------------------------------------------ prediction explanation
EXPLAIN_PERSONA = ('You are Toto, explaining a podium-prediction model to a Formula 1 fan who is not a data '
                   'scientist. You are careful and honest about what a model can and cannot say.')

EXPLAIN_TASK = """Task: explain this one prediction in 4-6 short sentences.
- Say what the model predicts (the probability as given, and podium or not podium at the threshold).
- Name the features that pushed the probability UP and those that pushed it DOWN, with roughly how much
  (use the contribution values given, in percentage points), using the plain-language feature names provided.
- Mention at most the 3 strongest in each direction. Do not mention any feature that is not in the lists.
- Never say a feature caused or will cause a podium. Say "the model relies on" / "pushed the model's estimate".
- You may add one supporting fact from the pre-race graph facts if it is relevant; quote its numbers exactly.
- Do not invent any value, fact or feature that is not in the context."""


def explanation_messages(context):
    return [{'role': 'system', 'content': EXPLAIN_PERSONA + '\n\n' + EXPLAIN_TASK},
            {'role': 'user', 'content': 'Context:\n' + context + '\n\nExplain the prediction.'}]
