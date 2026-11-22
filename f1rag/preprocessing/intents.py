"""The intent set: 9 question types (every row of the brief's table is covered), plus podium prediction and
out of scope. Each intent has a description, the entities it needs, example phrasings (used to build both
classifiers, and never reused in the labelled test set), and the rule-based patterns."""

INTENTS = {
    'driver_stats': {
        'row': 'Driver performance',
        'description': "A driver's wins, podiums, points, poles or average finish over a season, a range of seasons "
                       "or a whole career.",
        'required': ['driver'], 'optional': ['season'],
        'examples': [
            'How many podiums did Leclerc get in 2022?',
            'How many wins does Lewis Hamilton have in his career?',
            'How many points did Alonso score in 2023?',
            "What was Bottas's average finishing position in 2019?",
            'How many races did Kimi Raikkonen win between 2005 and 2007?',
            'Give me Sebastian Vettel stats for 2013',
            'How many times did Max Verstappen start from pole in 2023?',
            'How many points did he score in 2019?',
        ],
        'patterns': [r'\b(how many|number of|total)\b.*\b(wins?|won|podiums?|points|poles?|victories|races)\b',
                     r'\b(stats|statistics|record|career|average finish(ing)?( position)?)\b',
                     r'\b(score|scored|win|won)\b.*\b(in|during|between|from)\b'],
    },
    'driver_at_circuit': {
        'row': 'Driver performance',
        'description': "One driver's record at one circuit: wins, podiums, poles, best and average finish there.",
        'required': ['driver', 'circuit'], 'optional': ['season'],
        'examples': [
            'How many wins did Hamilton have at Silverstone before 2020?',
            "What is Verstappen's record at Zandvoort?",
            'How did Leclerc do at Monaco over the years?',
            'What was Alonso’s best finish at Suzuka?',
            'How many podiums has Sainz got at Monza?',
            'Average finish of Russell at the Hungaroring',
        ],
        'patterns': [r'\b(at|in)\b.*\b(record|wins?|won|podiums?|best finish|average finish|results?)\b',
                     r'\b(record|wins?|podiums?|best finish|average finish)\b.*\bat\b'],
    },
    'constructor_stats': {
        'row': 'Constructor performance',
        'description': "A constructor's points, wins, podiums or reliability (finishing rate, mechanical "
                       "retirements) over a season or a period.",
        'required': ['constructor'], 'optional': ['season'],
        'examples': [
            'How reliable was Williams between 2014 and 2021?',
            'How many points did Ferrari score in 2022?',
            'How many wins did McLaren get in 2024?',
            'How many mechanical retirements did Haas have from 2016 to 2020?',
            "What was Red Bull's finishing rate in 2018?",
            'How many podiums did Mercedes get between 2014 and 2020?',
        ],
        'patterns': [r'\b(reliab\w*|retire\w*|mechanical|finishing rate|dnf|breakdowns?|failures?)\b',
                     r'\b(team|constructor)\b.*\b(points|wins?|podiums?)\b'],
    },
    'head_to_head': {
        'row': 'Head-to-head',
        'description': 'Two drivers compared as teammates in shared races: who finished ahead more often.',
        'required': ['driver', 'driver2'], 'optional': ['constructor', 'season'],
        'examples': [
            'Who finished ahead more often, Norris or Ricciardo at McLaren?',
            'Compare Hamilton and Bottas as teammates',
            'Leclerc vs Sainz head to head',
            'Did Vettel beat Webber more often when they were teammates?',
            'Head-to-head between Russell and Hamilton at Mercedes',
            'Who won the teammate battle, Ocon or Gasly?',
        ],
        'patterns': [r'\b(vs\.?|versus|head[ -]?to[ -]?head|teammates?|team ?mates?|compare|comparison)\b',
                     r'\bahead\b.*\bmore often\b', r'\b(beat|outscored|outqualified)\b'],
    },
    'race_result': {
        'row': 'Race and circuit facts',
        'description': 'The winner, podium or top finishers of one specific race (a Grand Prix in a given year).',
        'required': ['race'], 'optional': [],
        'examples': [
            'Who won the 2019 German Grand Prix?',
            'Who was on the podium at the 2021 Italian GP?',
            'Results of the 2008 British Grand Prix',
            'Who finished third in Monaco 2017?',
            'Give me the top 10 of the 2023 Las Vegas Grand Prix',
            'Who won at Interlagos in 2016?',
        ],
        'patterns': [r'\b(who won|winner|won the|podium|finished (first|second|third|1st|2nd|3rd)|result|results|'
                     r'classification|top (3|three|5|five|10|ten))\b'],
    },
    'circuit_info': {
        'row': 'Race and circuit facts',
        'description': 'Facts about a circuit: where it is, how many races it has held, since when, and who has won '
                       'there most.',
        'required': ['circuit'], 'optional': [],
        'examples': [
            'Where is the Hungaroring?',
            'How many Grands Prix have been held at Spa?',
            'Which country is Suzuka in?',
            'Who has the most wins at Monza?',
            'When was the first race at Silverstone?',
            'Tell me about the Circuit of the Americas',
        ],
        'patterns': [r'\b(where is|located|location|which country|what country|which city)\b',
                     r'\b(races|grands? prix) (have been |were )?(held|hosted)\b', r'\bmost wins at\b',
                     r'\b(first|last) (race|grand prix) (at|in)\b', r'\btell me about\b'],
    },
    'qualifying_grid': {
        'row': 'Qualifying and grid',
        'description': 'Qualifying and starting grid: pole positions, Q3 appearances, qualifying order, and places '
                       'gained from the grid to the finish.',
        'required': [], 'optional': ['race', 'driver', 'season'],
        'examples': [
            'Which driver gained the most places from the grid at Monza in 2021?',
            'Who took pole for the 2022 Japanese Grand Prix?',
            'How many poles did Leclerc get in 2022?',
            'How many times did Norris reach Q3 in 2023?',
            'Who qualified on the front row at the 2020 Turkish GP?',
            "What was Hamilton's average qualifying position in 2021?",
        ],
        'patterns': [r'\b(pole|poles|pole position|q3|q2|q1|qualif\w*|front row|grid)\b',
                     r'\b(gained|gain|made up|climbed|places)\b'],
    },
    'pit_stops': {
        'row': 'Pit stops and strategy',
        'description': 'Pit stops: fastest stops in a race, number of stops, average stops per race for a driver.',
        'required': [], 'optional': ['race', 'driver', 'season'],
        'examples': [
            'What was the fastest pit stop at the 2023 Qatar Grand Prix?',
            'How many pit stops did drivers make on average in the 2021 Italian GP?',
            'How many times did Hamilton pit in 2022?',
            'Average number of stops per race for Norris in 2024',
            'Quickest stops at Monza 2019',
            'What strategy did teams use in the 2022 Bahrain GP, how many stops?',
        ],
        'patterns': [r'\b(pit|pits|pitted|pitting|pit ?stops?|stops|strategy|strategies|tyre change)\b'],
    },
    'season_summary': {
        'row': 'Season and history',
        'description': 'A whole season: points leader and top drivers or teams, how many drivers or races took '
                       'part, and drivers who changed teams between two seasons.',
        'required': ['season'], 'optional': [],
        'examples': [
            'Which drivers changed teams between 2020 and 2021?',
            'Who scored the most points in 2021?',
            'How many drivers took part in the 2016 season?',
            'Which team scored the most points in 2019?',
            'Who were the top drivers in 1988?',
            'How many races were there in 2023?',
        ],
        'patterns': [r'\b(changed teams?|switched teams?|moved teams?|team changes?|transfers?)\b',
                     r'\b(most points|points leader|champion|championship|standings|top drivers|top teams)\b',
                     r'\bhow many (drivers|races|teams|constructors)\b.*\b(season|in \d{4})\b'],
    },
    'podium_prediction': {
        'row': 'Prediction path',
        'description': 'Predict whether a driver will finish on the podium in a given race, optionally from '
                       'pre-race facts such as the starting position.',
        'required': ['driver', 'race'], 'optional': ['grid', 'quali'],
        'examples': [
            'Predict Norris at the 2024 Miami Grand Prix',
            'Will Leclerc finish on the podium at Monaco 2024 starting from pole?',
            'What are the chances of Hamilton getting a podium at the 2023 British GP from P7?',
            'Podium probability for Piastri in the 2024 Hungarian Grand Prix',
            'Could Alonso make the podium at the 2023 Saudi Arabian GP if he starts 3rd?',
            'Run the podium model for Russell at Silverstone 2024',
        ],
        'patterns': [r'\b(predict\w*|prediction|chances?|probability|likely|will .* (podium|finish)|'
                     r'podium model|forecast|could .* (make|get|reach) the podium)\b'],
    },
    'out_of_scope': {
        'row': 'Refusal',
        'description': 'Questions the knowledge graph cannot answer (car design, tyres, weather, salaries, lap '
                       'times, sprint races, news) and questions that are not about Formula 1.',
        'required': [], 'optional': [],
        'examples': [
            'Who designed the 2023 Red Bull car?',
            'What tyre compound did Hamilton use at Monza?',
            'What will the weather be at the next race?',
            'How much does Verstappen earn?',
            'What is the capital of France?',
            'Write me a poem about football',
            'Who won the 2023 Qatar sprint race?',
            'What was the fastest lap time at Spa in 2022?',
        ],
        'patterns': [],
    },
}

RETRIEVAL_INTENTS = [i for i in INTENTS if i not in ('podium_prediction', 'out_of_scope')]

# Topics the graph has no data for (evidence for out-of-scope, together with the classifier's vote).
OUT_OF_SCOPE_TOPICS = (
    r'\b(design(ed|er|s)?|engineer(s)?|aero\w*|downforce|tyres?|tires?|compounds?|weather|rain|temperature|salary|'
    r'salaries|earn\w*|paid|contract\w*|net worth|sprint\w*|lap times?|fastest laps?|lap records?|top speed|speed trap|'
    r'penalt(y|ies)|crash(ed|es)?|accident|news|rumou?rs?|tickets?|merch\w*|engine supplier|budget|sponsor\w*|'
    r'livery|liveries|f2|formula 2|formula e|indycar|motogp|nascar|wec|le mans)\b')

# Words that place a message in the Formula 1 domain even when no entity is recognised.
F1_VOCAB = (r'\b(f1|formula ?(1|one)|grand prix|gp|race|races|raced|racing|driver|drivers|team|teams|constructor|'
            r'constructors|podium|pole|qualif\w*|grid|pit|pits|season|championship|champion|circuit|track|lap|laps|'
            r'win|wins|won|points|teammate)\b')

FOLLOW_UP = (r'^\s*(and|what about|how about|same for|and in|and for|now|ok,? and|what if)\b|'
             r'^\s*((in )?(19[5-9]\d|20[0-4]\d)|(19[5-9]\d|20[0-4]\d)\s*(-|to)\s*(19[5-9]\d|20[0-4]\d))\s*\??\s*$')
