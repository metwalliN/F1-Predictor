"""Parameterised Cypher template library (Section 2).

Rules every template follows:
- user text never enters the Cypher string: entities are resolved to graph ids first and passed as $parameters;
- Milestone 2 directions: (Result|Qualifying|PitStop)-[:FOR_RACE]->(Race), -[:BY_DRIVER]->(Driver),
  (Result|Qualifying)-[:FOR_CONSTRUCTOR]->(Constructor), (Result)-[:HAS_STATUS]->(Status),
  (Race)-[:AT_CIRCUIT]->(Circuit), (Race)-[:IN_SEASON]->(Season);
- named return columns, and every template ends with an aggregation or a LIMIT;
- only labels, relationships and properties that already exist in the Milestone 2 schema are used.

`graph` describes how a template's rows map back to nodes and relationships, so the interface can draw the
retrieved subgraph. Templates without `graph` return values (counts, averages) and are shown as values.
"""
from dataclasses import dataclass, field

# Status texts counted as a mechanical retirement: the Milestone 1 Q3 mapping (section 12c), unchanged.
MECHANICAL_STATUSES = [
    'Engine', 'Power Unit', 'Gearbox', 'Brakes', 'Suspension', 'Hydraulics', 'Electrical', 'Power loss', 'Oil leak',
    'Overheating', 'Water pressure', 'Mechanical', 'Turbo', 'ERS', 'Fuel pressure', 'Exhaust', 'Electronics',
    'Water leak', 'Battery', 'Radiator', 'Transmission', 'Steering', 'Oil pressure', 'Technical', 'Clutch', 'Throttle',
    'Driveshaft', 'Fuel leak', 'Drivetrain', 'Brake duct', 'Seat', 'Spark plugs', 'Water pump', 'Cooling system',
    'Fuel pump', 'Differential']


@dataclass
class Template:
    id: str
    intent: str
    description: str
    cypher: str
    required: list
    scope: str                       # what the figures cover and exclude (passed to the LLM)
    hops: str
    graph: dict = field(default_factory=dict)
    fixed_params: dict = field(default_factory=dict)


T = {}


def _add(t):
    assert t.id not in T
    lines = [l.strip() for l in t.cypher.strip().splitlines() if l.strip()]
    last = lines[-1].upper()
    has_agg = any(f in ' '.join(lines).lower() for f in ('count(', 'sum(', 'avg(', 'min(', 'max(', 'collect('))
    assert last.startswith('LIMIT') or 'LIMIT' in last or has_agg, f'{t.id}: must end with LIMIT or aggregate'
    T[t.id] = t


SEASON_RANGE = 's.year >= $year_from AND s.year <= $year_to'

# Note: count(DISTINCT ...) is written as the last aggregate of a RETURN. Neo4j does not care about the order;
# the embedded test engine (Kùzu) mis-computes aggregates that follow a DISTINCT aggregate.

# ---------------------------------------------------------------- 1. Driver performance
_add(Template(
    'driver_season_stats', 'driver_stats',
    'Wins, podiums, points, poles and average finish for one driver over a season or a range of seasons',
    f"""
MATCH (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season)
WHERE d.driverId = $driverId AND {SEASON_RANGE}
RETURN d.forename + ' ' + d.surname AS driver,
       min(s.year) AS first_season, max(s.year) AS last_season,
       sum(CASE WHEN res.positionOrder = 1 THEN 1 ELSE 0 END) AS wins,
       sum(CASE WHEN res.positionOrder <= 3 THEN 1 ELSE 0 END) AS podiums,
       sum(res.points) AS points,
       sum(CASE WHEN res.grid = 1 THEN 1 ELSE 0 END) AS starts_from_pole,
       sum(CASE WHEN res.position IS NULL THEN 1 ELSE 0 END) AS not_classified,
       round(avg(res.position), 2) AS avg_classified_finish,
       count(DISTINCT r) AS races_entered
""",
    ['driverId', 'year_from', 'year_to'],
    'Grand Prix results only: sprint races are not in the graph; points are race points as recorded '
    '(official championship totals can differ where sprint points or dropped-score rules applied). '
    'A podium is a finishing position of 1 to 3; average finish uses classified finishes only.',
    'Result -> Race -> Season (2 hops) + Result -> Driver'))

_add(Template(
    'driver_at_circuit', 'driver_at_circuit',
    "One driver's record at one circuit, optionally limited to a range of seasons",
    """
MATCH (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(c:Circuit)
WHERE d.driverId = $driverId AND c.circuitId = $circuitId AND r.year >= $year_from AND r.year <= $year_to
RETURN d.forename + ' ' + d.surname AS driver, c.circuit_name AS circuit,
       min(r.year) AS first_year, max(r.year) AS last_year,
       sum(CASE WHEN res.positionOrder = 1 THEN 1 ELSE 0 END) AS wins,
       sum(CASE WHEN res.positionOrder <= 3 THEN 1 ELSE 0 END) AS podiums,
       sum(CASE WHEN res.grid = 1 THEN 1 ELSE 0 END) AS starts_from_pole,
       min(res.positionOrder) AS best_finish,
       round(avg(res.position), 2) AS avg_classified_finish,
       count(DISTINCT r) AS races_entered
""",
    ['driverId', 'circuitId', 'year_from', 'year_to'],
    'Grand Prix results at this circuit only, for the seasons in the question. '
    'Average finish uses classified finishes only.',
    'Result -> Race -> Circuit (2 hops) + Result -> Driver'))

# ---------------------------------------------------------------- 2. Constructor performance
_add(Template(
    'constructor_stats', 'constructor_stats',
    'Points, wins, podiums and reliability (classified rate, mechanical retirements) for a constructor over seasons',
    f"""
MATCH (k:Constructor)<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season)
WHERE k.constructorId = $constructorId AND {SEASON_RANGE}
MATCH (res)-[:HAS_STATUS]->(st:Status)
WITH k, s, r, res, st
RETURN k.constructor_name AS constructor,
       min(s.year) AS first_season, max(s.year) AS last_season,
       count(res) AS car_entries,
       sum(CASE WHEN res.positionOrder = 1 THEN 1 ELSE 0 END) AS wins,
       sum(CASE WHEN res.positionOrder <= 3 THEN 1 ELSE 0 END) AS podiums,
       sum(res.points) AS points,
       sum(CASE WHEN res.position IS NOT NULL THEN 1 ELSE 0 END) AS classified_finishes,
       round(1.0 * sum(CASE WHEN res.position IS NOT NULL THEN 1 ELSE 0 END) / count(res), 3) AS classified_rate,
       sum(CASE WHEN st.status IN $mechanical_statuses THEN 1 ELSE 0 END) AS mechanical_retirements,
       count(DISTINCT r) AS races
""",
    ['constructorId', 'year_from', 'year_to'],
    'Every car entry of this constructor in Grand Prix results for these seasons. Reliability = classified '
    'finishes / car entries (the Milestone 2 definition). Mechanical retirements use the Milestone 1 status mapping '
    '(engine, gearbox, hydraulics, ...); crashes and ambiguous statuses such as "Retired" are not counted. '
    'Points are summed race points (sprints not included).',
    'Result -> Race -> Season (2 hops) + Result -> Constructor, Result -> Status',
    fixed_params={'mechanical_statuses': MECHANICAL_STATUSES}))

# ---------------------------------------------------------------- 3. Head-to-head
_H2H = """
MATCH (d1:Driver)<-[:BY_DRIVER]-(r1:Result)-[:FOR_RACE]->(r:Race)<-[:FOR_RACE]-(r2:Result)-[:BY_DRIVER]->(d2:Driver)
WHERE d1.driverId = $driver1Id AND d2.driverId = $driver2Id AND r.year >= $year_from AND r.year <= $year_to
MATCH (r1)-[:FOR_CONSTRUCTOR]->(k:Constructor)<-[:FOR_CONSTRUCTOR]-(r2){extra}
RETURN d1.surname AS driver1, d2.surname AS driver2, k.constructor_name AS constructor,
       min(r.year) AS first_year, max(r.year) AS last_year,
       sum(CASE WHEN r1.positionOrder < r2.positionOrder THEN 1 ELSE 0 END) AS driver1_ahead,
       sum(CASE WHEN r2.positionOrder < r1.positionOrder THEN 1 ELSE 0 END) AS driver2_ahead,
       sum(r1.points) AS driver1_points,
       sum(r2.points) AS driver2_points,
       count(DISTINCT r) AS shared_races
ORDER BY shared_races DESC
LIMIT 5
"""
_add(Template(
    'head_to_head', 'head_to_head',
    'Two drivers in the same car (teammates): races shared and who finished ahead, per constructor',
    _H2H.format(extra=''),
    ['driver1Id', 'driver2Id', 'year_from', 'year_to'],
    'Only races where both drivers drove for the same constructor. "Ahead" compares the final classification order '
    '(positionOrder), so a retirement counts as finishing behind.',
    'Result -> Race <- Result, Result -> Constructor <- Result (2 hops each side)'))
_add(Template(
    'head_to_head_at_constructor', 'head_to_head',
    'Head-to-head of two teammates restricted to one constructor',
    _H2H.format(extra='\nWHERE k.constructorId = $constructorId'),
    ['driver1Id', 'driver2Id', 'constructorId', 'year_from', 'year_to'],
    'Only races where both drivers drove for this constructor. "Ahead" compares the final classification order '
    '(positionOrder), so a retirement counts as finishing behind.',
    'Result -> Race <- Result, Result -> Constructor <- Result (2 hops each side)'))

# ---------------------------------------------------------------- 4. Race and circuit facts
_add(Template(
    'race_podium', 'race_result',
    'Top finishers of one race (winner and podium by default)',
    """
MATCH (r:Race)<-[:FOR_RACE]-(res:Result)-[:BY_DRIVER]->(d:Driver)
WHERE r.raceId = $raceId AND res.positionOrder <= $top_n
MATCH (res)-[:FOR_CONSTRUCTOR]->(k:Constructor)
MATCH (res)-[:HAS_STATUS]->(st:Status)
MATCH (r)-[:AT_CIRCUIT]->(c:Circuit)
RETURN r.name AS race, r.year AS year, c.circuit_name AS circuit, res.positionOrder AS finish,
       d.forename + ' ' + d.surname AS driver, k.constructor_name AS constructor,
       res.grid AS grid, res.points AS points, res.laps AS laps, st.status AS status
ORDER BY finish
LIMIT 10
""",
    ['raceId', 'top_n'],
    'The official classification of this one race (positionOrder).',
    'Result -> Race -> Circuit, Result -> Driver, Result -> Constructor, Result -> Status',
    graph={'kind': 'race_result'}))

_add(Template(
    'circuit_info', 'circuit_info',
    'Where a circuit is and how many championship races it has held',
    """
MATCH (r:Race)-[:AT_CIRCUIT]->(c:Circuit)
WHERE c.circuitId = $circuitId
RETURN c.circuit_name AS circuit, c.circuit_loc AS location, c.circuit_country AS country,
       count(r) AS races_held, min(r.year) AS first_year, max(r.year) AS last_year
""",
    ['circuitId'],
    'Championship Grand Prix held at this circuit from 1950 to 2024.',
    'Race -> Circuit'))

_add(Template(
    'circuit_top_winners', 'circuit_info',
    'Drivers with the most wins at a circuit',
    """
MATCH (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(c:Circuit)
WHERE c.circuitId = $circuitId AND res.positionOrder = 1
RETURN c.circuit_name AS circuit, d.forename + ' ' + d.surname AS driver, count(r) AS wins,
       min(r.year) AS first_win, max(r.year) AS last_win
ORDER BY wins DESC, last_win DESC
LIMIT 5
""",
    ['circuitId'],
    'Wins in championship Grand Prix at this circuit, 1950 to 2024 (top 5 drivers).',
    'Result -> Race -> Circuit (2 hops) + Result -> Driver'))

# ---------------------------------------------------------------- 5. Qualifying and grid
_add(Template(
    'race_grid_gains', 'qualifying_grid',
    'Drivers who gained the most places from their starting grid slot to the finish in one race',
    """
MATCH (r:Race)<-[:FOR_RACE]-(res:Result)-[:BY_DRIVER]->(d:Driver)
WHERE r.raceId = $raceId AND res.grid > 0 AND res.position IS NOT NULL
MATCH (res)-[:FOR_CONSTRUCTOR]->(k:Constructor)
RETURN r.name AS race, r.year AS year, d.forename + ' ' + d.surname AS driver, k.constructor_name AS constructor,
       res.grid AS grid, res.position AS finish, res.grid - res.position AS places_gained
ORDER BY places_gained DESC, finish
LIMIT 5
""",
    ['raceId'],
    'Classified finishers of this race who started from a grid slot (pit-lane starts, grid 0, and retirements '
    'are excluded). Places gained = grid slot minus finishing position.',
    'Result -> Race, Result -> Driver, Result -> Constructor',
    graph={'kind': 'race_result'}))

_add(Template(
    'race_qualifying_top', 'qualifying_grid',
    'Pole position and the top of the qualifying order for one race',
    """
MATCH (r:Race)<-[:FOR_RACE]-(q:Qualifying)-[:BY_DRIVER]->(d:Driver)
WHERE r.raceId = $raceId AND q.position <= 3
MATCH (q)-[:FOR_CONSTRUCTOR]->(k:Constructor)
RETURN r.name AS race, r.year AS year, q.position AS qualifying_position,
       d.forename + ' ' + d.surname AS driver, k.constructor_name AS constructor,
       q.q1 AS q1, q.q2 AS q2, q.q3 AS q3
ORDER BY qualifying_position
LIMIT 3
""",
    ['raceId'],
    'Qualifying classification of this race (qualifying records exist from 1994). Pole = qualifying position 1; '
    'the actual starting grid can differ after penalties.',
    'Qualifying -> Race, Qualifying -> Driver, Qualifying -> Constructor',
    graph={'kind': 'qualifying'}))

_add(Template(
    'driver_qualifying_record', 'qualifying_grid',
    "A driver's qualifying record over seasons: poles, front rows, Q3 appearances, average qualifying position",
    f"""
MATCH (d:Driver)<-[:BY_DRIVER]-(q:Qualifying)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season)
WHERE d.driverId = $driverId AND {SEASON_RANGE}
RETURN d.forename + ' ' + d.surname AS driver, min(s.year) AS first_season, max(s.year) AS last_season,
       count(q) AS qualifying_sessions,
       sum(CASE WHEN q.position = 1 THEN 1 ELSE 0 END) AS poles,
       sum(CASE WHEN q.position <= 2 THEN 1 ELSE 0 END) AS front_row_qualifying,
       sum(CASE WHEN q.q3 IS NOT NULL THEN 1 ELSE 0 END) AS q3_appearances,
       round(avg(q.position), 2) AS avg_qualifying_position
""",
    ['driverId', 'year_from', 'year_to'],
    'Qualifying records only (they exist from 1994). Q3 exists only from 2006, so earlier seasons have no Q3 '
    'appearances. Pole = qualifying position 1, before any grid penalty.',
    'Qualifying -> Race -> Season (2 hops) + Qualifying -> Driver'))

# ---------------------------------------------------------------- 6. Pit stops and strategy
_add(Template(
    'race_fastest_pit_stops', 'pit_stops',
    'Fastest pit stops of one race',
    """
MATCH (r:Race)<-[:FOR_RACE]-(p:PitStop)-[:BY_DRIVER]->(d:Driver)
WHERE r.raceId = $raceId
RETURN r.name AS race, r.year AS year, d.forename + ' ' + d.surname AS driver,
       p.stop AS stop_number, p.lap AS lap, p.duration AS duration_seconds, p.milliseconds AS milliseconds
ORDER BY milliseconds
LIMIT 5
""",
    ['raceId'],
    'Pit-stop records exist from 2011. The duration is the time from pit-lane entry to exit, not the stationary '
    'time of the car.',
    'PitStop -> Race, PitStop -> Driver',
    graph={'kind': 'pit_stop'}))

_add(Template(
    'race_pit_summary', 'pit_stops',
    'How many stops drivers made in one race (average, most, fewest)',
    """
MATCH (r:Race)<-[:FOR_RACE]-(p:PitStop)-[:BY_DRIVER]->(d:Driver)
WHERE r.raceId = $raceId
WITH r, d, count(p) AS stops
RETURN r.name AS race, r.year AS year, count(d) AS drivers_who_stopped,
       sum(stops) AS total_stops, round(avg(stops), 2) AS avg_stops_per_driver,
       max(stops) AS most_stops, min(stops) AS fewest_stops
""",
    ['raceId'],
    'Pit-stop records exist from 2011; drivers who never stopped are not counted.',
    'PitStop -> Race, PitStop -> Driver'))

_add(Template(
    'driver_pit_stops', 'pit_stops',
    "A driver's pit stops over seasons: average stops per race and fastest stop",
    f"""
MATCH (d:Driver)<-[:BY_DRIVER]-(p:PitStop)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season)
WHERE d.driverId = $driverId AND {SEASON_RANGE}
WITH d, r, count(p) AS stops, min(p.milliseconds) AS fastest_ms
RETURN d.forename + ' ' + d.surname AS driver, min(r.year) AS first_season, max(r.year) AS last_season,
       count(r) AS races_with_stops, sum(stops) AS total_stops,
       round(1.0 * sum(stops) / count(r), 2) AS avg_stops_per_race,
       min(fastest_ms) AS fastest_stop_ms
""",
    ['driverId', 'year_from', 'year_to'],
    'Pit-stop records exist from 2011; races in which the driver did not stop are not counted. Durations are '
    'pit-lane times in milliseconds.',
    'PitStop -> Race -> Season (2 hops) + PitStop -> Driver'))

# ---------------------------------------------------------------- 7. Season and history
_add(Template(
    'season_driver_points', 'season_summary',
    'Drivers with the most race points (and wins) in one season',
    """
MATCH (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season)
WHERE s.year = $year
RETURN s.year AS season, d.forename + ' ' + d.surname AS driver, sum(res.points) AS points,
       sum(CASE WHEN res.positionOrder = 1 THEN 1 ELSE 0 END) AS wins
ORDER BY points DESC, wins DESC
LIMIT 5
""",
    ['year'],
    'Sum of Grand Prix race points in that season (top 5). Sprint points and dropped-score rules are not included, '
    'so totals can differ from the official championship table.',
    'Result -> Race -> Season (2 hops) + Result -> Driver'))

_add(Template(
    'season_constructor_points', 'season_summary',
    'Constructors with the most race points in one season',
    """
MATCH (k:Constructor)<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season)
WHERE s.year = $year
RETURN s.year AS season, k.constructor_name AS constructor, sum(res.points) AS points,
       sum(CASE WHEN res.positionOrder = 1 THEN 1 ELSE 0 END) AS wins
ORDER BY points DESC, wins DESC
LIMIT 5
""",
    ['year'],
    'Sum of Grand Prix race points of all the constructor\'s cars in that season (top 5). Sprint points are not '
    'included, and the official constructors\' table can differ.',
    'Result -> Race -> Season (2 hops) + Result -> Constructor'))

_add(Template(
    'season_overview', 'season_summary',
    'How many races, drivers and constructors took part in one season',
    """
MATCH (k:Constructor)<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season)
WHERE s.year = $year
MATCH (res)-[:BY_DRIVER]->(d:Driver)
RETURN s.year AS season, count(DISTINCT r) AS races, count(DISTINCT d) AS drivers,
       count(DISTINCT k) AS constructors
""",
    ['year'],
    'Every driver and constructor with at least one Grand Prix result entry in that season.',
    'Result -> Race -> Season (2 hops) + Result -> Driver, Result -> Constructor'))

_add(Template(
    'season_team_changes', 'season_summary',
    'Drivers who raced in both seasons but for a different constructor',
    """
MATCH (d:Driver)<-[:BY_DRIVER]-(a:Result)-[:FOR_RACE]->(:Race)-[:IN_SEASON]->(sa:Season)
WHERE sa.year = $year_from
MATCH (a)-[:FOR_CONSTRUCTOR]->(ka:Constructor)
WITH d, collect(DISTINCT ka.constructor_name) AS teams_before
MATCH (d)<-[:BY_DRIVER]-(b:Result)-[:FOR_RACE]->(:Race)-[:IN_SEASON]->(sb:Season)
WHERE sb.year = $year_to
MATCH (b)-[:FOR_CONSTRUCTOR]->(kb:Constructor)
WITH d, teams_before, collect(DISTINCT kb.constructor_name) AS teams_after
WHERE none(t IN teams_after WHERE t IN teams_before)
RETURN d.forename + ' ' + d.surname AS driver, teams_before, teams_after
ORDER BY driver
LIMIT 30
""",
    ['year_from', 'year_to'],
    'Drivers with a result in both seasons whose constructor(s) in the second season share no name with their '
    'constructor(s) in the first. A rebrand (e.g. Renault to Alpine) counts as a change.',
    'Result -> Race -> Season (2 hops) + Result -> Constructor, for both seasons'))

# ---------------------------------------------------------------- Prediction support: pre-race facts only
# Restricted to races strictly before the scored race: earlier season, or same season and an earlier round.
BEFORE_RACE = '(r.year < $year OR (r.year = $year AND r.round < $round))'

_add(Template(
    'prerace_driver_at_circuit', 'podium_prediction',
    "The driver's record at this circuit in races before the scored race",
    f"""
MATCH (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(c:Circuit)
WHERE d.driverId = $driverId AND c.circuitId = $circuitId AND {BEFORE_RACE}
RETURN d.forename + ' ' + d.surname AS driver, c.circuit_name AS circuit,
       sum(CASE WHEN res.positionOrder = 1 THEN 1 ELSE 0 END) AS earlier_wins_here,
       sum(CASE WHEN res.positionOrder <= 3 THEN 1 ELSE 0 END) AS earlier_podiums_here,
       min(res.positionOrder) AS best_earlier_finish_here,
       count(DISTINCT r) AS earlier_races_here
""",
    ['driverId', 'circuitId', 'year', 'round'],
    'Only races at this circuit held before the scored race.',
    'Result -> Race -> Circuit (2 hops) + Result -> Driver'))

_add(Template(
    'prerace_constructor_reliability', 'podium_prediction',
    "The constructor's finishing record over the previous and current season, before the scored race",
    f"""
MATCH (k:Constructor)<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season)
WHERE k.constructorId = $constructorId AND s.year >= $year - 1 AND {BEFORE_RACE}
MATCH (res)-[:HAS_STATUS]->(st:Status)
RETURN k.constructor_name AS constructor, min(s.year) AS from_season,
       count(res) AS car_entries,
       sum(CASE WHEN res.positionOrder <= 3 THEN 1 ELSE 0 END) AS podiums,
       round(1.0 * sum(CASE WHEN res.position IS NOT NULL THEN 1 ELSE 0 END) / count(res), 3) AS classified_rate,
       sum(CASE WHEN st.status IN $mechanical_statuses THEN 1 ELSE 0 END) AS mechanical_retirements
""",
    ['constructorId', 'year', 'round'],
    'Car entries of this constructor from the start of the previous season up to (not including) the scored race.',
    'Result -> Race -> Season (2 hops) + Result -> Constructor, Result -> Status',
    fixed_params={'mechanical_statuses': MECHANICAL_STATUSES}))
