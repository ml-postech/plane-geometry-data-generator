import random

import ddar
import graph as gh
import pretty as pt
import problem as pr
import geometry as gm


def run_query(client, query):
  assistant_prompt = "You are an geometry problem solving expert."
  temperature=0.0
  max_tokens=2048
  frequency_penalty=0.0

  user_prompt = query  # "Paraphrase the following geometry problem to be fluent: " + problem
  message=[{"role": "assistant", "content": assistant_prompt}, {"role": "user", "content": user_prompt}]
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = message,
    temperature=temperature,
    max_tokens=max_tokens,
    frequency_penalty=frequency_penalty
  )
  return response.choices[0].message.content


def generate_vars(cur_idx, n, alphabet=False):
  ret = []
  for i in range(n):
    if alphabet:
      ret.append(chr(97 + cur_idx + i))
    else:
      ret.append(f"x{cur_idx + i}")

  return ret


def sample_problem_txt(DEFINITIONS, object_defs, filtered_defs, n_clauses=2):
  object_def_idx = random.randrange(len(object_defs))
  object_def = DEFINITIONS[object_defs[object_def_idx]]

  if object_def.construction.name == 's_angle':
    vars = generate_vars(0, 3)
    angle = random.randrange(20, 160)
    clauses = [f"{vars[0]} {vars[1]} = segment", f"{vars[2]} = s_angle {vars[0]} {vars[1]} {vars[2]} {angle}"]
  elif object_def.construction.name == 'triangle12':
    vars = generate_vars(0, 3)
    r1 = random.randint(1, 20)
    r2 = random.randint(r1, 20)
    clauses = [f"{vars[0]} {vars[1]} {vars[2]} = triangle12 {vars[0]} {vars[1]} {vars[2]} {r1} {r2}"]
  else:
    n_vars = len(object_def.construction.args)
    vars = generate_vars(0, n_vars)
    clauses = [" ".join(vars) + " = " + object_def.construction.name]

  for _ in range(n_clauses):
    n_vars = random.randrange(2) + 1
    n_cons = 1

    new_vars = generate_vars(len(vars), n_vars)
    selected_cons = []
    cnt = 0
    angle_dist = list(range(15, 71))
    while cnt < n_cons:
      con = random.sample(filtered_defs[n_vars], k=1)[0]
      con_args = con.construction.args
      if len(con_args) - n_vars > len(vars):
        continue
      n_req_vars = len(con_args) if con.construction.name != 's_angle' else 3
      given_vars = random.sample(vars, k=n_req_vars - n_vars)
      given_vars.sort()
      if con.construction.name == 's_angle':
        deg = random.sample(angle_dist, k=1)[0]
        given_vars = given_vars + new_vars + [str(deg)]
      selected_cons.append(con.construction.name + f" " + " ".join(given_vars))
      cnt += 1

    if len(selected_cons) > 0:
      clause = " ".join(new_vars) + " = " + ", ".join(selected_cons)
      clauses.append(clause)
      vars = vars + new_vars
      
  problem_txt = "; ".join(clauses)

  return problem_txt, vars

def generate_problem(DEFINITIONS, object_defs, filtered_defs, n_clauses=2, GOALS=None):
  while True:
    try:
      problem_txt, vars = sample_problem_txt(DEFINITIONS, object_defs, filtered_defs, n_clauses)
     
      if GOALS is None:
        p = pr.Problem.from_txt(problem_txt)
      else:
        goal_idx = random.randint(0, len(GOALS) - 1)
        goal_name, goal_n_vars = GOALS[goal_idx]
        goal_vars = random.sample(vars, k=goal_n_vars)
        if goal_name == 'acompute2':
          goal_name = 'acompute'
          goal_vars.append(goal_vars[0])
        goal_txt = goal_name + " " + " ".join(goal_vars)
        problem_txt = problem_txt + " ? " + goal_txt

      p = pr.Problem.from_txt(problem_txt, translate=True, shuffle=True)
      g, deps = gh.Graph.build_problem(p, DEFINITIONS)

      # for dep in deps:
      #   if dep.name == "rconst":
      #     a, b, c, d, ratio = dep.args
      #     _ = g.get_line_thru_pair(a, b)
      #     _ = g.get_line_thru_pair(c, d)

      return g, deps, p, problem_txt, vars
    except:
      continue

def natural_language_statement(logical_statement: pr.Dependency) -> str:
  """Convert logical_statement to natural language.

  Args:
    logical_statement: pr.Dependency with .name and .args

  Returns:
    a string of (pseudo) natural language of the predicate for human reader.
  """
  names = [a.name.upper() for a in logical_statement.args]
  names = [(n[0] + '_' + n[1:]) if len(n) > 1 else n for n in names]
  return pt.pretty_nl(logical_statement.name, names, formal=True)


def proof_step_string(
    proof_step: pr.Dependency, refs: dict[tuple[str, ...], int], last_step: bool
) -> str:
  """Translate proof to natural language.

  Args:
    proof_step: pr.Dependency with .name and .args
    refs: dict(hash: int) to keep track of derived predicates
    last_step: boolean to keep track whether this is the last step.

  Returns:
    a string of (pseudo) natural language of the proof step for human reader.
  """
  premises, [conclusion] = proof_step
  premises_nl = ' & '.join(
    [natural_language_statement(p) for p in premises]
  )

  if not premises:
    premises_nl = 'similarly'

  refs[conclusion.hashed()] = len(refs)
  conclusion_nl = natural_language_statement(conclusion)
  return f'{premises_nl} \u21d2 {conclusion_nl}'


def generate_solution(g, RULES, p, max_level=5):
  ddar.solve(g, RULES, p, max_level=max_level)
  goal_args = g.names2nodes(p.goal.args)
  if not g.check(p.goal.name, goal_args):
    raise Exception

  a, b, c, d = goal_args
  ab = g._get_line(a, b)
  cd = g._get_line(c, d)
  answer = None
  for ang0 in g.aconst.values():
    for ang in ang0.val.neighbors(gm.Angle):
      d1, d2 = ang.directions
      if ab.val == d1 and cd.val == d2:
        answer = ang0.name
        p.goal = pr.Construction.from_txt(f"aconst {a.name} {b.name} {c.name} {d.name} {ang0.name}")
        break

  _, _, proof_steps, refs = ddar.get_proof_steps(g, p.goal, merge_trivials=True)
  if len(proof_steps) == 0:
    raise Exception

  # solution = ""
  solution = []
  for i, step in enumerate(proof_steps):
    _, [con] = step
    nl = proof_step_string(step, refs, last_step=i == len(proof_steps) - 1)
    # solution += nl + '\n'
    solution.append(nl)
  solution = '\n'.join(solution)
  return proof_steps, solution, answer


def generate_caption(DEFINITIONS, p, formal=True):
  texts = []
  for clause in p.clauses:
    for c in clause.constructions:
      cdef = DEFINITIONS[c.name]
      if len(cdef.construction.args) != len(c.args):
        c.args = clause.points + c.args
      
      mapping = dict(zip(cdef.construction.args, c.args))
      # c_args = [mapping[a].upper() for a in cdef.construction.args]
      # cons_txt = pt.construction2description(c.name, c_args)
      # deps_txts = []
      for d in cdef.deps.constructions:
        d_args = [mapping[a].upper() for a in d.args]
        txt = pt.pretty_nl(d.name, d_args, formal)
        if len(txt) > 0:
          texts.append(txt)

      for (_, bs) in cdef.basics:
        for b in bs:
          b_args = [mapping[a].upper() for a in b.args]
          txt = pt.pretty_nl(b.name, b_args, formal)
          if len(txt) > 0:
            texts.append(txt)

      # if len(deps_txts) > 0:
      #   if len(cons_txt) > 0:
      #     cons_txt += " where " + " , ".join(deps_txts)
      #   else:
      #     cons_txt = ", ".join(deps_txts)
      # if len(cons_txt) > 0:
      #   texts.append(cons_txt)

  question = " . ".join(texts)
  return question


def draw_image_and_boxes(g, deps, imsize=512, file_name="test.png", fontsize=15, is_dot=True, font='sans-serif', fill_color=None, lw=1.2):
  highlights = []
  cnt = 0
  for i, dep in enumerate(deps):
    if dep.name == 'aconst':
      cnt += 1
    else:
      cnt = 0

    if i > 0 and dep.name == 'aconst' and cnt % 2 == 0:
      continue
    if i > 0 and dep.name == 'rconst' and deps[i - 1].name == 'rconst':
      continue

    if dep.name in ['perp', 'coll', 'aconst', 'rconst', 'cyclic']:
      highlights.append((dep.name, dep.args))

  bboxes = gh.nm.draw(
      g.type2nodes[gh.Point],
      g.type2nodes[gh.Line],
      g.type2nodes[gh.Circle],
      g.type2nodes[gh.Segment],
      theme='light',
      highlights=highlights,
      figname=file_name,
      image_size=imsize,
      fontsize=fontsize,
      is_dot=is_dot,
      font=font,
      fill_color=fill_color,
      lw=lw
  )

  return bboxes


def prepare_defs_and_rules(defs_file='defs.txt', rules_file='rules.txt'):
  DEFINITIONS = pr.Definition.from_txt_file(defs_file, to_dict=True)
  RULES = pr.Theorem.from_txt_file(rules_file, to_dict=True)
  object_defs = [
    "eq_trapezoid",
    "iso_triangle",
    "r_trapezoid",
    "r_triangle",
    "rectangle",
    "risos",
    "isquare",
    "ieq_triangle",
    "s_angle",
    "s_angle",
    "s_angle",
    "s_angle",
  ]
  removed_defs = [
    "e5128",
    "cc_tangent0",
    "free",
    "triangle12",
    "trisegment",
    "tangent",
    "eqangle2"
  ]

  filtered_defs = {0: [], 1: [], 2: [], 3: [], 4: []}
  for d in DEFINITIONS.values():
    n_vars = len(d.rely)
    if d.construction.name not in object_defs + removed_defs:
      filtered_defs[n_vars].append(d)
  filtered_defs[1].append(DEFINITIONS['s_angle'])

  return DEFINITIONS, RULES, object_defs, filtered_defs
