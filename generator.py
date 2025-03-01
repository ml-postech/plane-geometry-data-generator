import json
import random
from tqdm import tqdm
from pathlib import Path
from openai import Client
from absl import app, flags, logging
from multiprocessing import Process, Queue

import ddar
import graph as gh
import pretty as pt
import problem as pr
import geometry as gm


def generate_vars(cur_idx, n):
  ret = []
  for i in range(n):
    ret.append(f"x{cur_idx + i}")

  return ret


def natural_language_statement(logical_statement: pr.Dependency) -> str:
  """Convert logical_statement to natural language.

  Args:
    logical_statement: pr.Dependency with .name and .args

  Returns:
    a string of (pseudo) natural language of the predicate for human reader.
  """
  names = [a.name.upper() for a in logical_statement.args]
  names = [(n[0] + '_' + n[1:]) if len(n) > 1 else n for n in names]
  return pt.pretty_nl(logical_statement.name, names)


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


def write_solution(g: gh.Graph, p: pr.Problem) -> str:
  """
  Args:
    g: gh.Graph object, containing the proof state.
    p: pr.Problem object, containing the theorem.
  """
  _, _, proof_steps, refs = ddar.get_proof_steps(
      g, p.goal, merge_trivials=False
  )

  solution = ""
  for i, step in enumerate(proof_steps):
    _, [con] = step
    nl = proof_step_string(step, refs, last_step=i == len(proof_steps) - 1)
    solution += nl + '\n'

  return solution

def construction2description(name, args):
  if name == "angle_bisector":
    x, a, b, c = args
    return f"Point {x} is on the bisector of angle {a}{b}{c}"
  elif name == "circle":
    x, a, b, c = args
    return f"Point {x} is the center of the circle which passes through point {a}, {b}, and {c}"
  elif name == "circumcenter":
    x, a, b, c = args
    return f"Point {x} is the center of the circumcircle of triangle {a}{b}{c}"
  elif name == "eq_trapezoid":
    a, b, c, d = args
    return f"Trapezoid {a}{b}{c}{d}"
  elif name == "eq_triangle":
    x, b, c = args
    return f"Triangle {x}{b}{c} is an equilateral triangle"
  elif name == "foot":
    x, a, b, c = args
    return f"Point {x} is the foot of point {a} on segment {b}{c}"
  elif name == "icenter":
    x, a, b, c = args
    return f"Point {x} is the incenter of triangle {a}{b}{c}"
  elif name == "excenter":
    x, a, b, c = args
    return f"Point {x} is the excenter of triangle {a}{b}{c}"
  elif name == "centroid":
    x, y, z, i, a, b, c = args
    return f"Point {i} is the centroid of triangle {a}{b}{c}"
  elif name == "ninepoints":
    x, y, z, i, a, b, c = args
    return f"Point {i} is the ninepoint of triangle {a}{b}{c}"
  elif name == "iso_triangle":
    a, b, c = args
    return f"Triangle {a}{b}{c} is an isocele"
  elif name == "midpoint":
    x, a, b = args
    return f"Point {x} is the midpoint of segment {a}{b}"
  elif name == "on_circle":
    x, o, a = args
    return f"Point {o} is the center of circle which passes through point {x} and {a}"
  elif name == "orthocenter":
    x, a, b, c = args
    return f"Point {x} is the center of the orthocenter of triangle {a}{b}{c}"
  elif name == "parallelogram":
    x, a, b, c = args
    return f"Parallelogram {x}{a}{b}{c}"
  elif name == "r_trapezoid":
    a, b, c, d = args
    return f"Right trapezoid {a}{b}{c}{d}"
  elif name == "rectangle":
    a, b, c, d = args
    return f"Rectangle {a}{b}{c}{d}"
  elif name == "risos":
    a, b, c = args
    return f"Triangle {a}{b}{c} is a right triangle and an isocele"
  elif name == "s_angle":
    a, b, x, y = args
    return f"The degree of angle {x}{a}{b} is {y}"
  elif name == "square":
    x, y, a, b = args
    return f"Square {x}{y}{a}{b}"
  elif name == "isquare":
    a, b, c, d = args
    return f"Square {a}{b}{c}{d}"
  elif name == "triangle12":
    a, b, c, x, y = args
    return f"Triangle {a}{b}{c}"
  elif name == "trisect":
    x, y, a, b, c = args
    return f"Line {x}{b} and {y}{b} trisect angle {a}{b}{c}"
  elif name == "trisegment":
    x, y, a, b = args
    return f"Points {x} and {y} trisegment segment {a}{b}"
  elif name == "on_dia":
    x, a, b = args
    return f"Triangle {x}{a}{b} is a right triangle."
  elif name == "ieq_triangle":
    a, b, c = args
    return f"Triangle {a}{b}{c} is an equilateral triangle"
  else:
    return ""


def run_query(client, query):
  assistant_prompt = "You are an geometry problem solving expert."
  temperature=0.5
  max_tokens=256
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


_MODE = flags.DEFINE_string(
    'mode',
    'description',
    '(description, qa, solve)'
)
_DEFS_FILE = flags.DEFINE_string(
    'defs_file',
    'defs.txt',
    'definitions of available constructions to state a problem.'
)
_RULES_FILE = flags.DEFINE_string(
    'rules_file',
    'rules.txt',
    'list of deduction rules used by DO.'
)
_OUT_FILE = flags.DEFINE_string(
    'out_file',
    'problems.jsonl',
    'the name of file for the sampled problems.'
)
_IMAGE_FOLDER = flags.DEFINE_string(
    'image_folder',
    'images/',
    'the name of the folder which the images saved.'
)
_N_WORKERS = flags.DEFINE_integer(
    'n_workers',
    1,
    'the number of workers..'
)
_N_PROBLEMS = flags.DEFINE_integer(
    'n_problems',
    1,
    'the number of problems going to be generated.'
)
_N_CLAUSES = flags.DEFINE_integer(
    'n_clauses',
    1,
    'the number of clauses for each randomly generated problem.'
)
_N_CONSTRUCTIONS = flags.DEFINE_integer(
    'n_cons',
    1,
    'the number of constructions for each clause.'
)
_START_IDX = flags.DEFINE_integer(
    'start_idx',
    0,
    'staring index.'
)

GOALS = [
  # ['nperp', 4], ['nperp2', 3],
  # ['coll', 3], # ['ncoll', 3],
  # ['para', 4], # ['npara', 4],
  # ['cong', 4],
  # ['circle', 4],
  ['acompute', 4], ['acompute2', 3],
  # ['eqangle', 8], ['eqangle2', 6],
  # ['eqratio', 8],
  # ['simtri', 6], ['contri', 6],
  # ['sameside', 6], ['sameside2', 4]
]
DEFINITIONS = None
RULES = None

object_defs = [
  "eq_quadrangle",
  "eq_trapezoid",
  "iso_triangle",
  "pentagon",
  "quadrangle",
  "r_trapezoid",
  "r_triangle",
  "rectangle",
  "risos",
  "segment",
  "isquare",
  "trapezoid",
  "triangle",
  "s_angle",
  "ieq_triangle"
]

removed_defs = [
  "e5128",
  "cc_tangent0",
  "free",
  "triangle12"
]


def work(start, end, DEFINITIONS, filtered_defs, data):
  logging.set_verbosity(logging.FATAL)
  # Generate client for using OpenAI GPT
  with open('few_shot_samples.txt', 'r') as f:
    qa_pre_query = ''.join(f.readlines()).strip()
  desc_pre_query = 'Paraphrase the following geometric conditions into plain text: '
  solve_pre_query_q = 'Paraphrase the following geometry problem into plain text: '
  solve_pre_query_a = 'Paraphrase the following step-by-step solution of a geometry problem into plain text: '
  client = Client()
  mode = _MODE.value

  for idx in tqdm(range(start, end)):
  # for idx in range(start, end):
    while True:
      try:
        # Random sample a problem
        n_clauses = _N_CLAUSES.value

        ## Sample the first clause which generates an object
        object_def_idx = random.randrange(len(object_defs))
        object_def = DEFINITIONS[object_defs[object_def_idx]]

        if object_def.construction.name == 's_angle' or object_def.construction.name == 'segment':
          vars = generate_vars(0, 3)
          angle = random.randint(30, 150)
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

        ## Sample the other clauses to construct the full problem statement
        for _ in range(n_clauses):
          n_vars = random.randrange(2) + 1
          n_cons = 1 if n_vars > 1 else random.randrange(_N_CONSTRUCTIONS.value) + 1

          new_vars = generate_vars(len(vars), n_vars)
          selected_cons = []
          cnt = 0
          while cnt < n_cons:
            con = random.sample(filtered_defs[n_vars], k=1)[0]
            con_args = con.construction.args
            if len(con_args) - n_vars > len(vars):
              continue
            n_req_vars = len(con_args) if con.construction.name != 's_angle' else 3
            given_vars = random.sample(vars, k=n_req_vars - n_vars)
            given_vars.sort()
            if con.construction.name == 's_angle':
              deg = random.randint(30, 150)
              given_vars = given_vars + new_vars + [str(deg)]
            selected_cons.append(con.construction.name + f" " + " ".join(given_vars))
            cnt += 1

          if len(selected_cons) > 0:
            clause = " ".join(new_vars) + " = " + ", ".join(selected_cons)
            clauses.append(clause)
            vars = vars + new_vars

        problem_txt = "; ".join(clauses)

        ## Sample a goal for the problem statement
        if mode == 'solve':
          goal_idx = random.randint(0, len(GOALS) - 1)
          goal_name, goal_n_vars = GOALS[goal_idx]
          goal_vars = random.sample(vars, k=goal_n_vars)
          if goal_name == 'acompute2':
            goal_name = 'acompute'
            goal_vars.append(goal_vars[0])
          goal_txt = goal_name + " " + " ".join(goal_vars)
          problem_txt += " ? " + goal_txt

        # Build problem
        p = pr.Problem.from_txt(problem_txt)  
        g, deps = gh.Graph.build_problem(p, DEFINITIONS, verbose=False)
        org_g = g.copy()

        # Construct question text
        texts = []
        desc_texts = []
        for clause in p.clauses:
          for c in clause.constructions:
            cdef = DEFINITIONS[c.name]
            if len(cdef.construction.args) != len(c.args):
              c.args = clause.points + c.args
            
            mapping = dict(zip(cdef.construction.args, c.args))
            c_args = [mapping[a].upper() for a in cdef.construction.args]
            cons_txt = construction2description(c.name, c_args)
            deps_txts = []
            for d in cdef.deps.constructions:
              d_args = [mapping[a].upper() for a in d.args]
              txt = pt.pretty_nl(d.name, d_args)
              if len(txt) > 0:
                deps_txts.append(txt)
              if d.name in ['perp', 'circle', 'cyclic', 'coll', 'aconst', 'acompute']:
                desc_texts.append(txt)

            for (_, bs) in cdef.basics:
              for b in bs:
                b_args = [mapping[a].upper() for a in b.args]
                txt = pt.pretty_nl(b.name, b_args)
                if len(txt) > 0:
                  deps_txts.append(txt)
                if b.name in ['perp', 'circle', 'cyclic', 'coll', 'aconst', 'acompute']:
                  desc_texts.append(txt)

            if len(deps_txts) > 0:
              if len(cons_txt) > 0:
                cons_txt += " where " + ", ".join(deps_txts)
              else:
                cons_txt = ", ".join(deps_txts)
            if len(cons_txt) > 0:
              texts.append(cons_txt)

        question = ". ".join(texts)

        conversations = []
        if mode == 'description':
          description = ". ".join(desc_texts)
          description = run_query(client, desc_pre_query + description)
          conversations.append(
              {"from": "human", "value": "<image>\nExplain about the figure."}
          )
          conversations.append(
              {"from": "gpt", "value": description}
          )

        elif mode == 'qa':
          new_query = qa_pre_query + ' ' + question
          qa_pairs = run_query(client, new_query).split('\n\n')
          for i, qa in enumerate(qa_pairs):
            q, a = qa.split('\n')
            if len(q) == 0 or len(a) == 0:
              continue
            if i == 0:
              q = '<image>\n' + question + '\n' + q
            conversations.append(
                {"from": "human", "value": q.replace("Question: ", "")}
            )
            conversations.append(
                {"from": "gpt", "value": a.replace("Answer: ", "")}
            )
          if len(conversations) == 0:
            continue

        elif mode == 'solve':
          ddar.solve(g, RULES, p, max_level=5)
          goal_args = g.names2nodes(p.goal.args)
          if not g.check(p.goal.name, goal_args):
            continue

          if p.goal.name == 'acompute':
            a, b, c, d = goal_args
            ab = g._get_line(a, b)
            cd = g._get_line(c, d)
            for ang0 in g.aconst.values():
              for ang in ang0.val.neighbors(gm.Angle):
                d1, d2 = ang.directions
                if ab.val == d1 and cd.val == d2:
                  p.goal = pr.Construction.from_txt(f"aconst {a.name} {b.name} {c.name} {d.name} {ang0.name}")
                  break
          
          _, _, proof_steps, _ = ddar.get_proof_steps(
              g, p.goal, merge_trivials=False
          )
          if len(proof_steps) == 0:
            continue

          sub_goal_idx = random.randint(0, len(proof_steps) - 1)
          sub_goal = proof_steps[sub_goal_idx][1][0]
          names = [a.name for a in sub_goal.args]
          goal_txt = pt.goal_nl(sub_goal.name, names)
          if len(goal_txt) == 0:
            continue

          sub_goal = pr.Construction.from_txt(sub_goal.name + " " + " ".join(names))
          p.goal = sub_goal
          answer = write_solution(g, p)

          question = question + '. ' + goal_txt
          question = run_query(client, solve_pre_query_q + question)
          answer = run_query(client, solve_pre_query_a + answer)

          conversations.append(
              {"from": "human", "value": "<image>\n" + question}
          )
          conversations.append(
              {"from": "gpt", "value": answer}
          )

        # Figure generation
        file_name = Path(_IMAGE_FOLDER.value) / f"{idx}.png"
        highlights = []
        for i, dep in enumerate(deps):
          if i > 0 and dep.name == 'aconst' and deps[i - 1].name == 'aconst':
            continue
          highlights.append((dep.name, dep.args))
        gh.nm.draw(
            org_g.type2nodes[gh.Point],
            org_g.type2nodes[gh.Line],
            org_g.type2nodes[gh.Circle],
            org_g.type2nodes[gh.Segment],
            highlights=highlights,
            theme='light',
            figname=file_name
        )

        d = {
          "idx": idx,
          "image": str(file_name),
          "problem": problem_txt,
          "conversations": conversations
        }
        data.put(d)
        break
      except Exception as e:
        continue

def main(_):
  global DEFINITONS
  global RULES

  DEFINITIONS = pr.Definition.from_txt_file(_DEFS_FILE.value, to_dict=True)
  RULES = pr.Theorem.from_txt_file(_RULES_FILE.value, to_dict=True)

  filtered_defs = {0: [], 1: [], 2: [], 3: [], 4: []}
  for d in DEFINITIONS.values():
    n_vars = len(d.rely)
    if d.construction.name not in object_defs + removed_defs:
      filtered_defs[n_vars].append(d)
  filtered_defs[1].append(DEFINITIONS['s_angle'])

  n_problems = _N_PROBLEMS.value
  n_workers = _N_WORKERS.value
  data = Queue()
  # work(0, 1, DEFINITIONS, filtered_defs, data)
  threads = []
  start_idx = _START_IDX.value
  for i in range(n_workers - 1):
    th = Process(
      target=work, 
      args=(start_idx + n_problems // n_workers * i, start_idx + n_problems // n_workers * (i + 1), DEFINITIONS, filtered_defs, data)
    )
    threads.append(th)
 
  th = Process(
    target=work, 
    args=(start_idx + n_problems // n_workers * (n_workers - 1), start_idx + n_problems, DEFINITIONS, filtered_defs, data)
  )
  threads.append(th)

  for th in threads:
    th.start()

  cnt = 0
  ps = []
  with open(_OUT_FILE.value, "w") as f:
    while cnt < n_problems:
      p = data.get()
      ps.append(p)
      cnt += 1

    json.dump(ps, f)

if __name__ == "__main__":
    app.run(main)

