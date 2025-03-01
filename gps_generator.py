import json
import random
from tqdm import tqdm
from pathlib import Path
from absl import app, flags, logging
from multiprocessing import Process, Queue

import ddar
import graph as gh
import pretty as pt
import problem as pr
import geometry as gm
import generation_lib as glib


_OUT_FILE = flags.DEFINE_string(
  'out_file',
  'geoground.json',
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
  'the number of workers.'
)
_N_PROBLEMS = flags.DEFINE_integer(
  'n_problems',
  1,
  'the number of problems going to be generated.'
)


def pretty_angle(a: str, b: str, c: str, d: str) -> str:
  if b in (c, d):
    a, b = b, a
  if a == d:
    c, d = d, c

  if a == c:
    return f'\u2220{b}{a}{d}'
  return f'formed by {a}{b} and {c}{d}'


def pretty_nl(name: str, args: list[str]) -> str:
  """Natural lang formatting a predicate."""
  if name == 'aconst':
    a, b, c, d, y = args
    num, dem = y.split('PI/')
    deg = int(float(num) * 180 / float(dem))
    return f"the degree of {pretty_angle(a, b, c, d)} is {deg}"
    # return f'{pretty_angle(a, b, c, d)} = {y}'
  if name == 'rconst':
    a, b, c, d, x, y = args
    return f"the length of segment {a}{b} is {x} and the length of segment {c}{d} is {y}"
    # return f'{a}{b}:{c}{d} = {y}'
  if name == 'acompute':
    a, b, c, d = args
    return f"Compute the degree of angle between {a}{b} and {c}{d}."
    # return f'{pretty_angle(a, b, c, d)}'
  if name in ['coll', 'C']:
    return '' + ','.join(args) + ' are collinear'
  if name in ['ncoll']:
    return '' + ','.join(args) + ' are not collinear'
  if name == 'collx':
    return '' + ','.join(list(set(args))) + ' are collinear'
  if name in ['cyclic', 'O']:
    return '' + ','.join(args) + ' are concyclic'
  if name in ['midp', 'midpoint', 'M']:
    x, a, b = args
    return f'point {x} is midpoint of segment {a}{b}'
  if name in ['eqangle', 'eqangle6', '^']:
    a, b, c, d, e, f, g, h = args
    return f'{pretty_angle(a, b, c, d)} = {pretty_angle(e, f, g, h)}'
  if name in ['eqratio', 'eqratio6', '/']:
    return '{}{}:{}{} = {}{}:{}{}'.format(*args)
  if name == 'eqratio3':
    a, b, c, d, o, o = args  # pylint: disable=redeclared-assigned-name
    return f'S {o} {a} {b} {o} {c} {d}'
  if name in ['cong', 'D']:
    a, b, c, d = args
    return f'{a}{b} = {c}{d}'
  if name in ['perp', 'T']:
    if len(args) == 2:  # this is algebraic derivation.
      ab, cd = args  # ab = 'd( ... )'
      return f'{ab} \u27c2 {cd}'
    a, b, c, d = args
    return f'{a}{b} \u27c2 {c}{d}'
  if name in ['nperp']:
    if len(args) == 2:  # this is algebraic derivation.
      ab, cd = args  # ab = 'd( ... )'
      return f'the two lines {ab} and {cd} are not perpendicular'
    a, b, c, d = args
    return f'{a}{b} \u27c2 {c}{d}'
  if name in ['para', 'P']:
    if len(args) == 2:  # this is algebraic derivation.
      ab, cd = args  # ab = 'd( ... )'
      return f'{ab} \u2225 {cd}'
    a, b, c, d = args
    return f'{a}{b} \u2225 {c}{d}'
  if name in ['simtri2', 'simtri', 'simtri*']:
    a, b, c, x, y, z = args
    return f'\u0394{a}{b}{c} is similar to \u0394{x}{y}{z}'
  if name in ['contri2', 'contri', 'contri*']:
    a, b, c, x, y, z = args
    return f'\u0394{a}{b}{c} is congruent to \u0394{x}{y}{z}'
  if name in ['circle', 'I']:
    o, a, b, c = args
    return f'{o} is the circumcenter of \\Delta {a}{b}{c}'
  if name == 'foot':
    a, b, c, d = args
    return f'{a} is the foot of {b} on {c}{d}'
  return ""


def work(start, end, DEFINITIONS, RULES, object_defs, filtered_defs, data, n_clauses=1, imsize=512):
  logging.set_verbosity(logging.FATAL)

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

  image_folder = Path(_IMAGE_FOLDER.value)
  for img_id in tqdm(range(start, end)):
    while True:
      try:
        g, deps, p, problem_txt = glib.generate_problem(DEFINITIONS, object_defs, filtered_defs, n_clauses, GOALS)

        file_name = f"{img_id}.png"
        imsize = 300
        fontsize = 10
        _ = glib.draw_image_and_boxes(g, deps, imsize, image_folder / file_name, fontsize=fontsize)

        _, solution, answer = glib.generate_solution(g, RULES, p)
        question_nl = glib.generate_caption(DEFINITIONS, p, formal=False) + " . " + pt.goal_nl(p.goal.name, p.goal.args)
        question_fl = glib.generate_caption(DEFINITIONS, p, formal=True) + " . " + pt.goal_nl(p.goal.name, p.goal.args)
        difficulty = len(solution.split('\n'))

        data.put({
          'idx': img_id,
          'image': file_name,
          'question_nl': question_nl,
          'question_fl': question_fl,
          'solution': solution,
          'answer': answer,
          'difficulty': difficulty,
          'problem_txt': problem_txt
        })
        break
      except Exception as e:
        continue


def main(_):
  global DEFINITONS
  global RULES

  DEFINITIONS, RULES, object_defs, filtered_defs = glib.prepare_defs_and_rules()
  n_problems = _N_PROBLEMS.value
  n_workers = _N_WORKERS.value
  data = Queue()

  image_folder = Path(_IMAGE_FOLDER.value) 
  problem_file = Path(_OUT_FILE.value)
  image_folder.mkdir(parents=True, exist_ok=True)
  problem_file.parent.mkdir(parents=True, exist_ok=True)

  threads = []
  for i in range(n_workers - 1):
    th = Process(
      target=work, 
      args=(
        n_problems // n_workers * i, 
        n_problems // n_workers * (i + 1), 
        DEFINITIONS, RULES, object_defs, filtered_defs, data
      )
    )
    threads.append(th)
 
  th = Process(
    target=work, 
    args=(
      n_problems // n_workers * (n_workers - 1), 
      n_problems, 
      DEFINITIONS, RULES, object_defs, filtered_defs, data
    )
  )
  threads.append(th)

  for th in threads:
    th.start()

  with open(_OUT_FILE.value, "w") as f:
    cnt = 0
    ps = []
    while cnt < n_problems:
      p = data.get()
      ps.append(p)
      cnt += 1

    json.dump(ps, f)

if __name__ == "__main__":
  app.run(main)
