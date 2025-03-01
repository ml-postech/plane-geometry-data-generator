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
from generation_lib import generate_vars, prepare_defs_and_rules


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
  'the number of workers.'
)
_N_PROBLEMS = flags.DEFINE_integer(
  'n_problems',
  1,
  'the number of problems going to be generated.'
)
_IMAGE_SIZE = flags.DEFINE_integer(
  'image_size',
  200,
  'image size.'
)
_SPLIT = flags.DEFINE_string(
    'split',
    'train',
    '(train, test, valid)'
)


def work(start, end, DEFINITIONS, RULES, object_defs, filtered_defs, data, imsize=300):
  logging.set_verbosity(logging.FATAL)

  split = _SPLIT.value
  image_folder = Path(_IMAGE_FOLDER.value) / split
  for img_id in tqdm(range(start, end)):
    object_def_idx = random.sample([2, 3, 4, 5], k=1)[0]
    while True:
      try:
        object_def = DEFINITIONS[random.sample(object_defs[object_def_idx], k=1)[0]]

        if object_def.construction.name == 'triangle12':
          vars = generate_vars(0, 3)
          r1 = random.randint(1, 20)
          r2 = random.randint(r1, 20)
          clauses = [f"{vars[0]} {vars[1]} {vars[2]} = triangle12 {vars[0]} {vars[1]} {vars[2]} {r1} {r2}"]
        else:
          n_vars = len(object_def.construction.args)
          vars = generate_vars(0, n_vars)
          clauses = [" ".join(vars) + " = " + object_def.construction.name]

        answer = object_def_idx - 2
        problem_txt = "; ".join(clauses)
        # Build problem
        p = pr.Problem.from_txt(problem_txt, translate=True, shuffle=True)
        g, deps = gh.Graph.build_problem(p, DEFINITIONS, verbose=False)

        file_name = image_folder / f"{img_id}.png"
        highlights = []
        for i, dep in enumerate(deps):
          if i > 0 and dep.name == 'aconst' and deps[i - 1].name == 'aconst':
            continue
          highlights.append((dep.name, dep.args))

        gh.nm.draw(
            g.type2nodes[gh.Point],
            g.type2nodes[gh.Line],
            g.type2nodes[gh.Circle],
            g.type2nodes[gh.Segment],
            highlights=highlights,
            theme='light',
            figname=file_name,
            draw_all_lines=True
        )

        d = {
          "idx": img_id,
          "image": f"{split}/{img_id}.png",
          "answer": answer
        }
        data.put(d)

        break
      except Exception as e:
        continue


def main(_):
  global DEFINITONS
  global RULES

  DEFINITIONS, RULES, object_defs, filtered_defs = prepare_defs_and_rules()
  object_defs = {
    2: ['segment'],
    3: ['iso_triangle', 'r_triangle', 'risos', 'triangle', 'ieq_triangle'],
    4: ['eq_quadrangle', 'eq_trapezoid', 'quadrangle', 'r_trapezoid', 'rectangle', 'isquare', 'trapezoid'],
    5: ['pentagon']
  }
  n_problems = _N_PROBLEMS.value
  n_workers = _N_WORKERS.value
  image_size = _IMAGE_SIZE.value
  data = Queue()
  # work(0, n_problems, DEFINITIONS, RULES, object_defs, filtered_defs, data, 1, 200)

  split = _SPLIT.value
  image_folder = Path(_IMAGE_FOLDER.value) / split
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
        DEFINITIONS, RULES, object_defs, filtered_defs, data, image_size
      )
    )
    threads.append(th)
 
  th = Process(
    target=work, 
    args=(
      n_problems // n_workers * (n_workers - 1), 
      n_problems, 
      DEFINITIONS, RULES, object_defs, filtered_defs, data, image_size
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
