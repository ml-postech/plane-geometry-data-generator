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
  'clip.json',
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


def work(start, end, DEFINITIONS, RULES, object_defs, filtered_defs, data, n_clauses=2, imsize=512):
  logging.set_verbosity(logging.FATAL)

  image_folder = Path(_IMAGE_FOLDER.value)
  for img_id in tqdm(range(start, end)):
    while True:
      try:
        g, deps, p, problem_txt, _ = glib.generate_problem(DEFINITIONS, object_defs, filtered_defs, n_clauses)

        visual_deps = []
        for i, dep in enumerate(deps):
          if i > 0 and dep.name == 'aconst' and deps[i - 1].name == 'aconst':
            continue
          if i > 0 and dep.name == 'rconst' and deps[i - 1].name == 'rconst':
            continue

          if dep.name in ['perp', 'aconst', 'rconst', 'cyclic']:
            visual_deps.append(dep)

        if len(visual_deps) == 0:
            continue
        random.shuffle(visual_deps)

        caption_nl = " . ".join([pt.pretty_nl(dep.name, [a.name.upper() for a in dep.args]) for dep in visual_deps])
        caption_fl = " . ".join([pt.pretty_nl(dep.name, [a.name.upper() for a in dep.args], formal=True) for dep in visual_deps])
        caption_nv = " . ".join([pt.pretty_nl(dep.name, [a.name.upper() for a in dep.args], formal=True) for dep in deps])
        file_name = f"{img_id}.png"

        _ = glib.draw_image_and_boxes(g, deps, imsize, image_folder / file_name)  # , fontsize=fontsize, font=fontfamily, lw=lw)
        data.put({
          'idx': img_id,
          'image': file_name,
          'caption_natural': caption_nl,
          'caption_formal': caption_fl,
          'caption_non_visual': caption_nv,
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
