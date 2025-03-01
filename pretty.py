# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random


"""Utilities for string manipulation in the DSL."""

MAP_SYMBOL = {
    'T': 'perp',
    'P': 'para',
    'D': 'cong',
    'S': 'simtri',
    'I': 'circle',
    'M': 'midp',
    'O': 'cyclic',
    'C': 'coll',
    '^': 'eqangle',
    '/': 'eqratio',
    '%': 'eqratio',
    '=': 'contri',
    'X': 'collx',
    'A': 'acompute',
    'R': 'rcompute',
    'Q': 'fixc',
    'E': 'fixl',
    'V': 'fixb',
    'H': 'fixt',
    'Z': 'fixp',
    'Y': 'ind',
}


def map_symbol(c: str) -> str:
  return MAP_SYMBOL[c]


def map_symbol_inv(c: str) -> str:
  return {v: k for k, v in MAP_SYMBOL.items()}[c]


def _gcd(x: int, y: int) -> int:
  while y:
    x, y = y, x % y
  return x


def simplify(n: int, d: int) -> tuple[int, int]:
  g = _gcd(n, d)
  return (n // g, d // g)


def pretty2r(a: str, b: str, c: str, d: str) -> str:
  if b in (c, d):
    a, b = b, a

  if a == d:
    c, d = d, c

  return f'{a} {b} {c} {d}'


def pretty2a(a: str, b: str, c: str, d: str) -> str:
  if b in (c, d):
    a, b = b, a

  if a == d:
    c, d = d, c

  return f'{a} {b} {c} {d}'


def pretty_angle(a: str, b: str, c: str, d: str) -> str:
  if b in (c, d):
    a, b = b, a
  if a == d:
    c, d = d, c

  if a == c:
    rnd = random.randint(0, 4)
    if rnd == 0:
      return f'\u2220{a}'
    else:
      return f'\u2220{b}{a}{d}'
  return f'\u2220({a}{b}-{c}{d})'


def pretty_nl(name: str, args: list[str], formal=False) -> str:
  """Natural lang formatting a predicate."""
  if name == 'aconst':
    a, b, c, d, y = args
    # num, dem = y.split('_PI/')
    num, dem = y.split('PI/')
    deg = int(float(num) * 180 / float(dem))
    if formal:
      return f"{pretty_angle(a, b, c, d)} = {deg}°"
    else:
      return f"the degree of {pretty_angle(a, b, c, d)} is {deg}°"
    # return f'{pretty_angle(a, b, c, d)} = {y}'
  if name == 'rconst':
    a, b, c, d, l = args
    x, y = l.split('/')
    if formal:
      return f"{a}{b} = {x} . {c}{d} = {y}"
    else:
      return f"the length of segment {a}{b} is {x} . the length of segment {c}{d} is {y}"
  # if name == 'rconst':
  #   a, b, c, d, x, y = args
  #   if formal:
  #     return f'{a}{b}:{c}{d} = {y}'
  #   else:
  #     return f"the length of line {a}{b} is {x} and the length of line {c}{d} is {y}"
  if name in ['coll', 'C']:
    args = [f'point {name}' for name in args]
    return '' + ' , '.join(args) + ' are collinear'
  if name in ['ncoll']:
    args = [f'point {name}' for name in args]
    return '' + ' , '.join(args) + ' are not collinear'
  if name == 'collx':
    args = list(set(args))
    args = [f'point {name}' for name in args]
    return '' + ' , '.join(args) + ' are collinear'
  if name in ['cyclic', 'O']:
    args = [f'point {name}' for name in args]
    return '' + ' , '.join(args) + ' are concyclic'
  if name in ['midp', 'midpoint', 'M']:
    x, a, b = args
    if formal:
      return f'{a}{x} = {b}{x}'
    else:
      return f'point {x} is the midpoint of line {a}{b}'
  if name in ['eqangle', 'eqangle6', '^']:
    a, b, c, d, e, f, g, h = args
    if formal:
      return f'{pretty_angle(a, b, c, d)} = {pretty_angle(e, f, g, h)}'
    else:
      return f"the degrees of {pretty_angle(a, b, c, d)} and {pretty_angle(e, f, g, h)} are the same"
  if name in ['eqratio', 'eqratio6', '/']:
    return '{}{}:{}{} = {}{}:{}{}'.format(*args)
  # if name == 'eqratio3':
  #   a, b, c, d, o, o = args  # pylint: disable=redeclared-assigned-name
  #   return f'S {o} {a} {b} {o} {c} {d}'
  if name in ['cong', 'D']:
    a, b, c, d = args
    if formal:
      return f"{a}{b} = {c}{d}"
    else:
      return f'the lengths of line {a}{b} and line {c}{d} are the same'
  if name in ['perp', 'T']:
    if len(args) == 2:  # this is algebraic derivation.
      ab, cd = args  # ab = 'd( ... )'
      if formal:
        return f"{ab} \u27c2 {cd}"
      else:
        return f'line {ab} and line {cd} are perpendicular'
    a, b, c, d = args
    if formal:
      return f"{a}{b} \u27c2 {c}{d}"
    else:
      return f'line {a}{b} and line {c}{d} are perpendicular'
  # if name in ['nperp']:
  #   if len(args) == 2:  # this is algebraic derivation.
  #     ab, cd = args  # ab = 'd( ... )'
  #     return f'the two lines {ab} and {cd} are not perpendicular'
  #   a, b, c, d = args
  #   return f'The two lines {a}{b} and {c}{d} are note perpendicular'
  if name in ['para', 'P']:
    if len(args) == 2:  # this is algebraic derivation.
      ab, cd = args  # ab = 'd( ... )'   
      if formal:
        return f"{ab} \u2225 {cd}"
      else:
        return f'line {ab} and line {cd} are parallel'
    a, b, c, d = args
    if formal:
      return f"{a}{b} \u2225 {c}{d}"
    else:
      return f'line {a}{b} and line {c}{d} are parallel'
  if name in ['simtri2', 'simtri', 'simtri*']:
    a, b, c, x, y, z = args
    if formal:
      return f'\u0394{a}{b}{c} is similar to \u0394{x}{y}{z}'
    else:
      return f'triangle {a}{b}{c} is similar to triangle {x}{y}{z}'
  if name in ['contri2', 'contri', 'contri*']:
    a, b, c, x, y, z = args
    if formal:
      return f'\u0394{a}{b}{c} is congruent to \u0394{x}{y}{z}'
    else:
      return f'triangle {a}{b}{c} is congruent to triangle {x}{y}{z}'
  if name in ['circle', 'I']:
    o, a, b, c = args
    return f'point {o} is the circumcenter of triangle {a}{b}{c}'
  if name == 'foot':
    a, b, c, d = args
    return f'point {a} is the foot of point {b} on line {c}{d}'
  return ""


def goal_nl(name: str, args: list[str]) -> str:
  """Natural lang formatting a predicate."""
  # if name == 'rconst':
  #   a, b, c, d, x, y = args
  #   return f"the length of segment {a}{b} is {x} and the length of segment {c}{d} is {y}"
  #   # return f'{a}{b}:{c}{d} = {y}'
  if name in ['acompute', 'aconst']:
    if name == 'acompute':
      a, b, c, d = args
    else:
      a, b, c, d, _ = args

    if a == c or a == d:
      return f"Compute the degree of angle \u2220{b.upper()}{a.upper()}{(c if a != c else d).upper()}"
    elif b == c or b == d:
      return f"Compute the degree of angle \u2220{a.upper()}{b.upper()}{(c if b != c else d).upper()}"
    else:
      return f"Compute the degree of angle formed by {a.upper()}{b.upper()} and {c.upper()}{d.upper()}"
  elif name == 'acompute2':
    a, b, c = args
    return f"Compute the degree of angle \u2220{b.upper()}{a.upper()}{c.upper()}"
  elif name in ['coll', 'C']:
    return f"Show that points " + ", ".join(args) + " are collinear"
  elif name in ['eqangle', 'eqangle6', '^']:
    a, b, c, d, e, f, g, h = args
    return f'Show that {pretty_angle(a, b, c, d)} = {pretty_angle(e, f, g, h)}'
  elif name in ['eqratio', 'eqratio6', '/']:
    return 'Show that {}{}:{}{} = {}{}:{}{}'.format(*args)
  elif name in ['cong', 'D']:
    a, b, c, d = args
    return f'Show that {a}{b} = {c}{d}'
  elif name in ['para', 'P']:
    a, b, c, d = args
    return f'Show that {a}{b} and {c}{d} are parallel'
  elif name in ['simtri2', 'simtri', 'simtri*']:
    a, b, c, x, y, z = args
    return f'Show that \u0394{a}{b}{c} is similar to \u0394{x}{y}{z}'
  elif name in ['contri2', 'contri', 'contri*']:
    a, b, c, x, y, z = args
    return f'Show that \u0394{a}{b}{c} is congruent to \u0394{x}{y}{z}'
  elif name in ['circle', 'I']:
    o, a, b, c = args
    return f'Show that {o} is the circumcenter of \\Delta {a}{b}{c}'
  else:
    return ""


def pretty(txt: str) -> str:
  """Pretty formating a predicate string."""
  if isinstance(txt, str):
    txt = txt.split(' ')
  name, *args = txt
  if name == 'ind':
    return 'Y ' + ' '.join(args)
  if name in ['fixc', 'fixl', 'fixb', 'fixt', 'fixp']:
    return map_symbol_inv(name) + ' ' + ' '.join(args)
  if name == 'acompute':
    a, b, c, d = args
    return 'A ' + ' '.join(args)
  if name == 'rcompute':
    a, b, c, d = args
    return 'R ' + ' '.join(args)
  if name == 'aconst':
    a, b, c, d, y = args
    return f'^ {pretty2a(a, b, c, d)} {y}'
  if name == 'rconst':
    a, b, c, d, y = args
    return f'/ {pretty2r(a, b, c, d)} {y}'
  if name == 'coll':
    return 'C ' + ' '.join(args)
  if name == 'collx':
    return 'X ' + ' '.join(args)
  if name == 'cyclic':
    return 'O ' + ' '.join(args)
  if name in ['midp', 'midpoint']:
    x, a, b = args
    return f'M {x} {a} {b}'
  if name == 'eqangle':
    a, b, c, d, e, f, g, h = args
    return f'^ {pretty2a(a, b, c, d)} {pretty2a(e, f, g, h)}'
  if name == 'eqratio':
    a, b, c, d, e, f, g, h = args
    return f'/ {pretty2r(a, b, c, d)} {pretty2r(e, f, g, h)}'
  if name == 'eqratio3':
    a, b, c, d, o, o = args  # pylint: disable=redeclared-assigned-name
    return f'S {o} {a} {b} {o} {c} {d}'
  if name == 'cong':
    a, b, c, d = args
    return f'D {a} {b} {c} {d}'
  if name == 'perp':
    if len(args) == 2:  # this is algebraic derivation.
      ab, cd = args  # ab = 'd( ... )'
      return f'T {ab} {cd}'
    a, b, c, d = args
    return f'T {a} {b} {c} {d}'
  if name == 'para':
    if len(args) == 2:  # this is algebraic derivation.
      ab, cd = args  # ab = 'd( ... )'
      return f'P {ab} {cd}'
    a, b, c, d = args
    return f'P {a} {b} {c} {d}'
  if name in ['simtri2', 'simtri', 'simtri*']:
    a, b, c, x, y, z = args
    return f'S {a} {b} {c} {x} {y} {z}'
  if name in ['contri2', 'contri', 'contri*']:
    a, b, c, x, y, z = args
    return f'= {a} {b} {c} {x} {y} {z}'
  if name == 'circle':
    o, a, b, c = args
    return f'I {o} {a} {b} {c}'
  if name == 'foot':
    a, b, c, d = args
    return f'F {a} {b} {c} {d}'
  return ' '.join(txt)


def construction2description(name, args):
  if name == "angle_bisector":
    x, a, b, c = args
    return f"point {x} is on the bisector of angle {a}{b}{c}"
  elif name == "circle":
    x, a, b, c = args
    return f"point {x} is the center of the circle which passes through point {a} , point {b} , and point {c}"
  elif name == "circumcenter":
    x, a, b, c = args
    return f"point {x} is the center of the circumcircle of triangle {a}{b}{c}"
  elif name == "eq_trapezoid":
    a, b, c, d = args
    return f"trapezoid {a}{b}{c}{d}"
  elif name == "eq_triangle":
    x, b, c = args
    return f"triangle {x}{b}{c} is an equilateral triangle"
  elif name == "foot":
    x, a, b, c = args
    return f"point {x} is the foot of point {a} on line {b}{c}"
  elif name == "icenter":
    x, a, b, c = args
    return f"point {x} is the incenter of triangle {a}{b}{c}"
  elif name == "excenter":
    x, a, b, c = args
    return f"point {x} is the excenter of triangle {a}{b}{c}"
  elif name == "centroid":
    x, y, z, i, a, b, c = args
    return f"point {i} is the centroid of triangle {a}{b}{c}"
  elif name == "ninepoints":
    x, y, z, i, a, b, c = args
    return f"point {i} is the ninepoint of triangle {a}{b}{c}"
  elif name == "iso_triangle":
    a, b, c = args
    return f"triangle {a}{b}{c} is an isocele"
  elif name == "midpoint":
    x, a, b = args
    return f"point {x} is the midpoint of line {a}{b}"
  elif name == "on_circle":
    x, o, a = args
    return f"point {o} is the center of circle which passes through point {x} and point {a}"
  elif name == "orthocenter":
    x, a, b, c = args
    return f"point {x} is the center of the orthocenter of triangle {a}{b}{c}"
  elif name == "parallelogram":
    x, a, b, c = args
    return f"parallelogram {x}{a}{b}{c}"
  elif name == "r_trapezoid":
    a, b, c, d = args
    return f"right trapezoid {a}{b}{c}{d}"
  elif name == "rectangle":
    a, b, c, d = args
    return f"rectangle {a}{b}{c}{d}"
  elif name == "risos":
    a, b, c = args
    return f"triangle {a}{b}{c} is a right triangle and an isocele"
  elif name == "s_angle":
    a, b, x, y = args
    return f"the degree of angle {x}{b}{a} is {y}"
  elif name == "square":
    x, y, a, b = args
    return f"square {x}{y}{a}{b}"
  elif name == "isquare":
    a, b, c, d = args
    return f"square {a}{b}{c}{d}"
  elif name == "triangle12":
    a, b, c, x, y = args
    return f"triangle {a}{b}{c}"
  elif name == "trisect":
    x, y, a, b, c = args
    return f"line {x}{b} and line {y}{b} trisect angle {a}{b}{c}"
  elif name == "trisegment":
    x, y, a, b = args
    return f"point {x} and point {y} trisegment line {a}{b}"
  elif name == "on_dia":
    x, a, b = args
    return f"triangle {x}{a}{b} is a right triangle."
  elif name == "ieq_triangle":
    a, b, c = args
    return f"triangle {a}{b}{c} is an equilateral triangle"
  else:
    return ""


def question_nl(name: str, args: list[str]) -> str:
  """Natural lang formatting a predicate."""
  if name == 'aconst':
    a, b, c, d, y = args
    num, dem = y.split('PI/')
    deg = int(float(num) * 180 / float(dem))
    return f"Is the degree of angle {pretty_angle(a, b, c, d)} {deg}?"
  if name in ['eqangle', 'eqangle6', '^']:
    a, b, c, d, e, f, g, h = args
    return f'Is {pretty_angle(a, b, c, d)} = {pretty_angle(e, f, g, h)}?'
  if name in ['eqratio', 'eqratio6', '/']:
    return 'Is {}{}:{}{} = {}{}:{}{}?'.format(*args)
  if name in ['cong', 'D']:
    a, b, c, d = args
    return f'Is {a}{b} = {c}{d}?'
  if name in ['coll', 'C']:
    return 'Are the points ' + ', '.join(args) + ' collinear?'
  if name in ['cyclic', 'O']:
    return 'Are the points ' + ', '.join(args) + ' concyclic?'
  if name in ['perp', 'T']:
    if len(args) == 2:  # this is algebraic derivation.
      ab, cd = args  # ab = 'd( ... )'
      return f'Is {ab} \u27c2 {cd}?'
    a, b, c, d = args
    return f'Is {a}{b} \u27c2 {c}{d}?'
  if name in ['para', 'P']:
    a, b, c, d = args
    return f'Are {a}{b} and {c}{d} parallel?'
  if name in ['simtri2', 'simtri', 'simtri*']:
    a, b, c, x, y, z = args
    return f'Are the two triangles \u0394{a}{b}{c} and \u0394{x}{y}{z} similar?'
  if name in ['contri2', 'contri', 'contri*']:
    a, b, c, x, y, z = args
    return f'Are the two triangles \u0394{a}{b}{c} and \u0394{x}{y}{z} congruent?'
  if name in ['circle', 'I']:
    o, a, b, c = args
    return f'Is point {o} the circumcenter of \\Delta {a}{b}{c}?'
  return ""
