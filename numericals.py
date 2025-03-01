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

"""Numerical representation of geometry."""
from __future__ import annotations

import math
import random
from typing import Any, Optional, Union

import geometry as gm
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib import colors
import numpy as np
from numpy.random import uniform as unif  # pylint: disable=g-importing-member
from matplotlib import rcParams


# matplotlib.use('TkAgg')


ATOM = 1e-12


# Some variables are there for better code reading.
# pylint: disable=unused-assignment
# pylint: disable=unused-argument
# pylint: disable=unused-variable

# Naming in geometry is a little different
# we stick to geometry naming to better read the code.
# pylint: disable=invalid-name


class Point:
  """Numerical point."""

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __lt__(self, other: Point) -> bool:
    return (self.x, self.y) < (other.x, other.y)

  def __gt__(self, other: Point) -> bool:
    return (self.x, self.y) > (other.x, other.y)

  def __add__(self, p: Point) -> Point:
    return Point(self.x + p.x, self.y + p.y)

  def __sub__(self, p: Point) -> Point:
    return Point(self.x - p.x, self.y - p.y)

  def __mul__(self, f: float) -> Point:
    return Point(self.x * f, self.y * f)

  def __rmul__(self, f: float) -> Point:
    return self * f

  def __truediv__(self, f: float) -> Point:
    return Point(self.x / f, self.y / f)

  def __floordiv__(self, f: float) -> Point:
    div = self / f  # true div
    return Point(int(div.x), int(div.y))

  def __str__(self) -> str:
    return 'P({},{})'.format(self.x, self.y)

  def close(self, point: Point, tol: float = 1e-12) -> bool:
    return abs(self.x - point.x) < tol and abs(self.y - point.y) < tol

  def midpoint(self, p: Point) -> Point:
    return Point(0.5 * (self.x + p.x), 0.5 * (self.y + p.y))

  def distance(self, p: Union[Point, Line, Circle]) -> float:
    if isinstance(p, Line):
      return p.distance(self)
    if isinstance(p, Circle):
      return abs(p.radius - self.distance(p.center))
    dx = self.x - p.x
    dy = self.y - p.y
    return np.sqrt(dx * dx + dy * dy)

  def distance2(self, p: Point) -> float:
    if isinstance(p, Line):
      return p.distance(self)
    dx = self.x - p.x
    dy = self.y - p.y
    return dx * dx + dy * dy

  def rotatea(self, ang: float) -> Point:
    sinb, cosb = np.sin(ang), np.cos(ang)
    return self.rotate(sinb, cosb)

  def rotate(self, sinb: float, cosb: float) -> Point:
    x, y = self.x, self.y
    return Point(x * cosb - y * sinb, x * sinb + y * cosb)

  def flip(self) -> Point:
    return Point(-self.x, self.y)

  def perpendicular_line(self, line: Line) -> Line:
    return line.perpendicular_line(self)

  def foot(self, line: Line) -> Point:
    if isinstance(line, Line):
      l = line.perpendicular_line(self)
      return line_line_intersection(l, line)
    elif isinstance(line, Circle):
      c, r = line.center, line.radius
      return c + (self - c) * r / self.distance(c)
    raise ValueError('Dropping foot to weird type {}'.format(type(line)))

  def parallel_line(self, line: Line) -> Line:
    return line.parallel_line(self)

  def norm(self) -> float:
    return np.sqrt(self.x**2 + self.y**2)

  def cos(self, other: Point) -> float:
    x, y = self.x, self.y
    a, b = other.x, other.y
    return (x * a + y * b) / self.norm() / other.norm()

  def dot(self, other: Point) -> float:
    return self.x * other.x + self.y * other.y

  def sign(self, line: Line) -> int:
    return line.sign(self)

  def is_same(self, other: Point) -> bool:
    return self.distance(other) <= ATOM


class Line:
  """Numerical line."""

  def __init__(
      self,
      p1: Point = None,
      p2: Point = None,
      coefficients: tuple[int, int, int] = None,
  ):
    if p1 is None and p2 is None and coefficients is None:
      self.coefficients = None, None, None
      return

    a, b, c = coefficients or (
        p1.y - p2.y,
        p2.x - p1.x,
        p1.x * p2.y - p2.x * p1.y,
    )

    # Make sure a is always positive (or always negative for that matter)
    # With a == 0, Assuming a = +epsilon > 0
    # Then b such that ax + by = 0 with y>0 should be negative.
    if a < 0.0 or a == 0.0 and b > 0.0:
      a, b, c = -a, -b, -c

    self.coefficients = a, b, c

  def parallel_line(self, p: Point) -> Line:
    a, b, _ = self.coefficients
    return Line(coefficients=(a, b, -a * p.x - b * p.y))  # pylint: disable=invalid-unary-operand-type

  def perpendicular_line(self, p: Point) -> Line:
    a, b, _ = self.coefficients
    return Line(p, p + Point(a, b))

  def greater_than(self, other: Line) -> bool:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    # b/a > y/x
    return b * x > a * y

  def __gt__(self, other: Line) -> bool:
    return self.greater_than(other)

  def __lt__(self, other: Line) -> bool:
    return other.greater_than(self)

  def same(self, other: Line) -> bool:
    a, b, c = self.coefficients
    x, y, z = other.coefficients
    return close_enough(a * y, b * x) and close_enough(b * z, c * y)

  def equal(self, other: Line) -> bool:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    # b/a == y/x
    return b * x == a * y

  def less_than(self, other: Line) -> bool:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    # b/a > y/x
    return b * x < a * y

  def intersect(self, obj: Union[Line, Circle]) -> tuple[Point, ...]:
    if isinstance(obj, Line):
      return line_line_intersection(self, obj)
    if isinstance(obj, Circle):
      return line_circle_intersection(self, obj)

  def distance(self, p: Point) -> float:
    a, b, c = self.coefficients
    return abs(self(p.x, p.y)) / math.sqrt(a * a + b * b)

  def __call__(self, x: Point, y: Point = None) -> float:
    if isinstance(x, Point) and y is None:
      return self(x.x, x.y)
    a, b, c = self.coefficients
    return x * a + y * b + c

  def is_parallel(self, other: Line) -> bool:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    return abs(a * y - b * x) < ATOM

  def is_perp(self, other: Line) -> bool:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    return abs(a * x + b * y) < ATOM

  def cross(self, other: Line) -> float:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    return a * y - b * x

  def dot(self, other: Line) -> float:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    return a * x + b * y

  def point_at(self, x: float = None, y: float = None) -> Optional[Point]:
    """Get a point on line closest to (x, y)."""
    a, b, c = self.coefficients
    # ax + by + c = 0
    if x is None and y is not None:
      if a != 0:
        return Point((-c - b * y) / a, y)  # pylint: disable=invalid-unary-operand-type
      else:
        return None
    elif x is not None and y is None:
      if b != 0:
        return Point(x, (-c - a * x) / b)  # pylint: disable=invalid-unary-operand-type
      else:
        return None
    elif x is not None and y is not None:
      if a * x + b * y + c == 0.0:
        return Point(x, y)
    return None

  def diff_side(self, p1: Point, p2: Point) -> Optional[bool]:
    d1 = self(p1.x, p1.y)
    d2 = self(p2.x, p2.y)
    if d1 == 0 or d2 == 0:
      return None
    return d1 * d2 < 0

  def same_side(self, p1: Point, p2: Point) -> Optional[bool]:
    d1 = self(p1.x, p1.y)
    d2 = self(p2.x, p2.y)
    if d1 == 0 or d2 == 0:
      return None
    return d1 * d2 > 0

  def sign(self, point: Point) -> int:
    s = self(point.x, point.y)
    if s > 0:
      return 1
    elif s < 0:
      return -1
    return 0

  def is_same(self, other: Line) -> bool:
    a, b, c = self.coefficients
    x, y, z = other.coefficients
    return abs(a * y - b * x) <= ATOM and abs(b * z - c * y) <= ATOM

  def sample_within(self, points: list[Point], n: int = 5) -> list[Point]:
    """Sample a point within the boundary of points."""
    center = sum(points, Point(0.0, 0.0)) * (1.0 / len(points))
    radius = max([p.distance(center) for p in points])
    if close_enough(center.distance(self), radius):
      center = center.foot(self)
    a, b = line_circle_intersection(self, Circle(center.foot(self), radius))

    result = None
    best = -1.0
    for _ in range(n):
      rand = unif(0.0, 1.0)
      x = a + (b - a) * rand
      mind = min([x.distance(p) for p in points])
      if mind > best:
        best = mind
        result = x

    return [result]


class InvalidLineIntersectError(Exception):
  pass


class HalfLine(Line):
  """Numerical ray."""

  def __init__(self, tail: Point, head: Point):  # pylint: disable=super-init-not-called
    self.line = Line(tail, head)
    self.coefficients = self.line.coefficients
    self.tail = tail
    self.head = head

  def intersect(self, obj: Union[Line, HalfLine, Circle, HoleCircle]) -> Point:
    if isinstance(obj, (HalfLine, Line)):
      return line_line_intersection(self.line, obj)

    exclude = [self.tail]
    if isinstance(obj, HoleCircle):
      exclude += [obj.hole]

    a, b = line_circle_intersection(self.line, obj)
    if any([a.close(x) for x in exclude]):
      return b
    if any([b.close(x) for x in exclude]):
      return a

    v = self.head - self.tail
    va = a - self.tail
    vb = b - self.tail
    if v.dot(va) > 0:
      return a
    if v.dot(vb) > 0:
      return b
    raise InvalidLineIntersectError()

  def sample_within(self, points: list[Point], n: int = 5) -> list[Point]:
    center = sum(points, Point(0.0, 0.0)) * (1.0 / len(points))
    radius = max([p.distance(center) for p in points])
    if close_enough(center.distance(self.line), radius):
      center = center.foot(self)
    a, b = line_circle_intersection(self, Circle(center.foot(self), radius))

    if (a - self.tail).dot(self.head - self.tail) > 0:
      a, b = self.tail, a
    else:
      a, b = self.tail, b  # pylint: disable=self-assigning-variable

    result = None
    best = -1.0
    for _ in range(n):
      x = a + (b - a) * unif(0.0, 1.0)
      mind = min([x.distance(p) for p in points])
      if mind > best:
        best = mind
        result = x

    return [result]


def _perpendicular_bisector(p1: Point, p2: Point) -> Line:
  midpoint = (p1 + p2) * 0.5
  return Line(midpoint, midpoint + Point(p2.y - p1.y, p1.x - p2.x))


def same_sign(
    a: Point, b: Point, c: Point, d: Point, e: Point, f: Point
) -> bool:
  a, b, c, d, e, f = map(lambda p: p.sym, [a, b, c, d, e, f])
  ab, cb = a - b, c - b
  de, fe = d - e, f - e
  return (ab.x * cb.y - ab.y * cb.x) * (de.x * fe.y - de.y * fe.x) > 0


class Circle:
  """Numerical circle."""

  def __init__(
      self,
      center: Optional[Point] = None,
      radius: Optional[float] = None,
      p1: Optional[Point] = None,
      p2: Optional[Point] = None,
      p3: Optional[Point] = None,
  ):
    if not center:
      if not (p1 and p2 and p3):
        self.center = self.radius = self.r2 = None
        return
        # raise ValueError('Circle without center need p1 p2 p3')

      l12 = _perpendicular_bisector(p1, p2)
      l23 = _perpendicular_bisector(p2, p3)
      center = line_line_intersection(l12, l23)

    self.center = center
    self.a, self.b = center.x, center.y

    if not radius:
      if not (p1 or p2 or p3):
        raise ValueError('Circle needs radius or p1 or p2 or p3')
      p = p1 or p2 or p3
      self.r2 = (self.a - p.x) ** 2 + (self.b - p.y) ** 2
      self.radius = math.sqrt(self.r2)
    else:
      self.radius = radius
      self.r2 = radius * radius

  def intersect(self, obj: Union[Line, Circle]) -> tuple[Point, ...]:
    if isinstance(obj, Line):
      return obj.intersect(self)
    if isinstance(obj, Circle):
      return circle_circle_intersection(self, obj)

  def sample_within(self, points: list[Point], n: int = 5) -> list[Point]:
    """Sample a point within the boundary of points."""
    result = None
    best = -1.0
    for _ in range(n):
      ang = unif(0.0, 2.0) * np.pi
      x = self.center + Point(np.cos(ang), np.sin(ang)) * self.radius
      mind = min([x.distance(p) for p in points])
      if mind > best:
        best = mind
        result = x

    return [result]


class HoleCircle(Circle):
  """Numerical circle with a missing point."""

  def __init__(self, center: Point, radius: float, hole: Point):
    super().__init__(center, radius)
    self.hole = hole

  def intersect(self, obj: Union[Line, HalfLine, Circle, HoleCircle]) -> Point:
    if isinstance(obj, Line):
      a, b = line_circle_intersection(obj, self)
      if a.close(self.hole):
        return b
      return a
    if isinstance(obj, HalfLine):
      return obj.intersect(self)
    if isinstance(obj, Circle):
      a, b = circle_circle_intersection(obj, self)
      if a.close(self.hole):
        return b
      return a
    if isinstance(obj, HoleCircle):
      a, b = circle_circle_intersection(obj, self)
      if a.close(self.hole) or a.close(obj.hole):
        return b
      return a


def solve_quad(a: float, b: float, c: float) -> tuple[float, float]:
  """Solve a x^2 + bx + c = 0."""
  a = 2 * a
  d = b * b - 2 * a * c
  if d < 0:
    return None  # the caller should expect this result.

  y = math.sqrt(d)
  return (-b - y) / a, (-b + y) / a


def circle_circle_intersection(c1: Circle, c2: Circle) -> tuple[Point, Point]:
  """Returns a pair of Points as intersections of c1 and c2."""
  # circle 1: (x0, y0), radius r0
  # circle 2: (x1, y1), radius r1
  x0, y0, r0 = c1.a, c1.b, c1.radius
  x1, y1, r1 = c2.a, c2.b, c2.radius

  d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
  if d == 0:
    raise InvalidQuadSolveError()

  a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
  h = r0 ** 2 - a ** 2
  if h < 0:
    raise InvalidQuadSolveError()
  h = np.sqrt(h)
  x2 = x0 + a * (x1 - x0) / d
  y2 = y0 + a * (y1 - y0) / d
  x3 = x2 + h * (y1 - y0) / d
  y3 = y2 - h * (x1 - x0) / d
  x4 = x2 - h * (y1 - y0) / d
  y4 = y2 + h * (x1 - x0) / d

  return Point(x3, y3), Point(x4, y4)


class InvalidQuadSolveError(Exception):
  pass


def line_circle_intersection(line: Line, circle: Circle) -> tuple[Point, Point]:
  """Returns a pair of points as intersections of line and circle."""
  a, b, c = line.coefficients
  r = float(circle.radius)
  center = circle.center
  p, q = center.x, center.y

  if b == 0:
    x = -c / a
    x_p = x - p
    x_p2 = x_p * x_p
    y = solve_quad(1, -2 * q, q * q + x_p2 - r * r)
    if y is None:
      raise InvalidQuadSolveError()
    y1, y2 = y
    return (Point(x, y1), Point(x, y2))

  if a == 0:
    y = -c / b
    y_q = y - q
    y_q2 = y_q * y_q
    x = solve_quad(1, -2 * p, p * p + y_q2 - r * r)
    if x is None:
      raise InvalidQuadSolveError()
    x1, x2 = x
    return (Point(x1, y), Point(x2, y))

  c_ap = c + a * p
  a2 = a * a
  y = solve_quad(
      a2 + b * b, 2 * (b * c_ap - a2 * q), c_ap * c_ap + a2 * (q * q - r * r)
  )
  if y is None:
    raise InvalidQuadSolveError()
  y1, y2 = y

  return Point(-(b * y1 + c) / a, y1), Point(-(b * y2 + c) / a, y2)


def _check_between(a: Point, b: Point, c: Point) -> bool:
  """Whether a is between b & c."""
  return (a - b).dot(c - b) > 0 and (a - c).dot(b - c) > 0


def circle_segment_intersect(
    circle: Circle, p1: Point, p2: Point
) -> list[Point]:
  l = Line(p1, p2)
  px, py = line_circle_intersection(l, circle)

  result = []
  if _check_between(px, p1, p2):
    result.append(px)
  if _check_between(py, p1, p2):
    result.append(py)
  return result


def line_segment_intersection(l: Line, A: Point, B: Point) -> Point:  # pylint: disable=invalid-name
  a, b, c = l.coefficients
  x1, y1, x2, y2 = A.x, A.y, B.x, B.y
  dx, dy = x2 - x1, y2 - y1
  alpha = (-c - a * x1 - b * y1) / (a * dx + b * dy)
  return Point(x1 + alpha * dx, y1 + alpha * dy)


def line_line_intersection(l1: Line, l2: Line) -> Point:
  a1, b1, c1 = l1.coefficients
  a2, b2, c2 = l2.coefficients
  # a1x + b1y + c1 = 0
  # a2x + b2y + c2 = 0
  d = a1 * b2 - a2 * b1
  if d == 0:
    raise InvalidLineIntersectError
  return Point((c2 * b1 - c1 * b2) / d, (c1 * a2 - c2 * a1) / d)


def check_too_close(
    newpoints: list[Point], points: list[Point], tol: int = 0.1
) -> bool:
  if not points:
    return False
  avg = sum(points, Point(0.0, 0.0)) * 1.0 / len(points)
  mindist = min([p.distance(avg) for p in points])
  for p0 in newpoints:
    for p1 in points:
      if p0.distance(p1) < tol * mindist:
        return True
  return False


def check_too_far(
    newpoints: list[Point], points: list[Point], tol: int = 4
) -> bool:
  if len(points) < 2:
    return False
  avg = sum(points, Point(0.0, 0.0)) * 1.0 / len(points)
  maxdist = max([p.distance(avg) for p in points])
  for p in newpoints:
    if p.distance(avg) > maxdist * tol:
      return True
  return False


def check_aconst(args: list[Point]) -> bool:
  a, b, c, d, num, den = args
  d = d + a - c
  ang = ang_between(a, b, d)
  if ang < 0:
    ang += np.pi
  return close_enough(ang, num * np.pi / den)


def check(name: str, args: list[Union[gm.Point, Point]]) -> bool:
  """Numerical check."""
  if name == 'eqangle6':
    name = 'eqangle'
  elif name == 'eqratio6':
    name = 'eqratio'
  elif name in ['simtri2', 'simtri*']:
    name = 'simtri'
  elif name in ['contri2', 'contri*']:
    name = 'contri'
  elif name == 'para':
    name = 'para_or_coll'
  elif name == 'on_line':
    name = 'coll'
  elif name in ['rcompute', 'acompute']:
    return True
  elif name in ['fixl', 'fixc', 'fixb', 'fixt', 'fixp']:
    return True

  fn_name = 'check_' + name
  if fn_name not in globals():
    return None

  fun = globals()['check_' + name]
  args = [p.num if isinstance(p, gm.Point) else p for p in args]
  return fun(args)


def check_circle(points: list[Point]) -> bool:
  if len(points) != 4:
    return False
  o, a, b, c = points
  oa, ob, oc = o.distance(a), o.distance(b), o.distance(c)
  return close_enough(oa, ob) and close_enough(ob, oc)


def check_coll(points: list[Point]) -> bool:
  a, b = points[:2]
  l = Line(a, b)
  for p in points[2:]:
    if abs(l(p.x, p.y)) > ATOM:
      return False
  return True


def check_ncoll(points: list[Point]) -> bool:
  return not check_coll(points)


def check_sameside(points: list[Point]) -> bool:
  b, a, c, y, x, z = points
  # whether b is to the same side of a & c as y is to x & z
  ba = b - a
  bc = b - c
  yx = y - x
  yz = y - z
  return ba.dot(bc) * yx.dot(yz) > 0


def check_para_or_coll(points: list[Point]) -> bool:
  return check_para(points) or check_coll(points)


def check_para(points: list[Point]) -> bool:
  a, b, c, d = points
  ab = Line(a, b)
  cd = Line(c, d)
  if ab.same(cd):
    return False
  return ab.is_parallel(cd)


def check_perp(points: list[Point]) -> bool:
  a, b, c, d = points
  ab = Line(a, b)
  cd = Line(c, d)
  return ab.is_perp(cd)


def check_cyclic(points: list[Point]) -> bool:
  points = list(set(points))
  (a, b, c), *ps = points
  circle = Circle(p1=a, p2=b, p3=c)
  for d in ps:
    if not close_enough(d.distance(circle.center), circle.radius):
      return False
  return True


def bring_together(
    a: Point, b: Point, c: Point, d: Point
) -> tuple[Point, Point, Point, Point]:
  ab = Line(a, b)
  cd = Line(c, d)
  x = line_line_intersection(ab, cd)
  unit = Circle(center=x, radius=1.0)
  y, _ = line_circle_intersection(ab, unit)
  z, _ = line_circle_intersection(cd, unit)
  return x, y, x, z


def same_clock(
    a: Point, b: Point, c: Point, d: Point, e: Point, f: Point
) -> bool:
  ba = b - a
  cb = c - b
  ed = e - d
  fe = f - e
  return (ba.x * cb.y - ba.y * cb.x) * (ed.x * fe.y - ed.y * fe.x) > 0


def check_const_angle(points: list[Point]) -> bool:
  """Check if the angle is equal to the given constant."""
  a, b, c, d, m, n = points
  a, b, c, d = bring_together(a, b, c, d)
  ba = b - a
  dc = d - c

  a3 = np.arctan2(ba.y, ba.x)
  a4 = np.arctan2(dc.y, dc.x)
  y = a3 - a4

  return close_enough(m / n % 1, y / np.pi % 1)


def check_eqangle(points: list[Point]) -> bool:
  """Check if 8 points make 2 equal angles."""
  a, b, c, d, e, f, g, h = points

  ab = Line(a, b)
  cd = Line(c, d)
  ef = Line(e, f)
  gh = Line(g, h)

  if ab.is_parallel(cd):
    return ef.is_parallel(gh)
  if ef.is_parallel(gh):
    return ab.is_parallel(cd)

  a, b, c, d = bring_together(a, b, c, d)
  e, f, g, h = bring_together(e, f, g, h)

  ba = b - a
  dc = d - c
  fe = f - e
  hg = h - g

  sameclock = (ba.x * dc.y - ba.y * dc.x) * (fe.x * hg.y - fe.y * hg.x) > 0
  if not sameclock:
    ba = ba * -1.0

  a1 = np.arctan2(fe.y, fe.x)
  a2 = np.arctan2(hg.y, hg.x)
  x = a1 - a2

  a3 = np.arctan2(ba.y, ba.x)
  a4 = np.arctan2(dc.y, dc.x)
  y = a3 - a4

  xy = (x - y) % (2 * np.pi)
  return close_enough(xy, 0, tol=1e-11) or close_enough(
      xy, 2 * np.pi, tol=1e-11
  )


def check_eqratio(points: list[Point]) -> bool:
  a, b, c, d, e, f, g, h = points
  ab = a.distance(b)
  cd = c.distance(d)
  ef = e.distance(f)
  gh = g.distance(h)
  return close_enough(ab * gh, cd * ef)


def check_cong(points: list[Point]) -> bool:
  a, b, c, d = points
  return close_enough(a.distance(b), c.distance(d))


def check_midp(points: list[Point]) -> bool:
  a, b, c = points
  return check_coll(points) and close_enough(a.distance(b), a.distance(c))


def check_simtri(points: list[Point]) -> bool:
  """Check if 6 points make a pair of similar triangles."""
  a, b, c, x, y, z = points
  ab = a.distance(b)
  bc = b.distance(c)
  ca = c.distance(a)
  xy = x.distance(y)
  yz = y.distance(z)
  zx = z.distance(x)
  tol = 1e-9
  return close_enough(ab * yz, bc * xy, tol) and close_enough(
      bc * zx, ca * yz, tol
  )


def check_contri(points: list[Point]) -> bool:
  a, b, c, x, y, z = points
  ab = a.distance(b)
  bc = b.distance(c)
  ca = c.distance(a)
  xy = x.distance(y)
  yz = y.distance(z)
  zx = z.distance(x)
  tol = 1e-9
  return (
      close_enough(ab, xy, tol)
      and close_enough(bc, yz, tol)
      and close_enough(ca, zx, tol)
  )


def check_ratio(points: list[Point]) -> bool:
  a, b, c, d, m, n = points
  ab = a.distance(b)
  cd = c.distance(d)
  return close_enough(ab * n, cd * m)


def draw_angle(
    ax: matplotlib.axes.Axes,
    svg,
    head: Point,
    p1: Point,
    p2: Point,
    head_name = None,
    color: Any = 'black',
    alpha: float = 1.0,
    frac: float = 1.0,
    deg = None,
    lines = None,
    circles = None,
    fontsize = 15,
    fill_color = None
):
  """Draw an angle on plt ax."""
  d1 = p1 - head
  d2 = p2 - head

  norm1 = np.sqrt(d1.x ** 2 + d1.y ** 2)
  norm2 = np.sqrt(d2.x ** 2 + d2.y ** 2)
  d3 = ((d1 / norm1) + (d2 / norm2)) / 2
  a1_ = np.arctan2(float(d1.y), float(d1.x))
  a2_ = np.arctan2(float(d2.y), float(d2.x))
  a3_ = np.arctan2(float(d3.y), float(d3.x))

  a1, a2 = a1_ * 180 / np.pi, a2_ * 180 / np.pi
  a1, a2 = a1 % 360, a2 % 360

  if a1 > a2:
    a1, a2 = a2, a1

  if a2 - a1 > 180:
    a1, a2 = a2, a1

  b1, b2 = a1, a2
  if b1 > b2:
    b2 += 360
  d = b2 - b1
  # if d >= 90:
  #   return

  # scale = min(2.0, 90 / d)
  # scale = max(scale, 0.4)
  r = fontsize
  fov = matplotlib.patches.Wedge(
      (float(head.x), float(head.y)),
      r * 1.5,
      a1,
      a2,
      fill=True if fill_color is not None else False,
      facecolor=fill_color,
      edgecolor=color,
      alpha=1.0  # alpha,
  )
  ax.add_artist(fov)
  if not (deg is None or lines is None or circles is None):
    tmp = a3_ * 180 / np.pi
    # x = Point(head.x + r * np.cos(a3_), head.y + r * np.sin(a3_))
    # pos_x, pos_y = naming_position(ax, x, lines, circles)
    pos_x, pos_y = head.x + 1.5 * r * np.cos(a3_), head.y + 1.5 * r * np.sin(a3_)
    if tmp > 90:
      pos_x = pos_x - fontsize * 3
    elif tmp < -90:
      pos_x = pos_x - fontsize * 3
      pos_y = pos_y - fontsize * 1.5
    elif tmp >= -90 and tmp < 0:
      pos_y = pos_y - fontsize * 1.5

    a, b = deg.split('pi/')
    deg = str(int(int(a) * 180 / int(b)))
    name = deg + "°"
    ax.annotate(name, [pos_x, pos_y], color=color, fontsize=fontsize)
    svg.append(f'<text x="{pos_x}" y="{400 - pos_y}" fill="black">{name}</text>\n')

    xmin = min(head.x, pos_x)
    xmax = max(head.x, pos_x + 4 * fontsize)
    ymin = min(head.y, pos_y)
    ymax = max(head.y, pos_y + 3 * fontsize)

    bbox_x, bbox_y = xmin, ymin
    h, w = xmax - xmin, ymax - ymin
    return [bbox_x, bbox_y, h, w, f"Degree {deg} {head_name}"]


def draw_perp(
    ax: matplotlib.axes.Axes,
    svg,
    head: Point,
    p1: Point,
    p2: Point,
    angle_name = None,
    color: Any = 'black',
    alpha: float = 1.0,
    frac: float = 1.0,
    fill_color = None,
    lw=2.0
):
  """Draw an angle on plt ax."""
  _draw_line(ax, svg, head, p1, color=color, lw=lw)
  _draw_line(ax, svg, head, p2, color=color, lw=lw)
  d1 = p1 - head
  d2 = p2 - head

  norm1 = np.sqrt(d1.x ** 2 + d1.y ** 2)
  norm2 = np.sqrt(d2.x ** 2 + d2.y ** 2)
  d3 = ((d1 / norm1) + (d2 / norm2)) / 2
  a1_ = np.arctan2(float(d1.y), float(d1.x))
  a2_ = np.arctan2(float(d2.y), float(d2.x))
  a3_ = np.arctan2(float(d3.y), float(d3.x))
  a1, a2 = a1_ * 180 / np.pi, a2_ * 180 / np.pi
  a1, a2 = a1 % 360, a2 % 360

  if a1 > a2:
    a1, a2 = a2, a1

  if a2 - a1 > 180:
    a1, a2 = a2, a1

  b1, b2 = a1, a2
  if b1 > b2:
    b2 += 360

  r = 15

  new_p1 = Point(head.x + r * np.cos(a1_), head.y + r * np.sin(a1_))
  new_p2 = Point(head.x + r * np.cos(a2_), head.y + r * np.sin(a2_))
  new_p3 = Point(head.x + r * np.sqrt(2) * np.cos(a3_), head.y + r * np.sqrt(2) * np.sin(a3_))

  _draw_line(ax, svg, new_p1, new_p3, color=color, lw=lw)
  _draw_line(ax, svg, new_p2, new_p3, color=color, lw=lw)
  xs = [head.x, new_p1.x, new_p3.x, new_p2.x]
  ys = [head.y, new_p1.y, new_p3.y, new_p2.y]
  ax.fill(xs, ys, color=fill_color)
  
  coors = " ".join([f"{x},{400 - y}" for x, y in zip(xs, ys)])
  svg.append(f'<polygon points="{coors}" fill="blue" fill-rule="nonzero" />\n')
  
  xmin = min(head.x, new_p1.x, new_p2.x, new_p3.x) - 5
  xmax = max(head.x, new_p1.x, new_p2.x, new_p3.x) + 5
  ymin = min(head.y, new_p1.y, new_p2.y, new_p3.y) - 5
  ymax = max(head.y, new_p1.y, new_p2.y, new_p3.y) + 5
  
  bbox_x, bbox_y = xmin, ymin
  h, w = xmax - xmin, ymax - ymin
  return [bbox_x, bbox_y, h, w, f"Perp {angle_name}"]
  

def naming_position(
    ax: matplotlib.axes.Axes, p: Point, lines: list[Line], circles: list[Circle], fontsize=15
) -> tuple[float, float]:
  """Figure out a good naming position on the drawing."""
  _ = ax
  r = fontsize
  c = Circle(center=p, radius=r)
  avoid = []
  for p1, p2 in lines:
    try:
      avoid.extend(circle_segment_intersect(c, p1, p2))
    except InvalidQuadSolveError:
      continue

  for x in circles:
    try:
      avoid.extend(circle_circle_intersection(c, x))
    except InvalidQuadSolveError:
      continue

  if not avoid:
    return [p.x + 0.01, p.y + 0.01]

  angs = sorted([ang_of(p, a) for a in avoid])
  angs += [angs[0] + 2 * np.pi]
  angs = [(angs[i + 1] - a, a) for i, a in enumerate(angs[:-1])]

  d, a = max(angs)
  ang = a + d / 2
  name_pos = p + Point(np.cos(ang), np.sin(ang)) * 2 * r

  x, y = (name_pos.x - r / 1.5, name_pos.y - r / 1.5)
  return x, y


def draw_point(
    ax: matplotlib.axes.Axes,
    svg, 
    p: Point,
    name: str,
    lines: list[Line],
    circles: list[Circle],
    color: Any = 'black',
    fontsize: float = 15,
    is_dot=True
):
  """draw a point."""
  if is_dot:
    ax.scatter(p.x, p.y, color=color, s=15)

  if color == 'white':
    color = 'lightgreen'
  else:
    color = 'black'

  name = name.upper()
  if len(name) > 1:
    name = name[0] + '_' + name[1:]

  # r = 20
  # if random.random() > 0.5:
  #   pos_x, pos_y = p.x + (random.random() - 0.5) * (r - 5), p.y + 5  # p.y - random.random() * (r - 5)
  # else:
  #   pos_x, pos_y = p.x + 5, p.y + (random.random() - 0.5) * (r - 5)
  pos_x, pos_y = naming_position(ax, p, lines, circles, fontsize)
  ax.annotate(
      name, [pos_x, pos_y], color=color, fontsize=fontsize
  )

  minx = min(pos_x, p.x)
  miny = min(pos_y, p.y)
  maxx = max(pos_x, p.x + 2 * fontsize)
  maxy = max(pos_y, p.y + 2 * fontsize)

  svg.append(f'<text x="{pos_x}" y="{400 - pos_y}" fill="black">{name}</text>\n')

  return [minx - 5, miny - 5, maxx - minx + 10, maxy - miny + 10, f"Point {name}"]


def _draw_line(
    ax: matplotlib.axes.Axes,
    svg,
    p1: Point,
    p2: Point,
    color: Any = 'black',
    lw: float = 2.0,
    alpha: float = 1.0,
) -> None:
  """Draw a line in matplotlib."""
  ls = '-'
  if color == '--':
    color = 'black'
    ls = '--'

  lx, ly = (p1.x, p2.x), (p1.y, p2.y)
  svg.append(f'<line x1="{p1.x}" y1="{400 - p1.y}" x2="{p2.x}" y2="{400 - p2.y}" style="stroke:black;stroke-width:{lw}" />\n')
  ax.plot(lx, ly, color=color, lw=lw, alpha=alpha, ls=ls)


def draw_line(
    ax: matplotlib.axes.Axes, svg, line: Line, color: Any = 'black', lw=2.0
) -> tuple[Point, Point]:
  """Draw a line."""
  points = line.neighbors(gm.Point)
  if len(points) <= 1:
    return

  points = [p.num for p in points]
  p1, p2 = points[:2]
  # svg.append(f'<line x1="{p1.num.x}" y1="{400 - p1.num.y}" x2="{p2.num.x}" y2="{400 - p2.num.y}" style="stroke:black;stroke-width:{lw}" />\n')

  pmin, pmax = (p1, 0.0), (p2, (p2 - p1).dot(p2 - p1))

  for p in points[2:]:
    v = (p - p1).dot(p2 - p1)
    if v < pmin[1]:
      pmin = p, v
    if v > pmax[1]:
      pmax = p, v

  p1, p2 = pmin[0], pmax[0]
  _draw_line(ax, svg, p1, p2, color=color, lw=lw)
  return p1, p2


def draw_line2(
    ax: matplotlib.axes.Axes, svg, points, color: Any = 'black', lw=2.0
) -> tuple[Point, Point]:
  """Draw a line."""
  if len(points) <= 1:
    return

  points = [p.num for p in points]
  p1, p2 = points[:2]

  pmin, pmax = (p1, 0.0), (p2, (p2 - p1).dot(p2 - p1))

  for p in points[2:]:
    v = (p - p1).dot(p2 - p1)
    if v < pmin[1]:
      pmin = p, v
    if v > pmax[1]:
      pmax = p, v

  p1, p2 = pmin[0], pmax[0]
  _draw_line(ax, svg, p1, p2, color=color, lw=lw)
  return p1, p2


def _draw_circle(
    ax: matplotlib.axes.Axes, c: Circle, color: Any = 'black', lw: float = 1.5
) -> None:
  ls = '-'
  if color == '--':
    color = 'black'
    ls = '--'

  ax.add_patch(
      plt.Circle(
          (c.center.x, c.center.y),
          c.radius,
          color=color,
          alpha=1.0,
          fill=False,
          lw=lw,
          ls=ls,
      )
  )


def draw_circle(
    ax: matplotlib.axes.Axes, circle: Circle, color: Any = 'black', lw=2.0
) -> Circle:
  """Draw a circle."""
  if circle.num is not None:
    circle = circle.num
  else:
    points = circle.neighbors(gm.Point)
    if len(points) <= 2:
      return
    points = [p.num for p in points]
    p1, p2, p3 = points[:3]
    circle = Circle(p1=p1, p2=p2, p3=p3)

  _draw_circle(ax, circle, color, lw=lw)
  return circle


def mark_segment(
    ax: matplotlib.axes.Axes, p1: Point, p2: Point, color: Any, alpha: float
) -> None:
  _ = alpha
  x, y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
  ax.scatter(x, y, color=color, alpha=1.0, marker='o', s=50)


def highlight_angle(
    ax: matplotlib.axes.Axes,
    a: Point,
    b: Point,
    c: Point,
    d: Point,
    color: Any,
    alpha: float,
) -> None:
  """Highlight an angle between ab and cd with (color, alpha)."""
  try:
    a, b, c, d = bring_together(a, b, c, d)
  except:  # pylint: disable=bare-except
    return
  draw_angle(ax, a, b, d, color=color, alpha=alpha, frac=1.0)


def highlight(
    ax: matplotlib.axes.Axes,
    svg,
    name: str,
    args: list[gm.Point],
    lcolor: Any,
    color1: Any,
    color2: Any,
    lines: Any,
    circles: Any,
    min_scale: float,
    fontsize=15,
    fill_color=None,
    lw=2.0
):
  """Draw highlights."""
  args_org = args
  args = list(map(lambda x: x.num if isinstance(x, gm.Point) else x, args))

  bboxes = []
  if name == 'cyclic':
    if len(args) == 4:
      a, b, c, d = args
      _draw_circle(ax, Circle(p1=a, p2=b, p3=c), color=color1, lw=lw)
  if name == 'coll':
    if len(args) != 3:
      return bboxes
    a, b, c = args
    a, b = max(a, b, c), min(a, b, c)
    _draw_line(ax, svg, a, b, color=color1, lw=lw)
  if name == 'para':
    a, b, c, d = args
    _draw_line(ax, svg, a, b, color=color1, lw=lw)
    _draw_line(ax, svg, c, d, color=color2, lw=lw)
  if name == 'eqangle':
    a, b, c, d, e, f, g, h = args

    x = line_line_intersection(Line(a, b), Line(c, d))
    if b.distance(x) > a.distance(x):
      a, b = b, a
    if d.distance(x) > c.distance(x):
      c, d = d, c
    a, b, d = x, a, c

    y = line_line_intersection(Line(e, f), Line(g, h))
    if f.distance(y) > e.distance(y):
      e, f = f, e
    if h.distance(y) > g.distance(y):
      g, h = h, g
    e, f, h = y, e, g

    _draw_line(ax, svg, a, b, color=lcolor, lw=lw)
    _draw_line(ax, svg, a, d, color=lcolor, lw=lw)
    _draw_line(ax, svg, e, f, color=lcolor, lw=lw)
    _draw_line(ax, svg, e, h, color=lcolor, lw=lw)
    # if color1 == '--':
    #   color1 = 'red'
    # draw_angle(ax, a, b, d, color=color1, alpha=0.5)
    # if color2 == '--':
    #   color2 = 'red'
    # draw_angle(ax, e, f, h, color=color2, alpha=0.5)
  if name == 'aconst':
    a, b, c, d, val = args
    if a == c:
      head_name = f"{args_org[1].name.upper()}{args_org[0].name.upper()}{args_org[3].name.upper()}"
    elif a == d:
      head_name = f"{args_org[1].name.upper()}{args_org[0].name.upper()}{args_org[2].name.upper()}"
    elif b == c:
      head_name = f"{args_org[0].name.upper()}{args_org[1].name.upper()}{args_org[3].name.upper()}"
    elif b == d:
      head_name = f"{args_org[0].name.upper()}{args_org[1].name.upper()}{args_org[2].name.upper()}"
    else:
      head_name = f"{args_org[0].name.upper()}{args_org[1].name.upper()},{args_org[2].name.upper()}{args_org[3].name.upper()}"

    x = line_line_intersection(Line(a, b), Line(c, d))
    if b.distance(x) > a.distance(x):
      a, b = b, a
    if d.distance(x) > c.distance(x):
      c, d = d, c
    a, b, d = x, a, c
    _draw_line(ax, svg, a, b, color=lcolor, lw=lw)
    _draw_line(ax, svg, a, d, color=lcolor, lw=lw)
    bbox = draw_angle(ax, svg, a, b, d, head_name=head_name, color=color1, frac=min_scale, deg=val.name, lines=lines, circles=circles, fontsize=fontsize, fill_color=fill_color)
    bboxes.append(bbox)
  if name == 'rconst':
    a, b, c, d, val = args
    _draw_line(ax, svg, a, b, color=lcolor, lw=lw)
    _draw_line(ax, svg, c, d, color=lcolor, lw=lw)
    m1 = (a + b) / 2
    m2 = (c + d) / 2
    a, b = val.name.split('/')
    ax.annotate(
      a, naming_position(ax, m1, lines, circles, fontsize), color=lcolor, fontsize=fontsize
    )
    ax.annotate(
      b, naming_position(ax, m2, lines, circles, fontsize), color=lcolor, fontsize=fontsize
    )
    
  if name == 'perp':
    a, b, c, d = args
    head_name = f"{args_org[0].name.upper()}{args_org[1].name.upper()} {args_org[2].name.upper()}{args_org[3].name.upper()}"
    # _draw_line(ax, svg, a, b, color=color1, lw=lw)
    # _draw_line(ax, svg, c, d, color=color1, lw=lw)
    x = line_line_intersection(Line(a, b), Line(c, d))
    if b.distance(x) > a.distance(x):
      a, b = b, a
    if d.distance(x) > c.distance(x):
      c, d = d, c
    a, b, d = x, a, c
    
    bbox = draw_perp(ax, svg, a, b, d, head_name, frac=min_scale, color=color1, fill_color=fill_color, lw=lw)
    # bboxes.append(bbox)

  if name == 'ratio':
    a, b, c, d, m, n = args
    _draw_line(ax, svg, a, b, color=color1, lw=lw)
    _draw_line(ax, svg, c, d, color=color2, lw=lw)
  if name == 'cong':
    a, b, c, d = args
    _draw_line(ax, svg, a, b, color=color1, lw=lw)
    _draw_line(ax, svg, c, d, color=color2, lw=lw)
  if name == 'midp':
    m, a, b = args
    _draw_line(ax, svg, a, m, color=color1, lw=lw)
    _draw_line(ax, svg, b, m, color=color2, lw=lw)
  if name == 'eqratio':
    a, b, c, d, m, n, p, q = args
    _draw_line(ax, svg, a, b, color=color1, lw=lw, alpha=1.0)
    _draw_line(ax, svg, c, d, color=color2, lw=lw, alpha=1.0)
    _draw_line(ax, svg, m, n, color=color1, lw=lw, alpha=1.0)
    _draw_line(ax, svg, p, q, color=color2, lw=lw, alpha=1.0)

  return bboxes


HCOLORS = None


def _draw(
    ax: matplotlib.axes.Axes,
    svg, 
    points: list[gm.Point],
    lines: list[gm.Line],
    circles: list[gm.Circle],
    goal: Any,
    equals: list[tuple[Any, Any]],
    highlights: Any,
    draw_all_lines: bool=False,
    draw_square: bool=False,
    fontsize=15,
    is_dot=True,
    fill_color=None,
    lw=2.0
):
  """Draw everything."""
  # colors = ['red', 'green', 'blue', 'orange', 'magenta', 'purple']
  colors = ['black']
  pcolor = 'black'
  lcolor = 'black'
  ccolor = 'grey'
  if get_theme() == 'dark':
    pcolor, lcolor, ccolor = 'white', 'white', 'white'  # , 'cyan'
  elif get_theme() == 'light':
    pcolor, lcolor, ccolor = 'black', 'black', 'black'
  elif get_theme() == 'grey':
    pcolor, lcolor, ccolor = 'black', 'black', 'grey'
    colors = ['grey']

  line_boundaries = []
  min_scale = 1e9
  for l in lines:
    p1, p2 = draw_line(ax, svg, l, color=lcolor, lw=lw)
    line_boundaries.append((p1, p2))
    min_scale = min(min_scale, p1.distance(p2))
  circles = [draw_circle(ax, c, color=ccolor) for c in circles]

  bboxes = []
  for p in points:
    bbox = draw_point(ax, svg, p.num, p.name, line_boundaries, circles, color=pcolor, fontsize=fontsize, is_dot=is_dot)
    p.bbox = bbox
    bboxes.append(bbox)

  if equals:
    for i, segs in enumerate(equals['segments']):
      color = colors[i % len(colors)]
      for a, b in segs:
        mark_segment(ax, a, b, color, 0.5)

    for i, angs in enumerate(equals['angles']):
      color = colors[i % len(colors)]
      for a, b, c, d in angs:
        highlight_angle(ax, a, b, c, d, color, 0.5)

  if highlights:
    global HCOLORS
    if HCOLORS is None:
      HCOLORS = [k for k in mcolors.TABLEAU_COLORS.keys() if 'red' not in k]

    for i, (name, args) in enumerate(highlights):
      # color_i = HCOLORS[0]  # HCOLORS[i % len(HCOLORS)]
      # if name != 'eqangle':
      #   continue
      color_i = 'black'
      bboxes_ = highlight(ax, svg, name, args, color_i, color_i, color_i, line_boundaries, circles, min_scale, fontsize, fill_color, lw)
      bboxes += bboxes_

  if draw_all_lines:
    p_idx = len(points)
    for i in range(p_idx):
      x = points[i]
      y = points[(i + 1) % p_idx]
      draw_line2(ax, svg, [x, y], color=lcolor)

  if draw_square:
    p_idx = 4
    for i in range(p_idx):
      x = points[i]
      y = points[(i + 1) % p_idx]
      draw_line2(ax, svg, [x, y], color=lcolor)

  # if goal:
  #   name, args = goal
  #   lcolor = color1 = color2 = 'black'
  #   highlight(ax, name, args, lcolor, color1, color2, line_boundaries, circles, min_scale)

  return bboxes


THEME = 'dark'


def set_theme(theme) -> None:
  global THEME
  THEME = theme


def get_theme() -> str:
  return THEME


def draw(
    points: list[gm.Point],
    lines: list[gm.Line],
    circles: list[gm.Circle],
    segments: list[gm.Segment],
    goal: Any = None,
    highlights: Any = None,
    equals: list[tuple[Any, Any]] = None,
    block: bool = True,
    save_to: str = None,
    theme: str = 'dark',
    figname: str = 'test.png',
    draw_all_lines: bool = False,
    draw_square: bool = False,
    image_size: int = 200,
    fontsize = 10,
    font = 'sans-serif',
    is_dot = True,
    fill_color = None,
    lw = 1.2
):
  """Draw everything on the same canvas."""
  image_size = random.choice(list(range(200, 401, 20)))
  fontsize = random.choice(list(range(8, 15)))
  font = random.choice(['sans-serif', 'serif', 'monospace'])
  lw = random.uniform(0.8, 2.0)

  plt.close()

  rcParams['font.family'] = font

  imsize = image_size / 100
  fig, ax = plt.subplots(figsize=(imsize, imsize), dpi=100)
  svg = []  # open(figname.replace('png', 'svg'), 'w')
  # svg.append(f'<svg height="{image_size + 100}" width="{image_size + 100}">\n')
  if points:
    xmin = min([p.num.x for p in points])
    xmax = max([p.num.x for p in points])
    ymin = min([p.num.y for p in points])
    ymax = max([p.num.y for p in points])

    for circle in circles:
      xmin = min(xmin, circle.num.center.x - circle.num.radius)
      ymin = min(ymin, circle.num.center.y - circle.num.radius)
      xmax = max(xmax, circle.num.center.x + circle.num.radius)
      ymax = max(ymax, circle.num.center.y + circle.num.radius)

    x_end = image_size - 20
    y_end = image_size - 20
    x_offset = -xmin
    y_offset = -ymin
    extra_offset = 50
    x_mul = x_end / (xmax + x_offset)
    y_mul = y_end / (ymax + y_offset)
    mul = min(x_mul, y_mul)

    for circle in circles:
      circle.num.radius = circle.num.radius * mul
      flag = True
      for p in points:
          if p.num.x == circle.num.center.x and p.num.y == circle.num.center.y:
              flag = False
              break

      if flag:
        circle.num.center.x = (circle.num.center.x + x_offset) * mul + extra_offset
        circle.num.center.y = (circle.num.center.y + y_offset) * mul + extra_offset

      svg.append('<circle r="{circle.num.radius}" cx="{circle.num.center.x}" cy="{image_size - circle.num.center.y}" stroke="black" stroke-width="{lw}" fill="none" />\n')

    for p in points:
      p.num.x = (p.num.x + x_offset) * mul + extra_offset
      p.num.y = (p.num.y + y_offset) * mul + extra_offset

  set_theme(theme)

  if get_theme() == 'dark':
    ax.set_facecolor((0.0, 0.0, 0.0))
  else:
    ax.set_facecolor((1.0, 1.0, 1.0))

  bboxes = _draw(ax, svg, points, lines, circles, goal, equals, highlights, draw_all_lines, draw_square, fontsize, is_dot, fill_color, lw)

  for line in lines:
    points = line.neighbors(gm.Point)
    p1, p2 = points[:2]
    # svg.append(f'<line x1="{p1.num.x}" y1="{image_size + 100 - p1.num.y}" x2="{p2.num.x}" y2="{image_size + 100 - p2.num.y}" style="stroke:black;stroke-width:{lw}" />\n')
    xmin = min(p1.bbox[0], p2.bbox[0])
    xmax = max(p1.bbox[0] + p1.bbox[2], p2.bbox[0] + p2.bbox[2])
    ymin = min(p1.bbox[1], p2.bbox[1])
    ymax = max(p1.bbox[1] + p1.bbox[3], p2.bbox[1] + p2.bbox[3])

    bbox_x, bbox_y = xmin, ymin
    h, w = xmax - xmin, ymax - ymin
    bboxes.append([bbox_x, bbox_y, h, w, f"Line {p1.name.upper()}{p2.name.upper()}"])

  # ax.set_xlim(-50, image_size + 50)
  # ax.set_ylim(-50, image_size + 50)
  ax.set_xlim(0, image_size + 100)
  ax.set_ylim(0, image_size + 100)
  plt.axis('off')
  plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
  plt.show(block=block)
  plt.savefig(figname)

  bbox_ret = []
  for bbox in bboxes:
    x, y, h, w, label = bbox
    x, y = (x + 50) * image_size / (image_size + 100), (y + 50) * image_size / (image_size + 100)
    h, w = h * image_size / (image_size + 100), w * image_size / (image_size + 100)
    x, y = int(x), int(y)
    h, w = int(h), int(w)

    y = image_size - y - w
    bbox_ret.append([x, y, h, w, label])

  # svg.append('</svg>')

  # svg = list(set(svg))
  # with open(f"{figname.replace('png', 'svg')}", 'w') as f:
  #   f.write(f'<svg height="{image_size + 100}" width="{image_size + 100}">\n')
  #   for row in svg:
  #     f.write(row)
  #   f.write('</svg>')

  return bbox_ret


def close_enough(a: float, b: float, tol: float = 1e-12) -> bool:
  return abs(a - b) < tol


def assert_close_enough(a: float, b: float, tol: float = 1e-12) -> None:
  assert close_enough(a, b, tol), f'|{a}-{b}| = {abs(a-b)} >= {tol}'


def ang_of(tail: Point, head: Point) -> float:
  vector = head - tail
  arctan = np.arctan2(vector.y, vector.x) % (2 * np.pi)
  return arctan


def ang_between(tail: Point, head1: Point, head2: Point) -> float:
  ang1 = ang_of(tail, head1)
  ang2 = ang_of(tail, head2)
  diff = ang1 - ang2
  # return diff % (2*np.pi)
  if diff > np.pi:
    return diff - 2 * np.pi
  if diff < -np.pi:
    return 2 * np.pi + diff
  return diff


def head_from(tail: Point, ang: float, length: float = 1) -> Point:
  vector = Point(np.cos(ang) * length, np.sin(ang) * length)
  return tail + vector


def random_points(n: int = 3) -> list[Point]:
  return [Point(unif(-1, 1), unif(-1, 1)) for _ in range(n)]


def random_rfss(*points: list[Point]) -> list[Point]:
  """Random rotate-flip-scale-shift a point cloud."""
  # center point cloud.
  average = sum(points, Point(0.0, 0.0)) * (1.0 / len(points))
  points = [p - average for p in points]

  # rotate
  ang = unif(0.0, 2 * np.pi)
  sin, cos = np.sin(ang), np.cos(ang)
  # scale and shift
  scale = unif(0.5, 2.0)
  shift = Point(unif(-1, 1), unif(-1, 1))
  points = [p.rotate(sin, cos) * scale + shift for p in points]

  # randomly flip
  if np.random.rand() < 0.5:
    points = [p.flip() for p in points]

  return points


def reduce(
    objs: list[Union[Point, Line, Circle, HalfLine, HoleCircle]],
    existing_points: list[Point],
) -> list[Point]:
  """Reduce intersecting objects into one point of intersections."""
  if all(isinstance(o, Point) for o in objs):
    return objs

  elif len(objs) == 1:
    return objs[0].sample_within(existing_points)

  elif len(objs) == 2:
    a, b = objs
    result = a.intersect(b)
    if isinstance(result, Point):
      return [result]
    a, b = result
    a_close = any([a.close(x) for x in existing_points])
    if a_close:
      return [b]
    b_close = any([b.close(x) for x in existing_points])
    if b_close:
      return [a]
    return [np.random.choice([a, b])]

  else:
    raise ValueError(f'Cannot reduce {objs}')


def sketch(
    name: str, args: list[Union[Point, gm.Point]]
) -> list[Union[Point, Line, Circle, HalfLine, HoleCircle]]:
  fun = globals()['sketch_' + name]
  args = [p.num if isinstance(p, gm.Point) else p for p in args]
  out = fun(args)

  # out can be one or multiple {Point/Line/HalfLine}
  if isinstance(out, (tuple, list)):
    return list(out)
  return [out]


def sketch_on_opline(args: tuple[gm.Point, ...]) -> HalfLine:
  a, b = args
  return HalfLine(a, a + a - b)


def sketch_on_hline(args: tuple[gm.Point, ...]) -> HalfLine:
  a, b = args
  return HalfLine(a, b)


def sketch_ieq_triangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(1.0, 0.0)

  c, _ = Circle(a, p1=b).intersect(Circle(b, p1=a))
  return a, b, c


def sketch_incenter2(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a, b, c = args
  l1 = sketch_bisect([b, a, c])
  l2 = sketch_bisect([a, b, c])
  i = line_line_intersection(l1, l2)
  x = i.foot(Line(b, c))
  y = i.foot(Line(c, a))
  z = i.foot(Line(a, b))
  return x, y, z, i


def sketch_excenter2(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a, b, c = args
  l1 = sketch_bisect([b, a, c])
  l2 = sketch_exbisect([a, b, c])
  i = line_line_intersection(l1, l2)
  x = i.foot(Line(b, c))
  y = i.foot(Line(c, a))
  z = i.foot(Line(a, b))
  return x, y, z, i


def sketch_centroid(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a, b, c = args
  x = (b + c) * 0.5
  y = (c + a) * 0.5
  z = (a + b) * 0.5
  i = line_line_intersection(Line(a, x), Line(b, y))
  return x, y, z, i


def sketch_ninepoints(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a, b, c = args
  x = (b + c) * 0.5
  y = (c + a) * 0.5
  z = (a + b) * 0.5
  c = Circle(p1=x, p2=y, p3=z)
  return x, y, z, c.center


def sketch_2l1c(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  """Sketch a circle touching two lines and another circle."""
  a, b, c, p = args
  bc, ac = Line(b, c), Line(a, c)
  circle = Circle(p, p1=a)

  d, d_ = line_circle_intersection(p.perpendicular_line(bc), circle)
  if bc.diff_side(d_, a):
    d = d_

  e, e_ = line_circle_intersection(p.perpendicular_line(ac), circle)
  if ac.diff_side(e_, b):
    e = e_

  df = d.perpendicular_line(Line(p, d))
  ef = e.perpendicular_line(Line(p, e))
  f = line_line_intersection(df, ef)

  g, g_ = line_circle_intersection(Line(c, f), circle)
  if bc.same_side(g_, a):
    g = g_

  b_ = c + (b - c) / b.distance(c)
  a_ = c + (a - c) / a.distance(c)
  m = (a_ + b_) * 0.5
  x = line_line_intersection(Line(c, m), Line(p, g))
  return x.foot(ac), x.foot(bc), g, x


def sketch_3peq(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a, b, c = args
  ab, bc, ca = Line(a, b), Line(b, c), Line(c, a)

  z = b + (c - b) * np.random.uniform(-0.5, 1.5)

  z_ = z * 2 - c
  l = z_.parallel_line(ca)
  x = line_line_intersection(l, ab)
  y = z * 2 - x
  return x, y, z


def try_to_sketch_intersect(
    name1: str,
    args1: list[Union[gm.Point, Point]],
    name2: str,
    args2: list[Union[gm.Point, Point]],
    existing_points: list[Point],
) -> Optional[Point]:
  """Try to sketch an intersection between two objects."""
  obj1 = sketch(name1, args1)[0]
  obj2 = sketch(name2, args2)[0]

  if isinstance(obj1, Line) and isinstance(obj2, Line):
    fn = line_line_intersection
  elif isinstance(obj1, Circle) and isinstance(obj2, Circle):
    fn = circle_circle_intersection
  else:
    fn = line_circle_intersection
    if isinstance(obj2, Line) and isinstance(obj1, Circle):
      obj1, obj2 = obj2, obj1

  try:
    x = fn(obj1, obj2)
  except:  # pylint: disable=bare-except
    return None

  if isinstance(x, Point):
    return x

  x1, x2 = x

  close1 = check_too_close([x1], existing_points)
  far1 = check_too_far([x1], existing_points)
  if not close1 and not far1:
    return x1
  close2 = check_too_close([x2], existing_points)
  far2 = check_too_far([x2], existing_points)
  if not close2 and not far2:
    return x2

  return None


def sketch_acircle(args: tuple[gm.Point, ...]) -> Circle:
  a, b, c, d, f = args
  de = sketch_aline([c, a, b, f, d])
  fe = sketch_aline([a, c, b, d, f])
  e = line_line_intersection(de, fe)
  return Circle(p1=d, p2=e, p3=f)


def sketch_aline(args: tuple[gm.Point, ...]) -> HalfLine:
  """Sketch the construction aline."""
  A, B, C, D, E = args
  ab = A - B
  cb = C - B
  de = D - E

  dab = A.distance(B)
  ang_ab = np.arctan2(ab.y / dab, ab.x / dab)

  dcb = C.distance(B)
  ang_bc = np.arctan2(cb.y / dcb, cb.x / dcb)

  dde = D.distance(E)
  ang_de = np.arctan2(de.y / dde, de.x / dde)

  ang_ex = ang_de + ang_bc - ang_ab
  X = E + Point(np.cos(ang_ex), np.sin(ang_ex))
  return HalfLine(E, X)


def sketch_amirror(args: tuple[gm.Point, ...]) -> HalfLine:
  """Sketch the angle mirror."""
  A, B, C = args  # pylint: disable=invalid-name
  ab = A - B
  cb = C - B

  dab = A.distance(B)
  ang_ab = np.arctan2(ab.y / dab, ab.x / dab)
  dcb = C.distance(B)
  ang_bc = np.arctan2(cb.y / dcb, cb.x / dcb)

  ang_bx = 2 * ang_bc - ang_ab
  X = B + Point(np.cos(ang_bx), np.sin(ang_bx))  # pylint: disable=invalid-name
  return HalfLine(B, X)


def sketch_bisect(args: tuple[gm.Point, ...]) -> Line:
  a, b, c = args
  ab = a.distance(b)
  bc = b.distance(c)
  x = b + (c - b) * (ab / bc)
  m = (a + x) * 0.5
  return Line(b, m)


def sketch_exbisect(args: tuple[gm.Point, ...]) -> Line:
  a, b, c = args
  return sketch_bisect(args).perpendicular_line(b)


def sketch_bline(args: tuple[gm.Point, ...]) -> Line:
  a, b = args
  m = (a + b) * 0.5
  return m.perpendicular_line(Line(a, b))


def sketch_dia(args: tuple[gm.Point, ...]) -> Circle:
  a, b = args
  return Circle((a + b) * 0.5, p1=a)


def sketch_tangent(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  a, o, b = args
  dia = sketch_dia([a, o])
  return circle_circle_intersection(Circle(o, p1=b), dia)


def sketch_circle(args: tuple[gm.Point, ...]) -> Circle:
  a, b, c = args
  return Circle(center=a, radius=b.distance(c))


def sketch_cc_tangent(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  """Sketch tangents to two circles."""
  o, a, w, b = args
  ra, rb = o.distance(a), w.distance(b)

  ow = Line(o, w)
  if close_enough(ra, rb):
    oo = ow.perpendicular_line(o)
    oa = Circle(o, ra)
    x, z = line_circle_intersection(oo, oa)
    y = x + w - o
    t = z + w - o
    return x, y, z, t

  swap = rb > ra
  if swap:
    o, a, w, b = w, b, o, a
    ra, rb = rb, ra

  oa = Circle(o, ra)
  q = o + (w - o) * ra / (ra - rb)

  x, z = circle_circle_intersection(sketch_dia([o, q]), oa)
  y = w.foot(Line(x, q))
  t = w.foot(Line(z, q))

  if swap:
    x, y, z, t = y, x, t, z

  return x, y, z, t


def sketch_hcircle(args: tuple[gm.Point, ...]) -> HoleCircle:
  a, b = args
  return HoleCircle(center=a, radius=a.distance(b), hole=b)


def sketch_e5128(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  a, b, c, d = args
  ad = Line(a, d)

  g = (a + b) * 0.5
  de = Line(d, g)

  e, f = line_circle_intersection(de, Circle(c, p1=b))

  if e.distance(d) < f.distance(d):
    e = f
  return e, g


def sketch_eq_quadrangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  """Sketch quadrangle with two equal opposite sides."""
  a = Point(0.0, 0.0)
  b = Point(1.0, 0.0)

  length = np.random.uniform(0.5, 2.0)
  ang = np.random.uniform(np.pi / 3, np.pi * 2 / 3)
  d = head_from(a, ang, length)

  ang = ang_of(b, d)
  ang = np.random.uniform(ang / 10, ang / 9)
  c = head_from(b, ang, length)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_eq_trapezoid(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(1.0, 0.0)
  l = unif(0.5, 2.0)

  height = unif(0.5, 2.0)
  c = Point(0.5 + l / 2.0, height)
  d = Point(0.5 - l / 2.0, height)

  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_eqangle2(args: tuple[gm.Point, ...]) -> Point:
  """Sketch the def eqangle2."""
  a, b, c = args

  d = c * 2 - b

  ba = b.distance(a)
  bc = b.distance(c)
  l = ba * ba / bc

  if unif(0.0, 1.0) < 0.5:
    be = min(l, bc)
    be = unif(be * 0.1, be * 0.9)
  else:
    be = max(l, bc)
    be = unif(be * 1.1, be * 1.5)

  e = b + (c - b) * (be / bc)
  y = b + (a - b) * (be / l)
  return line_line_intersection(Line(c, y), Line(a, e))


def sketch_eqangle3(args: tuple[gm.Point, ...]) -> Circle:
  a, b, d, e, f = args
  de = d.distance(e)
  ef = e.distance(f)
  ab = b.distance(a)
  ang_ax = ang_of(a, b) + ang_between(e, d, f)
  x = head_from(a, ang_ax, length=de / ef * ab)
  return Circle(p1=a, p2=b, p3=x)


def sketch_eqdia_quadrangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  """Sketch quadrangle with two equal diagonals."""
  m = unif(0.3, 0.7)
  n = unif(0.3, 0.7)
  a = Point(-m, 0.0)
  c = Point(1 - m, 0.0)
  b = Point(0.0, -n)
  d = Point(0.0, 1 - n)

  ang = unif(-0.25 * np.pi, 0.25 * np.pi)
  sin, cos = np.sin(ang), np.cos(ang)
  b = b.rotate(sin, cos)
  d = d.rotate(sin, cos)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_free(args: tuple[gm.Point, ...]) -> Point:
  return random_points(1)[0]


def sketch_isos(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  base = unif(0.5, 1.5)
  height = unif(0.5, 1.5)

  b = Point(-base / 2, 0.0)
  c = Point(base / 2, 0.0)
  a = Point(0.0, height)
  a, b, c = random_rfss(a, b, c)
  return a, b, c


def sketch_line(args: tuple[gm.Point, ...]) -> Line:
  a, b = args
  return Line(a, b)


def sketch_cyclic(args: tuple[gm.Point, ...]) -> Circle:
  a, b, c = args
  return Circle(p1=a, p2=b, p3=c)


def sketch_hline(args: tuple[gm.Point, ...]) -> HalfLine:
  a, b = args
  return HalfLine(a, b)


def sketch_midp(args: tuple[gm.Point, ...]) -> Point:
  a, b = args
  return (a + b) * 0.5


def sketch_pentagon(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  points = [Point(1.0, 0.0)]
  ang = 0.0

  for i in range(4):
    ang += (2 * np.pi - ang) / (5 - i) * unif(0.5, 1.5)
    point = Point(np.cos(ang), np.sin(ang))
    points.append(point)

  a, b, c, d, e = points  # pylint: disable=unbalanced-tuple-unpacking
  a, b, c, d, e = random_rfss(a, b, c, d, e)
  return a, b, c, d, e


def sketch_pline(args: tuple[gm.Point, ...]) -> Line:
  a, b, c = args
  return a.parallel_line(Line(b, c))


def sketch_pmirror(args: tuple[gm.Point, ...]) -> Point:
  a, b = args
  return b * 2 - a


def sketch_quadrangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  """Sketch a random quadrangle."""
  m = unif(0.3, 0.7)
  n = unif(0.3, 0.7)

  a = Point(-m, 0.0)
  c = Point(1 - m, 0.0)
  b = Point(0.0, -unif(0.25, 0.75))
  d = Point(0.0, unif(0.25, 0.75))

  ang = unif(-0.25 * np.pi, 0.25 * np.pi)
  sin, cos = np.sin(ang), np.cos(ang)
  b = b.rotate(sin, cos)
  d = d.rotate(sin, cos)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_r_trapezoid(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 1.0)
  d = Point(0.0, 0.0)
  b = Point(unif(0.5, 1.5), 1.0)
  c = Point(unif(0.5, 1.5), 0.0)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_r_triangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(0.0, unif(0.5, 2.0))
  c = Point(unif(0.5, 2.0), 0.0)
  a, b, c = random_rfss(a, b, c)
  return a, b, c


def sketch_rectangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(0.0, 1.0)
  l = unif(0.5, 2.0)
  c = Point(l, 1.0)
  d = Point(l, 0.0)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_reflect(args: tuple[gm.Point, ...]) -> Point:
  a, b, c = args
  m = a.foot(Line(b, c))
  return m * 2 - a


def sketch_risos(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(0.0, 1.0)
  c = Point(1.0, 0.0)
  a, b, c = random_rfss(a, b, c)
  return a, b, c


def sketch_rotaten90(args: tuple[gm.Point, ...]) -> Point:
  a, b = args
  ang = -np.pi / 2
  return a + (b - a).rotate(np.sin(ang), np.cos(ang))


def sketch_rotatep90(args: tuple[gm.Point, ...]) -> Point:
  a, b = args
  ang = np.pi / 2
  return a + (b - a).rotate(np.sin(ang), np.cos(ang))


def sketch_s_angle(args: tuple[gm.Point, ...]) -> HalfLine:
  a, b, y = args
  ang = y / 180 * np.pi
  x = b + (a - b).rotatea(ang)
  return HalfLine(b, x)


def sketch_segment(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  a, b = random_points(2)
  return a, b


def sketch_shift(args: tuple[gm.Point, ...]) -> Point:
  a, b, c = args
  return c + (b - a)


def sketch_square(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  a, b = args
  c = b + (a - b).rotatea(-np.pi / 2)
  d = a + (b - a).rotatea(np.pi / 2)
  return c, d


def sketch_isquare(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(1.0, 0.0)
  c = Point(1.0, 1.0)
  d = Point(0.0, 1.0)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_tline(args: tuple[gm.Point, ...]) -> Line:
  a, b, c = args
  return a.perpendicular_line(Line(b, c))


def sketch_trapezoid(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  d = Point(0.0, 0.0)
  c = Point(1.0, 0.0)

  base = unif(0.5, 2.0)
  height = unif(0.5, 2.0)
  a = Point(unif(0.2, 0.5), height)
  b = Point(a.x + base, height)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_triangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(1.0, 0.0)
  ac = unif(0.5, 2.0)
  ang = unif(0.2, 0.8) * np.pi
  c = head_from(a, ang, ac)
  return a, b, c


def sketch_triangle12(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  x, y = args
  r0 = min(int(x), int(y))
  r1 = max(int(x), int(y))
  b = Point(0.0, 0.0)
  c = Point(unif(r1 - r0 + 0.5, r0 + r1 - 0.5), 0.0)
  a, _ = circle_circle_intersection(Circle(b, int(x)), Circle(c, int(y)))
  a, b, c = random_rfss(a, b, c)
  return a, b, c


def sketch_trisect(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  """Sketch two trisectors of an angle."""
  a, b, c = args
  ang1 = ang_of(b, a)
  ang2 = ang_of(b, c)

  swap = 0
  if ang1 > ang2:
    ang1, ang2 = ang2, ang1
    swap += 1

  if ang2 - ang1 > np.pi:
    ang1, ang2 = ang2, ang1 + 2 * np.pi
    swap += 1

  angx = ang1 + (ang2 - ang1) / 3
  angy = ang2 - (ang2 - ang1) / 3

  x = b + Point(np.cos(angx), np.sin(angx))
  y = b + Point(np.cos(angy), np.sin(angy))

  ac = Line(a, c)
  x = line_line_intersection(Line(b, x), ac)
  y = line_line_intersection(Line(b, y), ac)

  if swap == 1:
    return y, x
  return x, y


def sketch_trisegment(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  a, b = args
  x, y = a + (b - a) * (1.0 / 3), a + (b - a) * (2.0 / 3)
  return x, y
