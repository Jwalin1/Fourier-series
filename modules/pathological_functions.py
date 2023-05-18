import numpy as np
from sympy import primepi


def prime_staircase(x_range=(0,100), num_points=1000):
  x = np.linspace(*x_range, num_points)
  return x, np.array(list(map(lambda x: float(primepi(x)), x)))

def _cantor(x, n):
  if n==0:
    return x
  else:  
    if 0 <= x < 1/3:
      return 0.5*_cantor(3*x, n-1)
    elif 1/3 <= x <= 2/3:
      return 0.5
    elif 2/3 < x <= 1:
      return 0.5*(1 + _cantor(3*x-2, n-1))

def devils_staircase(x_range=(0,1), num_points=1000):
  x = np.linspace(*x_range, num_points)
  return x, np.array(list(map(lambda x: _cantor(x, 100), x)))

def weierstrass_sin(a=2, x_range=(0,1), num_points=1000,
                    num_iters=20):
  x = np.linspace(*x_range, num_points)
  sum_ = 0
  for k in range(1,100):
    t = np.pi * (k**a)
    sum_ += np.sin(t*x) / t
  return x, sum_

def weierstrass_cos(a=0.5, b=5, x_range=(0,1), num_points=1000,
                    num_iters=20):
  x = np.linspace(*x_range, num_points)
  sum_ = 0
  for k in range(num_iters):
    t = np.pi * (b**k)
    sum_ += (a**k) * np.cos(t*x)
  return x, sum_

def _s(x):
  return abs(np.floor(x+0.5) - x)

def blancmange(x_range=(0,0.5), num_points=1000,
               num_iters=20):
  x = np.linspace(*x_range, num_points)
  sum_ = sum([_s((2**n)*x) / (2**n)
              for n in range(num_iters)])
  return x, sum_

def _question_mark(x):
  if x > 1 or x < 0:
      return np.floor(x) + minkowski(x - np.floor(x))

  p = int(x)
  q = 1
  r = p + 1
  s = 1
  d = 1.0
  y = float(p)

  while True:
      d /= 2
      if y + d == y:
          break

      m = p + r
      if m < 0 or p < 0:
          break

      n = q + s
      if n < 0:
          break

      if x < m / n:
          r = m
          s = n
      else:
          y += d
          p = m
          q = n

  return y + d

def  minkowski(x_range=(0,1), num_points=1000):
  x = np.linspace(*x_range, num_points)
  return x, np.array(list(map(_question_mark, x)))