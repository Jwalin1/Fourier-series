import numpy as np
from scipy import integrate
tau = 2 * np.pi



'''Gives exact result if full length is used but doesnt improve progressively'''
def DFT(x, n_k = None):
  if n_k is None:
    n_k = len(x)

  N = len(x)
  k = np.arange(n_k)
  n = np.arange(N)

  X = x * np.exp(-1j * tau * k[:,None] * n / N)
  X = X.sum(1)
  return X


def inv_DFT(X, n_n = None):
  if n_n is None:
    n_n = len(X)

  N = len(X)
  k = np.arange(N)
  n = np.arange(n_n)

  x = X * np.exp(1j * tau * k * n[:,None] / N)
  x = x.sum(1) / N
  return x


def get_DFT_approximation(x, n=None):
  if n is None:
    n = len(x)
  X = DFT(x, n)
  x_appr = inv_DFT(X, len(x))
  return x_appr



'''Improves progressively as we increase order of approximation'''
def get_coeffs(pts, order=None, rule='Riemann'):
  if order is None:
    order = len(pts) // 2

  indices = np.arange(1 + 2*order)   #  0,  1,  2,  3,  4, ...
  signs = 2*(indices % 2) - 1      # -1,  1, -1,  1, -1, ...
  magnitudes = (indices + 1) // 2  #  0,  1,  1,  2,  2, ...
  sequence = signs * magnitudes    #  0,  1, -1,  2, -2, ...

  if rule == 'Riemann':
    period = np.linspace(0, 1, len(pts), endpoint=False)
    coeffs = pts * np.exp(-1j * tau * sequence[:,None] * period)
    coeffs = coeffs.sum(axis=1) / len(pts)

  elif rule in ['Trapezoidal', 'Simpson']:
    period = np.linspace(0, 1, len(pts))
    coeffs = pts * np.exp(-1j * tau * sequence[:,None] * period)
    if rule == 'Trapezoidal':
      coeffs = integrate.trapezoid(coeffs, period)
    elif rule == 'Simpson':
      coeffs = integrate.simpson(coeffs, period)

  else:
    raise Exception('Invalid rule')
  
  return coeffs


def fourier_series(coeffs, samples=None):
  if samples is None:
    samples = len(coeffs)
  order = len(coeffs) // 2
  period = np.linspace(0, 1, samples)

  indices = np.arange(len(coeffs))  #  0,  1,  2,  3,  4, ...
  signs = 2*(indices % 2) - 1       # -1,  1, -1,  1, -1, ...
  magnitudes = (indices + 1) // 2   #  0,  1,  1,  2,  2, ...
  sequence = signs * magnitudes     #  0,  1, -1,  2, -2, ...
  
  series = coeffs * np.exp(1j * tau * sequence * period[:,None])
  series = series.sum(axis=1)
  return series


def get_fourier_approximation(pts, order=None):
  coeffs = get_coeffs(pts, order)
  appr_pts = fourier_series(coeffs, len(pts))
  return appr_pts
