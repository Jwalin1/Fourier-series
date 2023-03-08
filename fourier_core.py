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



'''Improves progressively as we increase number of coefficients used for approximation'''
def get_coeffs(points, num_coeffs=None, rule='Riemann'):
  if num_coeffs is None:
    num_coeffs = len(points)

  indices = np.arange(num_coeffs)   #  0,  1,  2,  3,  4, ...
  signs = 2*(indices % 2) - 1      # -1,  1, -1,  1, -1, ...
  magnitudes = (indices + 1) // 2  #  0,  1,  1,  2,  2, ...
  sequence = signs * magnitudes    #  0,  1, -1,  2, -2, ...

  if rule == 'Riemann':
    period = np.linspace(0, 1, len(points), endpoint=False)
    coeffs = points * np.exp(-1j * tau * sequence[:,None] * period)
    coeffs = coeffs.sum(axis=1) / len(points)

  elif rule in ['Trapezoidal', 'Simpson']:
    period = np.linspace(0, 1, len(points))
    coeffs = points * np.exp(-1j * tau * sequence[:,None] * period)
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
  num_coeffs = len(coeffs)
  period = np.linspace(0, 1, samples)

  indices = np.arange(len(coeffs))  #  0,  1,  2,  3,  4, ...
  signs = 2*(indices % 2) - 1       # -1,  1, -1,  1, -1, ...
  magnitudes = (indices + 1) // 2   #  0,  1,  1,  2,  2, ...
  sequence = signs * magnitudes     #  0,  1, -1,  2, -2, ...
  
  series = coeffs * np.exp(1j * tau * sequence * period[:,None])
  series = series.sum(axis=1)
  return series


def get_fourier_approximation(points, num_coeffs=None):
  coeffs = get_coeffs(points, num_coeffs)
  appr_points = fourier_series(coeffs, len(points))
  return appr_points
