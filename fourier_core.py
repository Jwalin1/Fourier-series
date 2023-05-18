import numpy as np
from scipy import integrate
tau = 2 * np.pi



def DFT(x, num_coeffs=None):
  """
    Time goes from 0 to 1.
    Only uses positive coeffs.
    Gives exact result if full length is used
    but doesnt improve progressively.
  """
  if num_coeffs is None:
    num_coeffs = len(x)

  N = len(x)
  k, n = np.arange(num_coeffs), np.arange(N)
  X = np.dot(x, np.exp(-1j * tau * k[:,None] * n / N).T)
  X /= N
  return X


def inv_DFT(X, num_samples=None):
  if num_samples is None:
    num_samples = len(X)

  N = len(X)
  k, n = np.arange(N), np.linspace(0, N-1, num_samples)
  x = np.dot(X, np.exp(1j * tau * k * n[:,None] / N).T)
  return x


def compute_DFT_approximation(x, num_coeffs=None, num_samples=None):
  X = DFT(x, num_coeffs)
  x_appr = inv_DFT(X, num_samples)
  return x_appr



def compute_coeffs(points, num_coeffs=None, rule='Riemann'):
  """
    If rule is Riemann, it would be similar to DFT.
    Time goes from 0 to 1.
    Pairs of positive and negative coeffs are added.
    Improves progressively as we increase the
    number of coefficients used for approximation.
  """
  if num_coeffs is None:
    num_coeffs = len(points)

  indices = np.arange(num_coeffs)  #  0,  1,  2,  3,  4, ...
  signs = 2*(indices % 2) - 1      # -1,  1, -1,  1, -1, ...
  magnitudes = (indices + 1) // 2  #  0,  1,  1,  2,  2, ...
  sequence = signs * magnitudes    #  0,  1, -1,  2, -2, ...

  if rule == 'Riemann':
    # Last point is to be excluded in Riemann sum (Left Rule).
    period = np.linspace(0, 1, len(points), endpoint=False)
    coeffs = np.dot(points, np.exp(-1j * tau * sequence[:,None] * period).T)
    coeffs /= len(points)

  elif rule in ['Trapezoidal', 'Simpson']:
    period = np.linspace(0, 1, len(points))
    coeffs = points * np.exp(-1j * tau * sequence[:,None] * period)
    if rule == 'Trapezoidal':
      coeffs = integrate.trapezoid(coeffs, period)
    elif rule == 'Simpson':
      coeffs = integrate.simpson(coeffs, period)

  else:
    raise Exception('Invalid rule')
  
  return dict(zip(sequence, coeffs))


def fourier_series(coeffs, num_samples=None):
  if num_samples is None:
    num_samples = len(coeffs)
  period = np.linspace(0, 1, num_samples)

  sequence, coeffs = map(np.array, zip(*coeffs.items()))
  series = np.dot(coeffs, np.exp(1j * tau * sequence * period[:,None]).T)
  return series


def compute_fourier_approximation(points, num_coeffs=None, num_samples=None):
  coeffs = compute_coeffs(points, num_coeffs)
  appr_points = fourier_series(coeffs, num_samples)
  return appr_points
