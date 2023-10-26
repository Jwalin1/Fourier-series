import numpy as np
from tqdm.auto import tqdm
from fourier_core import get_alternating_sequence, fourier_series

tau = 2 * np.pi



def compute_apprs(coeffs, num_samples):
  num_coeffs = len(coeffs)
  points_apprs = np.zeros((num_coeffs, num_samples), dtype=complex)
  for n in tqdm(range(1, 1 + num_coeffs), desc='calculating approximations'):
    # coeffs = [c0, c1, c-1, c2, c-2, ...]
    coeffs_n = dict(list(coeffs.items())[:n])
    points_apprs[n-1] = fourier_series(coeffs_n, num_samples)
  return points_apprs


def compute_apprs_FFT(coeffs, num_coeffs, sequence=None):
  points_apprs = []
  if sequence is None:
    sequence = get_alternating_sequence(num_coeffs)
  else:
    assert len(sequence) == num_coeffs
  coeffs_partial = np.zeros_like(coeffs)
  
  for n in tqdm(range(1,1+num_coeffs)):
    coeffs_partial[sequence[:n]] = coeffs[sequence[:n]]
    points_apprs.append(np.fft.ifft(coeffs_partial))
  
  points_apprs = np.array(points_apprs)
  # Add the constant line from the origin (optional).
  points_apprs = np.insert(points_apprs, 0, 0, axis=0)
  return points_apprs



def compute_circle_centers(coeffs, num_samples, sort=False):
  # Generate data to animate.
  num_centers = len(coeffs)

  time_points = np.linspace(0, 1, num_samples)
  centers_time = np.zeros((num_samples, num_centers), dtype=complex)
  sequence, coeffs = map(np.array, zip(*coeffs.items()))

  if sort:  # Sort all except first coefficient
    sort_indices = np.argsort(abs(coeffs[1:]))[::-1]
    coeffs = np.append(coeffs[0], coeffs[1:][sort_indices])
    sequence =  np.append(0, sequence[1:][sort_indices])

  for i, time in enumerate(tqdm(time_points, desc='calculating centers and radii')):
    vectors = coeffs * np.exp(1j * tau * sequence * time)
    centers = vectors.cumsum(0)
    centers_time[i] = centers

  return centers_time