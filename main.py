import os
import argparse
from matplotlib import animation

from modules.data_utils import parse_svg
from fourier_core import compute_coeffs

from modules.fourier_utils import compute_apprs
from modules.visualize import evolution_animate

from modules.fourier_utils import compute_circle_centers
from modules.visualize import epicycles_animate



def main(svg_path, num_harmonics=100):
    num_coeffs = 1 + 2*num_harmonics
    print(f"SVG file path: {svg_path}")
    print(f"Number of coefficients used for approximation: {num_harmonics}")

    base = os.path.basename(svg_path)
    file_name = os.path.splitext(base)[0]

    print('\nReading svg')
    points = parse_svg(svg_path)
    points -= points.mean()  # Make it centered at origin (optional).

    print('\nEvolution animation')
    coeffs = compute_coeffs(points, num_coeffs)
    points_apprs = compute_apprs(points, coeffs)
    # At every even index we have sum of c_n and c_-n
    points_apprs = points_apprs[::2]
    anim = evolution_animate(points_apprs)
    anim.save(f"animations/{file_name}_evolution.gif", writer=animation.PillowWriter(fps=10))


    print('\nEpicycles animation')
    n_samples = len(points)  # Number of time points to sample.
    centers_time = compute_circle_centers(coeffs, n_samples)
    anim = epicycles_animate(centers_time)
    anim.save(f"animations/{file_name}_epicycles_{num_harmonics}.gif", writer=animation.PillowWriter(fps=10))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg_path", help="Path to the SVG file")
    parser.add_argument("-n", "--num_harmonics", type=int, default=100,
                        help="Number of coefficients used for approximation (default: 1)")
    args = parser.parse_args()

    main(args.svg_path, args.num_harmonics)