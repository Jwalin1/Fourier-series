import os
import argparse
from matplotlib import animation

from modules.data_utils import parse_svg
from fourier_core import get_coeffs, get_fourier_approximation

from modules.fourier_utils import get_apprs
from modules.visualize import evolution_animate

from modules.fourier_utils import get_circle_centers
from modules.visualize import add_arrows, epicycles_animate



def main(svg_path, num_coeffs=100):
    # Your main code here
    print(f"SVG file path: {svg_path}")
    print(f"Number of coefficients used for approximation: {num_coeffs}")

    base = os.path.basename(svg_path)
    file_name = os.path.splitext(base)[0]

    print('\nReading svg')
    points = parse_svg(svg_path)

    print('\nEvolution animation')
    coeffs = get_coeffs(points, num_coeffs)
    points_apprs = get_apprs(points, coeffs)
    anim = evolution_animate(points_apprs)
    anim.save(f"animations/{file_name}_evolution.gif", writer=animation.PillowWriter(fps=10))


    print('\nEpicycles animation')
    n_samples = len(points)  # Number of time points to sample.
    centers_time = get_circle_centers(coeffs, n_samples)
    anim = epicycles_animate(centers_time)
    anim.save(f"animations/{file_name}_epicycles.gif", writer=animation.PillowWriter(fps=10))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg_path", help="Path to the SVG file")
    parser.add_argument("-n", "--num_coeffs", type=int, default=100,
                        help="Number of coefficients used for approximation (default: 1)")
    args = parser.parse_args()

    main(args.svg_path, args.num_coeffs)