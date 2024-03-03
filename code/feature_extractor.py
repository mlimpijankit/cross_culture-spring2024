# ===== IMPORTS =====
import os
import cv2
import scipy
import shutil
import argparse
import statistics
import numpy as np
import pandas as pd
import porespy as ps
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft2
from feature_extractor_utils import *

# ===== ARGS =====
parser = argparse.ArgumentParser()
parser.add_argument('input_directory')
parser.add_argument('output_directory')
args = parser.parse_args()

# ===== BEGIN CODE =====

# setting up input, output paths
input_directory = args.input_directory
output_directory = args.output_directory

assert os.path.exists(input_directory), "ERROR: input directory does not exist"
assert os.path.exists(output_directory), "ERROR: output directory does not exist"

filepaths = []

for file in os.listdir(input_directory):
    if file[-5:]=='.jpeg' or file[-4:]=='.jpg' or file[-4:]=='.png':
        in_file = f'{input_directory}/{file}'
        out_file = f'{output_directory}/{file}'
        filepaths.append((in_file, out_file))

print(f'Beginning feature extraction at path {input_directory} ...\n')
print(f'Number of images: {len(filepaths)}')

# run feature extraction
out = []
index = 0

for count, file in enumerate(filepaths):

    print(f'processing image {count} ...')

    try:

        similarity_scores = []

        for level in [1, 2, 3]:
            similarity_scores.append(self_similarity_ground(file[0], level))
            similarity_scores.append(self_similarity_parent(file[0], level))
            similarity_scores.append(self_similarity_neighbor(file[0], level))

        img_similarity = np.average([i for i in similarity_scores if i == i])
        img_complexity = complexity(file[0])
        img_anisotropy = anisotropy(file[0])
        img_birkoff_measure = img_similarity/img_complexity
        img_fractal_dimension = fractal_dimension(file[0])
        img_fourier_slope = fourier_slope(file[0])

        out.append([f'img_{index}', img_similarity, img_complexity, img_anisotropy, img_birkoff_measure, img_fractal_dimension, img_fourier_slope])

        extension = file[1].split(".")[-1]
        shutil.copy(file[0], f'{output_directory}/img_{index}.{extension}')
        index += 1

    except Exception as e:
        print(f'Error on image: {file}')

df = pd.DataFrame(out, columns=['image', 'similarity', 'complexity', 'anisotropy', 'birkoff_measure', 'fractal_dimension', 'fourier_slope'])
df.to_csv(f'{output_directory}/features.csv', index=False)