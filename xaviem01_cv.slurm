#!/bin/bash
#
#SBATCH --job-name=xaviem01_cv_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:v100:1

module load anaconda3/5.3.1
source activate cv2
python fair-sslime/setup.py install
conda install -n cv2 nb_conda_kernels
python fair-sslime/tools/train.py --config_file fair-sslime/extra_scripts/unsupervised_vgg_a_rotation_stl_10.yaml
