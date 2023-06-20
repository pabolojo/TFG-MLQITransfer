#!/bin/sh

#SBATCH -p gpu       # Partición (cola)

#SBATCH -N 1                # Numero de nodos

#SBATCH -n 8                 # Numero de cores(CPUs)

#SBATCH --gres=gpu:1       # SOLO en particion gpu

#SBATCH -t 0-08:00     # Duración (D-HH:MM)

#SBATCH --mail-type=END,FAIL      # Notificación cuando el trabajo termina o falla

#SBATCH --mail-user=pablo.soetard@estudiante.uam.es # Enviar correo a la dirección

module load Miniconda3
module load cuDNN

source activate tf

python3 main_SAC.py -r $1 -shp > $1_out.txt -hp $2
