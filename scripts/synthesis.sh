#!/bin/bash
#SBATCH -J graspem
#SBATCH --comment "Data synthesis for GraspEm"
#SBATCH -p SC-A800
#SBATCH --qos pro
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH -t 01-00:00:00

obj_list=("bulb" "camera" "cube" "cylinder" "duck" "knob" "mouse" "pear")

python run.py --object_models duck cylinder --seed 42 --batch_size 1024 --tag demo

# Large-scale synthesis

# for i ((i=0; i<8; i++))
# do
#     for ((j=i; j<8; j++))
#     do
#         o=${obj_list[i]}
#         oo=${obj_list[j]}
#         echo "==== Synthesis for $o and $oo with seed $seed ===="
#         python run.py --object_models $o $oo --seed $seed --batch_size 8192 --tag synthesis-$o+$oo
#     done
# done
