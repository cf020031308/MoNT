#!/bin/bash -

# Heterophilic
python3 -u main.py NT roman_empire --runs 10 --max-epochs 2500 --early-stop-epochs 500 --lr 0.001 --agg sum --dropout 0.4 --to-bidir --add-self-loops --sep --heads 6 --hidden 32 --n-layers 5 # 91.71±0.57
python3 -u main.py NT amazon_ratings --runs 10 --max-epochs 2500 --early-stop-epochs 500 --lr 0.001 --agg mean --dropout 0.3 --to-bidir --remove-self-loops --sep --heads 8 --hidden 40 --n-layers 1 # 54.25±0.50
python3 -u main.py NT minesweeper --runs 10 --max-epochs 2500 --early-stop-epochs 500 --lr 0.001 --agg sum --dropout 0.2 --to-bidir --remove-self-loops --sep --heads 1 --hidden 53 --n-layers 5 # 97.42±0.50
python3 -u main.py NT tolokers --runs 10 --max-epochs 2500 --early-stop-epochs 500 --lr 0.001 --agg gatedsum --dropout 0.1 --to-bidir --sep --heads 2 --hidden 30 --n-layers 5 # 85.69±0.54
python3 -u main.py NT questions --runs 10 --max-epochs 2500 --early-stop-epochs 500 --lr 0.001 --agg sum --dropout 0.2 --to-bidir --remove-self-loops --sep --heads 4 --hidden 32 --n-layers 1 # 78.46±1.10

# Directed Heterophilic
python3 -u main.py NT roman_empire --runs 10 --max-epochs 2500 --early-stop-epochs 500 --lr 0.001 --agg max --dropout 0.4 --sep --heads 5 --hidden 36 --n-layers 5 # 94.77±0.31
python3 -u main.py NT amazon_ratings --runs 10 --max-epochs 1000 --early-stop-epochs 200 --lr 0.001 --agg max --dropout 0.4 --sep --heads 7 --hidden 23 --n-layers 4 # 49.43±0.62
python3 -u main.py NT minesweeper --runs 10 --max-epochs 1000 --early-stop-epochs 200 --lr 0.001 --agg sum --dropout 0.1 --add-self-loops --sep --heads 2 --hidden 15 --n-layers 5 # 93.92±0.59
python3 -u main.py NT tolokers --runs 10 --max-epochs 1000 --early-stop-epochs 200 --lr 0.001 --agg gatedsum --dropout 0.2 --remove-self-loops --sep --heads 4 --hidden 9 --n-layers 4 # 85.02±0.77
python3 -u main.py NT questions --runs 10 --max-epochs 1000 --early-stop-epochs 200 --lr 0.001 --agg gatedsum --dropout 0.3 --remove-self-loops --sep --heads 7 --hidden 27 --n-layers 1 # 77.99±1.00

# Homophilic
python3 -u main.py NT amazon-com --runs 10 --max-epochs 2500 --early-stop-epochs 500 --lr 0.001 --agg sum --dropout 0.4 --to-bidir --add-self-loops --heads 4 --hidden 17 --n-layers 5 # 92.61±0.63
python3 -u main.py NT amazon-photo --runs 10 --max-epochs 2500 --early-stop-epochs 500 --lr 0.001 --agg mean --dropout 0.6 --to-bidir --add-self-loops --sep --heads 7 --hidden 18 --n-layers 4 # 96.12±0.39
python3 -u main.py NT coauthor-cs --runs 10 --max-epochs 1000 --early-stop-epochs 200 --lr 0.001 --agg weightedmean --dropout 0.3 --to-bidir --heads 8 --hidden 41 --n-layers 2 # 96.07±0.32
python3 -u main.py NT coauthor-phy --runs 10 --max-epochs 2500 --early-stop-epochs 500 --lr 0.001 --agg weightedmean --dropout 0.1 --to-bidir --remove-self-loops --heads 2 --hidden 16 --n-layers 2 # 97.32±0.11
python3 -u main.py NT wikics --runs 10 --max-epochs 2500 --early-stop-epochs 500 --lr 0.001 --agg mean --dropout 0.2 --to-bidir --add-self-loops --heads 1 --hidden 38 --n-layers 3 # 80.04±0.61
