GPU=0
RUN=10

# heterophilic datasets
python main.py --dataset roman-empire --hidden_channels 32 --local_epochs 100 --global_epochs 2500 --lr 0.001 --runs $RUN --local_layers 5 --global_layers 2 --weight_decay 0.0 --dropout 0.2 --global_dropout 0.5 --in_dropout 0.15 --num_heads 6 --device $GPU --save_model --beta 0.5 --agg sum # 92.90±0.30
python main.py --dataset amazon-ratings --hidden_channels 45 --local_epochs 0 --global_epochs 2500 --lr 0.001 --runs $RUN --local_layers 5 --global_layers 0 --weight_decay 0.0 --dropout 0.2 --in_dropout 0.1 --num_heads 8 --device $GPU --save_model --agg max --global_dropout 0.1 # 54.55±0.39
python main.py --dataset minesweeper --hidden_channels 53 --local_epochs 100 --global_epochs 2000 --lr 0.001 --runs $RUN --local_layers 5 --global_layers 3 --weight_decay 0.0 --dropout 0.1 --in_dropout 0.2 --num_heads 1 --metric rocauc --device $GPU --save_model --agg sum --pre_ln # 97.92±0.28
python main.py --dataset tolokers --hidden_channels 30 --local_epochs 100 --global_epochs 800 --lr 0.001 --runs $RUN --local_layers 5 --global_layers 4 --weight_decay 0.0 --dropout 0.2 --in_dropout 0.2 --num_heads 2 --metric rocauc --device $GPU --save_model --beta 0.1 --agg gatedsum --pre_ln # 85.91±0.83
python main.py --dataset questions --hidden_channels 27 --local_epochs 200 --global_epochs 1500 --lr 0.001 --runs $RUN --local_layers 5 --global_layers 2 --weight_decay 0.0 --dropout 0.5 --global_dropout 0.4 --num_heads 2 --metric rocauc --device $GPU --in_dropout 0.6 --save_model --beta 0.4 --agg sum --pre_ln # 79.36±0.69

# homophilic datasets
python main.py --dataset amazon-computer --hidden_channels 24 --local_epochs 200 --global_epochs 1000 --lr 0.001 --runs $RUN --local_layers 2 --global_layers 5 --weight_decay 5e-5 --dropout 0.3 --in_dropout 0.5 --num_heads 7 --device $GPU --save_model --agg sum --pre_ln --global_dropout 0.5 # 93.87±0.23
python main.py --dataset amazon-photo --hidden_channels 18 --local_epochs 200 --global_epochs 1000 --lr 0.001 --runs $RUN --local_layers 4 --global_layers 2 --weight_decay 5e-5 --dropout 0.6 --in_dropout 0.2 --num_heads 7 --device $GPU --save_model --agg mean --pre_ln # 96.67±0.17
python main.py --dataset coauthor-cs --hidden_channels 41 --local_epochs 100 --global_epochs 1500 --lr 0.001 --runs $RUN --local_layers 2 --global_layers 2 --weight_decay 5e-4 --dropout 0.3 --in_dropout 0.1 --num_heads 8 --device $GPU --save_model --agg weightedmean --pre_ln # 96.09±0.11
python main.py --dataset coauthor-physics --hidden_channels 16 --local_epochs 100 --global_epochs 1500 --lr 0.001 --runs $RUN --local_layers 2 --global_layers 4 --weight_decay 5e-4 --dropout 0.2 --in_dropout 0.5 --num_heads 2 --device $GPU --save_model --agg weightedmean --pre_ln # 97.33±0.09
python main.py --dataset wikics --hidden_channels 38 --local_epochs 0 --global_epochs 1000 --lr 0.001 --runs $RUN --local_layers 3 --global_layers 0 --weight_decay 0.0 --dropout 0.4 --in_dropout 0.2 --num_heads 1 --device $GPU --save_model --agg mean --global_dropout 0.2 # 80.17±0.71
