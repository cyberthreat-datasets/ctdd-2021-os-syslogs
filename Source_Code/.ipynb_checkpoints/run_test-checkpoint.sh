#!/bin/bash

# no_clients=(2 4 8 10)
no_layers=(4 6)
no_heads=(1 2)
# dropouts=(0.1 0.2 0.4 0.5)


# for head in "${no_heads[@]}"; do
#     for client in "${no_clients[@]}"; do
# #     python3 federated_train.py --num_classes=29 --epochs=10 --batch=512 --num_layers=4 --num_heads=2 --num_gpus=1 --dropout=0.2 --clients="$client" --rounds=10
#       python3 federated_train.py --num_classes=448 --epochs=5 --batch=2048 --num_layers=4 --num_heads="$head" --num_gpus=1  --dropout=0.2 --clients="$client" --rounds=10
#     done
# done

# for layer in "${no_layers[@]}"; do
# #     for head in "${no_heads[@]}"; do
# #         python3 train.py --num_classes=29 --epochs=10 --batch=512 --num_layers="$layer" --num_heads="$head" --num_gpus=2 --dropout=0.2
#     python3 train.py --num_classes=29 --num_layers="$layer" --num_heads=1 --num_gpus=1
# #     done
# done

for layer in "${no_layers[@]}"; do
    for head in "${no_heads[@]}"; do
#         python3 train.py --num_classes=448 --num_layers="$layer" --num_heads="$head" --num_gpus=1 --epochs= --batch_size=2048
        python3 federated_train.py --num_classes=449 --epochs=5 --batch=2048 --num_layers="$layer" --num_heads="$head" --num_gpus=1 --dropout=0.2 --clients=1 --rounds=10
#         python3 test.py
#     python3 train.py --num_classes=29 --num_layers="$layer" --num_heads=1 --num_gpus=1
    done
done