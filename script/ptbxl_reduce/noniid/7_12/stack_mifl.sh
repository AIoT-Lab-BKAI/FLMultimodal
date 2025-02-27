python generate_fedtask.py \
    --benchmark ptbxl_reduce_classification \
    --dist 1 \
    --skew 0.5 \
    --num_clients 20 \
    --seed 0 \
    --missing_7_12

python main.py \
    --task ptbxl_reduce_classification_cnum20_dist1_skew0.5_seed0_missing_7_12 \
    --model mifl_contrastive2 \
    --algorithm multimodal.ptbxl_reduce_classification.mifl \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 50  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --contrastive_weight 3 \
    --learning_rate 0.5 \
    --num_epochs 3 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --test_batch_size 128 \
    --gpu 3 \
    --wandb

python main.py \
    --task ptbxl_reduce_classification_cnum20_dist1_skew0.5_seed0_missing_7_12 \
    --model mifl_contrastive2 \
    --algorithm multimodal.ptbxl_reduce_classification.mifl \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 50  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --contrastive_weight 4 \
    --learning_rate 0.5 \
    --num_epochs 3 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --test_batch_size 128 \
    --gpu 3 \
    --wandb

python main.py \
    --task ptbxl_reduce_classification_cnum20_dist1_skew0.5_seed0_missing_7_12 \
    --model mifl_contrastive2 \
    --algorithm multimodal.ptbxl_reduce_classification.mifl \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 50  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --contrastive_weight 5 \
    --learning_rate 0.5 \
    --num_epochs 3 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --test_batch_size 128 \
    --gpu 3 \
    --wandb
