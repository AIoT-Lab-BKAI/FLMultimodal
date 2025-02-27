# python main.py \
#     --task edf_classification_cnum10_dist0_skew0_seed0_missing_1_5 \
#     --model mifl \
#     --algorithm multimodal.edf_classification.mifl \
#     --sample full \
#     --aggregate other \
#     --num_rounds 1000 \
#     --early_stop 50  \
#     --proportion 1.0 \
#     --lr_scheduler 0 \
#     --seed 1234 \
#     --fedmsplit_prox_lambda 0.01 \
#     --learning_rate 0.01 \
#     --contrastive_weight 5 \
#     --num_epochs 3 \
#     --learning_rate_decay 1.0 \
#     --batch_size 256 \
#     --test_batch_size 256 \
#     --gpu 1 \
#     --wandb
# python main.py \
#     --task edf_classification_cnum20_dist0_skew0_seed0_missing_1_5 \
#     --model mifl \
#     --algorithm multimodal.edf_classification.mifl \
#     --sample full \
#     --aggregate other \
#     --num_rounds 1000 \
#     --early_stop 50  \
#     --proportion 1.0 \
#     --lr_scheduler 0 \
#     --seed 1234 \
#     --fedmsplit_prox_lambda 0.01 \
#     --learning_rate 0.01 \
#     --contrastive_weight 5 \
#     --num_epochs 3 \
#     --learning_rate_decay 1.0 \
#     --batch_size 256 \
#     --test_batch_size 256 \
#     --gpu 1 \
#     --wandb
python main.py \
    --task edf_classification_cnum40_dist0_skew0_seed0_missing_1_5 \
    --model mifl \
    --algorithm multimodal.edf_classification.mifl \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 50  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.01 \
    --contrastive_weight 5 \
    --num_epochs 3 \
    --learning_rate_decay 1.0 \
    --batch_size 256 \
    --test_batch_size 256 \
    --gpu 0 \
    --wandb
python main.py \
    --task edf_classification_cnum50_dist0_skew0_seed0_missing_1_5 \
    --model mifl \
    --algorithm multimodal.edf_classification.mifl \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 50  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.01 \
    --contrastive_weight 5 \
    --num_epochs 3 \
    --learning_rate_decay 1.0 \
    --batch_size 256 \
    --test_batch_size 256 \
    --gpu 0 \
    --wandb