python main.py \
    --task vehicle_classification_cnum23_dist0_skew0_seed0_missing_rate_0.3 \
    --model mifl \
    --algorithm multimodal.vehicle_classification.mifl \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 50  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --contrastive_weight 2 \
    --learning_rate 0.5 \
    --num_epochs 3 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --test_batch_size 128 \
    --gpu 1 \
    --wandb