python main.py \
    --task edf_classification_cnum30_dist0_skew0_seed0_missing_1_5 \
    --model fedavg \
    --algorithm multimodal.edf_classification.fedavg \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 50  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.5 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 256 \
    --test_batch_size 256 \
    --gpu 0 
    # \
    # --wandb