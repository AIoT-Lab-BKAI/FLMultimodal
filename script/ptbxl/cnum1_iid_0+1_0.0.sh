python main.py \
    --task ptbxl_cnum1_iid_0+1 \
    --model inception1d \
    --algorithm mm_ptbxl \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --proportion 1.0 \
    --num_epochs 1 \
    --learning_rate 0.5 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --gpu 1 \
    --seed 1234 \
    --test_batch_size 128 \
    --contrastive_weight 0.0 \
    --temperature 0.0