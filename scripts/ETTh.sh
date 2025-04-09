pred_l=(96 192 336 720)

for i in ${pred_l[@]}
do
    python -u run.py \
        --model mcanet \
        --mode mean \
        --data ETTh1 \
        --features M \
        --freq h \
        --conv_kernel 12 16 24 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $i \
        --top_k 3 
done

for i in ${pred_l[@]}
do
    python -u run.py \
        --model mcanet \
        --mode mean \
        --data ETTh2 \
        --features M \
        --freq h \
        --conv_kernel 12 16  24\
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $i \
        --top_k 3 
done