python pretrain_transformers.py \
    --output_dir="simplify_model_wiki" \
    --model_type=gpt2 \
    --model_name_or_path=sberbank-ai/rugpt3large_based_on_gpt2  \
    --do_train \
    --train_data_file=train.txt \
    --per_gpu_train_batch_size 1 \
    --line_by_line \
    --gradient_accumulation_steps 5 \
    --num_train_epochs 2 \
    --block_size 128 \
    --save_total_limit 2 \
    --overwrite_output_dir 
   
