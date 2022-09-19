export CUDA_VISIBLE_DEVICES=7
TRANSFORMERS_OFFLINE=1 python -m WebQSP.train \
    --input_dir data/AnonyQA \
    --save_dir data/AnonyQA/ckpt \
    --kg_name essentail \
    --batch_size 16 
