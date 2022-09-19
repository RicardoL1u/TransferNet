export CUDA_VISIBLE_DEVICES=7
TRANSFORMERS_OFFLINE=1 python -m WebQSP.train --input_dir data/WebQSP --save_dir data/WebQSP/ckpt --batch_size 1
