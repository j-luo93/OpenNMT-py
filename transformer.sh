gpu=$1
data=$2
save=$3
#type=transformer
type=eat
CUDA_VISIBLE_DEVICES=$gpu python -m ipdb -c continue train.py -data data/$data -save_model ~/tmp/$save -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2096 -heads 8 -encoder_type $type -decoder_type $type -position_encoding -train_steps 20000 -max_generator_batches 2 -dropout 0.1 -batch_size 128 -batch_type sents -normalization sents -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -label_smoothing 0.1 -valid_steps 1000 -save_checkpoint_steps 1000 -world_size 1  -gpu_ranks 0
