import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_prefix', type=str)
    parser.add_argument('save_path', type=str)
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--model', default='transformer', type=str)
    parser.add_argument('--layers', default=6, type=int)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--transformer_ff', default=2096, type=int)
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--train_steps', default=20000, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--save_checkpoint_steps', default=1000, type=int)
    parser.add_argument('--pool_factor', type=int)
    parser.add_argument('--report_every', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--options', type=str)

    args = parser.parse_args()

    assert args.model in ['eat', 'transformer']
    if args.debug:
        args.layers = 2
        args.hidden_size = 32
        args.transformer_ff = 64
        args.heads = 4
        args.train_steps = 20
        args.valid_steps = 2
        args.save_checkpoint_steps = 4
        args.pool_factor = 4
        args.report_every = 4

    cmd = f'CUDA_VISIBLE_DEVICES=$gpu python -m ipdb -c continue train.py --data {args.data_prefix} --save_model {args.save_path}'
    cmd += f' --layers {args.layers}'
    cmd += f' --rnn_size {args.hidden_size} --word_vec_size {args.hidden_size}'
    cmd += f' --transformer_ff {args.transformer_ff}'
    cmd += f' -heads {args.heads}'
    cmd += f' --encoder_type {args.model} --decoder_type {args.model}'

    cmd += f' --position_encoding -max_generator_batches 2 --dropout 0.1 --batch_size 128 --batch_type sents --normalization sents --accum_count 2'
    cmd += f' --optim adam --adam_beta2 0.998 --decay_method noam --warmup_steps 8000 --learning_rate 2 --max_grad_norm 0 --param_init 0 --param_init_glorot --label_smoothing 0.1'
    cmd += f' --train_steps {args.train_steps} --valid_steps {args.valid_steps} --save_checkpoint_steps {args.save_checkpoint_steps}'

    if args.gpu:
        cmd += ' --world_size 1 --gpu_ranks 0'

    if args.pool_factor:
        cmd += f' --pool_factor {args.pool_factor}'

    if args.report_every:
        cmd += f' --report_every {args.report_every}'

    if args.options:
        cmd += f' {args.options}'

    print(f'Executing {cmd}')
    subprocess.call(cmd, shell=True)
