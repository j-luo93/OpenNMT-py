#!/usr/bin/env python
"""Train models."""
import os
import signal
import torch

import onmt.opts as opts
import onmt.utils.distributed

from onmt.utils.misc import set_random_seed
from onmt.utils import aeq
from onmt.utils.logging import init_logger, logger
from onmt.modules.crosslingual import Eat2PlainCrosslingualTask, Eat2PlainMonoTask, Eat2PlainAuxMonoTask, EatLMMonoTask, EatLMCrosslingualTask
from onmt.train_single import main as single_main
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple, build_crosslingual_dataset_iter

from itertools import cycle


def main(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    # Load checkpoint if we resume from a previous training.
    aux_vocab = None
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
        if opt.crosslingual:
            aux_vocab = checkpoint['aux_vocab']
    elif opt.crosslingual:
        assert opt.crosslingual in ['old', 'lm']
        vocab = torch.load(opt.data + '.vocab.pt')
        aux_vocab = torch.load(opt.aux_train_data + '.vocab.pt')
    else:
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    def get_fields(vocab):
        if old_style_vocab(vocab):
            return load_old_vocab(
                vocab, opt.model_type, dynamic_dict=opt.copy_attn)
        else:
            return vocab
    fields = get_fields(vocab)
    aux_fields = None
    if opt.crosslingual:
        aux_fields = get_fields(aux_vocab)

    if opt.crosslingual:
        if opt.crosslingual == 'old':
            aeq(len(opt.eat_formats), 3)
            fields_info = [('train', fields, 'data', Eat2PlainMonoTask, 'base', opt.eat_formats[0]),
                           ('train', aux_fields, 'aux_train_data', Eat2PlainAuxMonoTask, 'aux', opt.eat_formats[1]),
                           ('train', aux_fields, 'aux_train_data', Eat2PlainCrosslingualTask, 'crosslingual', opt.eat_format[2])]
        else:
            aeq(len(opt.eat_formats), 4)
            fields_info = [('train', fields, 'data', Eat2PlainMonoTask, 'base', opt.eat_formats[0]),
                           ('train', fields, 'data', EatLMMonoTask, 'lm', opt.eat_formats[1]),
                           ('train', aux_fields, 'aux_train_data', Eat2PlainAuxMonoTask, 'aux', opt.eat_formats[2]),
                           ('train', aux_fields, 'aux_train_data', EatLMCrosslingualTask, 'crosslingual', opt.eat_formats[3])]
        train_iter = build_crosslingual_dataset_iter(fields_info, opt)
    elif len(opt.data_ids) > 1:
        train_shards = []
        for train_id in opt.data_ids:
            shard_base = "train_" + train_id
            train_shards.append(shard_base)
        train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
    else:
        if opt.data_ids[0] is not None:
            shard_base = "train_" + opt.data_ids[0]
        else:
            shard_base = "train"
        train_iter = build_dataset_iter(shard_base, fields, opt)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        queues = []
        mp = torch.multiprocessing.get_context('spawn')
        semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            q = mp.Queue(opt.queue_size)
            queues += [q]
            procs.append(mp.Process(target=run, args=(
                opt, device_id, error_queue, q, semaphore), daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        producer = mp.Process(target=batch_producer,
                              args=(train_iter, queues, semaphore, opt,),
                              daemon=True)
        producer.start()
        error_handler.add_child(producer.pid)

        for p in procs:
            p.join()
        producer.terminate()

    else:
        device_id = 0 if nb_gpu == 1 else -1
        # NOTE Only pass train_iter in my crosslingual mode.
        train_iter = train_iter if opt.crosslingual else None
        passed_fields = {'main': fields, 'crosslingual': aux_fields} if opt.crosslingual else None
        single_main(opt, device_id, train_iter=train_iter, passed_fields=passed_fields)


def batch_producer(generator_to_serve, queues, semaphore, opt):
    init_logger(opt.log_file)
    set_random_seed(opt.seed, False)
    # generator_to_serve = iter(generator_to_serve)

    def pred(x):
        """
        Filters batches that belong only
        to gpu_ranks of current node
        """
        for rank in opt.gpu_ranks:
            if x[0] % opt.world_size == rank:
                return True

    generator_to_serve = filter(
        pred, enumerate(generator_to_serve))

    def next_batch(device_id):
        new_batch = next(generator_to_serve)
        semaphore.acquire()
        return new_batch[1]

    b = next_batch(0)

    for device_id, q in cycle(enumerate(queues)):
        b.dataset = None
        if isinstance(b.src, tuple):
            b.src = tuple([_.to(torch.device(device_id))
                           for _ in b.src])
        else:
            b.src = b.src.to(torch.device(device_id))
        b.tgt = b.tgt.to(torch.device(device_id))
        b.indices = b.indices.to(torch.device(device_id))
        b.alignment = b.alignment.to(torch.device(device_id)) \
            if hasattr(b, 'alignment') else None
        b.src_map = b.src_map.to(torch.device(device_id)) \
            if hasattr(b, 'src_map') else None

        # hack to dodge unpicklable `dict_keys`
        b.fields = list(b.fields)
        q.put(b)
        b = next_batch(device_id)


def run(opt, device_id, error_queue, batch_queue, semaphore):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id, batch_queue, semaphore)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
