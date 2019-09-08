"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from copy import deepcopy
import torch
import traceback

import onmt.utils
from onmt.utils.logging import logger
from onmt.utils import aeq
from onmt.inputters.inputter import CrosslingualDatasetIter


def build_trainer(opt, device_id, model, fields, optim, model_saver=None, aux_fields=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, tgt_field, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt)
    crosslingual_train_loss = None
    aux_train_loss = None
    if opt.crosslingual:
        if opt.crosslingual == 'old':
            # FIXME This is actually wrong. The vocab here is right, but the dataset is built on the wrong (for aux task) vocab.
            crosslingual_tgt_field = tgt_field
        else:
            # NOTE Use the src side, since I'm only predicting the source side (EAT side).
            crosslingual_tgt_field = dict(aux_fields)['src'].base_field
        crosslingual_train_loss = onmt.utils.loss.build_loss_compute(
            model, crosslingual_tgt_field, opt, crosslingual=opt.crosslingual)
        aux_tgt_field = dict(aux_fields)["tgt"].base_field
        aux_train_loss = onmt.utils.loss.build_loss_compute(model, aux_tgt_field, opt, crosslingual='')

    trainer = onmt.Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps,
                           crosslingual_train_loss=crosslingual_train_loss,
                           aux_train_loss=aux_train_loss,
                           almt_reg_hyper=opt.almt_reg_hyper,
                           crosslingual=opt.crosslingual)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0], crosslingual_train_loss=None, aux_train_loss=None, almt_reg_hyper=0.0, crosslingual=''):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.almt_reg_hyper = almt_reg_hyper
        self.crosslingual = crosslingual

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

        # Set crosslingual and aux training losses.
        self.crosslingual_train_loss = crosslingual_train_loss
        self.aux_train_loss = aux_train_loss

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            # print(batch.src[0].max(), batch.tgt.max())
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iters=None,
              valid_steps=10000,
              cl_valid_iter=None):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iters is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = {'base': onmt.utils.Statistics()}
        report_stats = {'base': onmt.utils.Statistics()}
        if isinstance(train_iter, CrosslingualDatasetIter):
            total_stats['aux'] = onmt.utils.Statistics()
            report_stats['aux'] = onmt.utils.Statistics()
            total_stats['crosslingual'] = onmt.utils.Statistics()
            report_stats['crosslingual'] = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats['base'].start_time)

        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            step = self.optim.training_step
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d"
                            % (self.gpu_rank, i + 1, len(batches)))

            if self.n_gpu > 1:
                normalization = sum(onmt.utils.distributed
                                    .all_gather_list
                                    (normalization))

            self._gradient_accumulation(
                batches, normalization, total_stats,
                report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if valid_iters is not None and step % valid_steps == 0:

                def evaluate(dataset_iter, try_earlystop=False):
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: validate step %d'
                                    % (self.gpu_rank, step))
                    valid_stats = self.validate(
                        dataset_iter, moving_average=self.moving_average)
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: gather valid stat \
                                    step %d' % (self.gpu_rank, step))
                    valid_stats = self._maybe_gather_stats(valid_stats)
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: report stat step %d'
                                    % (self.gpu_rank, step))
                    self._report_step(self.optim.learning_rate(),
                                      step, valid_stats=valid_stats)
                    # Run patience mechanism
                    if self.earlystopper is not None and try_earlystop:
                        self.earlystopper(valid_stats, step)
                        # If the patience has reached the limit, stop training
                        return self.earlystopper.has_stopped()
                    return False

                stopped = True
                for valid_iter in valid_iters:
                    this_stopped = evaluate(valid_iter, try_earlystop=True)
                    stopped = stopped and this_stopped
                if cl_valid_iter:
                    evaluate(cl_valid_iter, try_earlystop=False)
                if stopped:
                    break

            if (self.model_saver is not None
                and (save_checkpoint_steps != 0
                     and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        if moving_average:
            valid_model = deepcopy(self.model)
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                param.data = avg.data
        else:
            valid_model = self.model

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                    else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                task = self._get_task(batch)
                outputs, attns = valid_model(src, tgt, src_lengths, task=task)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)

        if moving_average:
            del valid_model
        else:
            # Set model back to training mode.
            valid_model.train()

        return stats

    def _get_task(self, batch):
        try:
            return batch.task
        except AttributeError:
            return None

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            task = self._get_task(batch)
            if task.category == 'lm':
                target_size = batch.tgt_event.size(0)
            else:
                target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            if self.crosslingual == 'lm':
                src_attr = 'src_event'
            elif self.crosslingual == 'old':
                src_attr = 'src_old' if batch.eat_format == 'combined' else 'src'
            elif batch.eat_format in ['old', 'new']:
                src_attr = 'src'
            else:
                src_attr = 'src_old'
            src, src_lengths = getattr(batch, src_attr) if isinstance(getattr(batch, src_attr), tuple) \
                else (getattr(batch, src_attr), None)

            r_stats = report_stats[task.name]
            t_stats = total_stats[task.name]
            if src_lengths is not None:
                r_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt

            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()

                # 2.5 Prepare for crosslingual mode if needed.
                outputs, attns = self.model(src, tgt, src_lengths, bptt=bptt, task=task)
                bptt = True

                # 2.9 Get appropriate loss function.
                if task.name == 'crosslingual':
                    loss_func = self.crosslingual_train_loss
                elif task.name == 'aux':
                    loss_func = self.aux_train_loss
                else:
                    loss_func = self.train_loss

                # 3. Compute loss.
                try:
                    def loss_call(batch, outputs):
                        loss, batch_stats = loss_func(
                            batch,
                            outputs,
                            attns,
                            normalization=normalization,
                            shard_size=self.shard_size,
                            trunc_start=j,
                            trunc_size=trunc_size)
                        return loss, batch_stats

                    if task.category == 'lm':
                        agent_preds = outputs['agent']
                        theme_preds = outputs['theme']

                        batch.agent_preds = agent_preds
                        batch.theme_preds = theme_preds
                        batch.tgt_backup = batch.tgt

                        def update_loss_and_stats(tgt_attr, preds, loss, batch_stats):
                            batch.tgt = getattr(batch, tgt_attr)
                            loss_call, batch_stats_call = loss_call(batch, preds)
                            loss += loss_call
                            batch_stats.update(batch_stats_call)

                        loss = 0.0
                        batch_stats = onmt.utils.Statistics()
                        update_loss_and_stats('tgt_agent', agent_preds, loss, batch_stats)
                        update_loss_and_stats('tgt_agent_mod', agent_preds, loss, batch_stats)
                        update_loss_and_stats('tgt_theme', theme_preds, loss, batch_stats)
                        update_loss_and_stats('tgt_theme_mod', theme_preds, loss, batch_stats)
                    else:
                        loss, batch_stats = loss_call(batch)

                    if task.name == 'crosslingual' and self.almt_reg_hyper > 0.0:
                        weight = self.model.encoder.embeddings.almt_layers['mapping'].weight
                        reg_loss = weight
                        d1, d2 = weight.shape
                        aeq(d1, d2)
                        eye = torch.eye(d1).to(weight.device)
                        reg_loss = ((weight @ weight.T - eye) ** 2).sum()
                        reg_loss.backward()

                    if loss is not None:
                        self.optim.backward(loss)

                    t_stats.update(batch_stats)
                    r_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d",
                                self.optim.training_step, k)

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        ret = dict()
        if self.report_manager is not None:
            for k, stats in report_stats.items():
                ret[k] = self.report_manager.report_training(
                    step, num_steps, learning_rate, stats,
                    multigpu=self.n_gpu > 1, task_name=k)
        return ret

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)
