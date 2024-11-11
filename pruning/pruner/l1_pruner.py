import copy
from collections import OrderedDict

import torch.optim as optim

from smilelogging.utils import Timer

from .meta_pruner import MetaPruner
from .reinit_model import orth_dist, deconv_orth_dist


class Pruner(MetaPruner):
    def __init__(self, model, loader, logger, accelerator, args, passer=None):
        super(Pruner, self).__init__(model, loader, logger, accelerator, args, passer)

    def prune(self):
        self.prune_init()

        # Orthogonal regularization training
        if hasattr(self.args, "orth_reg_iter") and self.args.orth_reg_iter > 0:
            self.logger.info(
                "\n--> Start orthogonal regularization training.",
                unprefix=True,
                color="yellow",
            )
            self.model = self.__orth_reg_train(self.model)  # update self.model
            self.logger.info(
                "<-- End orthogonal regularization training.\n",
                unprefix=True,
                color="yellow",
            )

        if self.args.prune_with_hooks:
            assert self.args.wg in ["filter"], "When using --prune_with_hooks, --wg must be filter. Please check."
            from .prune_utils import register_hooks_for_pruning

            (
                self.pr,
                self.kept_filters,
                self.kept_channels,
                handles,
            ) = register_hooks_for_pruning(
                model=self.model,
                pr=self.pr,
                constrained_groups=self.constrained_groups,
                print_fn=self.logger.info,
                align_constrained=self.args.align_constrained,
            )

            self.logger.info(f"L1 pruning with hooks: Register hooks done")
            self.layer_builder.model_forward()
            self.logger.info(f"L1 pruning with hooks: model forward done")
            [h.remove() for h in handles]
            self.logger.info(f"L1 pruning with hooks: hooks removed")
            self.kept_wg = self.kept_filters
            self.pruned_wg = OrderedDict()  # Maintain API.
        else:
            # [TODO-@huanwangx-20240107: Deprecated, to remove.]
            self.pr, self.pruned_wg, self.kept_wg = self.get_kept_weight_groups(
                self.args.align_constrained
            )

        self.model = self.prune_and_build_new_model()
        return self.model

    def __orth_reg_train(self, model):
        # TODO-@mst: other optims
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.args.lr_prune,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        acc1 = acc5 = 0
        epoch = -1
        timer = Timer(self.args.orth_reg_iter / self.args.print_interval)
        self.total_iter = -1
        self.prune_state = "orth_reg"
        while True:
            epoch += 1
            for _, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                self.total_iter += 1
                total_iter = self.total_iter

                if total_iter % self.args.print_interval == 0:
                    self.logger.info("")
                    self.logger.info(
                        "Iter = %d [prune_state = %s, method = %s] "
                        % (total_iter, self.prune_state, self.args.method)
                        + "-" * 40
                    )

                # forward
                model.train()
                y_ = model(inputs)

                # normal training forward
                loss = self.criterion(y_, targets)
                logtmp = f"loss_cls {loss:.4f}"

                # Orth reg
                loss_orth_reg = 0
                for name, module in model.named_modules():
                    if isinstance(module, self.learnable_types):
                        if self.args.orth_reg_method in ["CVPR20"]:
                            if (
                                self.learnable_layers[name].index != 0
                            ):  # per the CVPR20 paper, do not reg the 1st conv
                                shape = self.learnable_layers[name].size
                                if len(shape) == 2 or shape[-1] == 1:  # FC and 1x1 conv
                                    loss_orth_reg += orth_dist(module.weight)
                                else:
                                    loss_orth_reg += deconv_orth_dist(module.weight)
                        elif self.args.orth_reg_method in ["CVPR17"]:
                            loss_orth_reg += orth_dist(module.weight)
                        else:
                            raise NotImplementedError
                loss += loss_orth_reg * self.args.lw_orth_reg

                # print loss
                if self.total_iter % self.args.print_interval == 0:
                    logtmp += f" loss_orth_reg (*{self.args.lw_orth_reg}) {loss_orth_reg:.10f} Iter {self.total_iter}"
                    self.logger.info(logtmp)
                    self.logger.info(f"predicted_finish_time of orth_reg: {timer()}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # test
                if total_iter % self.args.test_interval == 0:
                    acc1, acc5, *_ = self.test(model)
                    self.accprint(
                        "Acc1 = %.4f Acc5 = %.4f Iter = %d (after update) [prune_state = %s, method = %s]"
                        % (acc1, acc5, total_iter, self.prune_state, self.args.method)
                    )

                # save model (save model before a batch starts)
                if total_iter % self.args.save_interval == 0:
                    self.__save_model(model, optimizer, acc1, acc5)
                    self.logger.info(
                        "Periodically save model done. Iter = {}".format(total_iter)
                    )

                # return
                if total_iter > self.args.orth_reg_iter:
                    return copy.deepcopy(model)

    def __save_model(self, model, optimizer, acc1=0, acc5=0, mark=""):
        state = {
            "iter": self.total_iter,
            "arch": self.args.arch,
            "model": model,
            "state_dict": model.state_dict(),
            "acc1": acc1,
            "acc5": acc5,
            "optimizer": optimizer.state_dict(),
            "ExpID": self.logger.ExpID,
        }
        self.save(state, is_best=False, mark=mark)
