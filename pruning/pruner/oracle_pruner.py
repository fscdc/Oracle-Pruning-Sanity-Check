import torch.nn as nn
import copy
import numpy as np
from .meta_pruner import MetaPruner
import itertools
from collections import OrderedDict
import random
from .prune_utils import replace_module
import torch

class Pruner(MetaPruner):
    def __init__(self, model, loader, logger, accelerator, args, passer=None):
        super(Pruner, self).__init__(model, loader, logger, accelerator, args, passer)
        

        self.test_trainset = lambda net: passer.train(
            net
        )
        self.test_trainset_raw = lambda net: passer.train_raw(
            net
        )


        self.retrain = lambda net: passer.retrain(
            net,
            loader.train_loader,
            loader.train_sampler,
            passer.criterion,
            args,
        )

        self.batch_oracle = args.batch_oracle
        self.random = args.random
        self.save_combination = args.save_combination
        self.arch = args.arch
        self.dataset = args.dataset
        self.lr_ft = args.lr_ft
        self.num_missions = args.num_missions
        self.num_batches = args.num_batches

        self.num_channels, self.input_height, self.input_width = passer.num_channels, passer.input_height, passer.input_width

    def prune(self):
        self.prune_init()

        if self.args.prune_with_hooks:
            assert self.args.wg in ["filter", "channel"]
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

        self._get_kept_wg_oracle()

        if self.args.reinit:
            raise NotImplementedError

        return self.model

    def one_prune_iter(self, pair, cnt, pruned_loss, n_pairs):
        cnt[0] += 1
        cnt_m = 0
        # model = copy.deepcopy()
        for name, m in self.model.named_modules():
            if name in self.pr:
                n_wg = self.learnable_layers[name].shape[0]
                if self.pr[name]:
                    self.pruned_wg[name] = pair[cnt_m]
                    self.kept_wg[name] = [
                        x for x in range(n_wg) if x not in self.pruned_wg[name]
                    ]
                    cnt_m += 1
                else:
                    self.pruned_wg[name] = []
                    self.kept_wg[name] = list(range(n_wg))
        self.kept_filters = self.kept_wg
        model = self.prune_and_build_new_model()
        self.logger.info(f"==> Pruned model: {model}")
        if self.dataset == "imagenet":
            *_, pruned_train_loss = self.test_trainset_raw(model)
        else:
            *_, pruned_train_loss = self.test_trainset(model)
        pruned_loss.append(pruned_train_loss)
        self.logger.info("")
        self.logger.info("[%d/%d] pruned_index_pair {%s}" % (cnt[0], n_pairs, pair))
        self.logger.info("[%d/%d] pruned_train_loss %.6f" % (cnt[0], n_pairs, pruned_train_loss))
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(model, (self.num_channels, self.input_height, self.input_width), as_strings=True, print_per_layer_stat=False, verbose=False)
        self.logger.info(f"Pruned model: macs: {macs}, params: {params}")

        # retrain the pruned model (Notice-@fengsicheng: the code skip by the prarmeter `ft_in_oracle_pruning`)
        # if self.args.ft_in_oracle_pruning:
        best, last5 = self.retrain(
            model
        )  # it will return the acc/loss of the best model during finetune
        self.logger.info(
            "[%d/%d] final_train_loss %.6f final_test_loss %.6f final_test_acc %.6f"
            % (cnt[0], n_pairs, best[1], best[2], best[0])
        )
        self.logger.info(
            "[%d/%d] last5_train_loss %.6f last5_test_loss %.6f last5_test_acc %.6f (mean)"
            % (cnt[0], n_pairs, last5[2], last5[4], last5[0])
        )
        self.logger.info(
            "[%d/%d] last5_train_loss %.6f last5_test_loss %.6f last5_test_acc %.6f (std)"
            % (cnt[0], n_pairs, last5[3], last5[5], last5[1])
        )

    def _get_kept_wg_oracle(self):
        # get all the possible wg combinations to prune

        combinations_layer = []  # pruned index combination of each layer
        pruned_index_pairs = []
        if self.random == "no":
            for name, module in self.model.named_modules():
                if self.pr.get(name):
                    if self.args.wg == "filter":
                        n_wg = self.learnable_layers[name].shape[0]
                    elif self.args.wg == "channel":
                        n_wg = self.learnable_layers[name].shape[1]
                    elif self.args.wg == "weight":
                        n_wg = np.prod(self.learnable_layers[name].shape)
                    n_pruned = int(n_wg * self.pr[name])
                    combinations_layer.append(
                        list(itertools.combinations(range(n_wg), n_pruned))
                    )

            # orable pruning
            pruned_index_pairs = list(itertools.product(*combinations_layer))
        else:
            if self.save_combination == "yes":
                # record all (n_wg, n_pruned)
                for name, module in self.model.named_modules():
                    if self.pr.get(name):
                        if self.args.wg == "filter":
                            n_wg = self.learnable_layers[name].shape[0]
                        elif self.args.wg == "channel":
                            n_wg = self.learnable_layers[name].shape[1]
                        elif self.args.wg == "weight":
                            n_wg = np.prod(self.learnable_layers[name].shape)
                        n_pruned = int(n_wg * self.pr[name])
                        combinations_layer.append((n_wg, n_pruned))

                pruned_index_pairs = set()
                while len(pruned_index_pairs) < self.num_missions:
                    current_combination = []
                    for n_wg, n_pruned in combinations_layer:
                        pruned_indices = tuple(sorted(random.sample(range(n_wg), n_pruned)))
                        current_combination.append(pruned_indices)
                    current_combination = tuple(current_combination)
                    # self.logger.info(f"==> Current combination: {current_combination}")
                    if current_combination not in pruned_index_pairs:
                        pruned_index_pairs.add(current_combination)

                # Convert set to list for further processing if needed
                pruned_index_pairs = list(pruned_index_pairs)

                import pickle
                import os

                if not os.path.exists('./Combinations'):
                    os.makedirs('./Combinations')
                file_path = os.path.expanduser(f'./Combinations/{self.arch}-{self.dataset}-{self.lr_ft}-combination.pkl')
                with open(file_path, 'wb') as f:
                    pickle.dump(pruned_index_pairs, f)

                self.logger.info("==> Pruned index pairs have been saved to pruned_index_pairs.pkl")

                import sys
                sys.exit("exit after saving pruned index pairs")

            elif self.save_combination == "no":
                # Load pruned index pairs from file
                import pickle
                import os
                
                file_path = os.path.expanduser(f'./Combinations/{self.arch}-{self.dataset}-{self.lr_ft}-combination.pkl')
                with open(file_path, 'rb') as f:
                    pruned_index_pairs = pickle.load(f)

                self.logger.info("==> Pruned index pairs have been loaded from pruned_index_pairs.pkl")

            else:
                raise ValueError("Invalid value for save_combination")

        n_pairs = len(pruned_index_pairs)
        n_pairs_perbatch = int(n_pairs / self.num_batches)
        print("==> Start oracle pruning: %d pairs of pruned index to ablate" % n_pairs)

        if self.num_batches == 8:
            pruned_loss_batch0, cnt_batch0 = [], [0]
            pruned_loss_batch1, cnt_batch1 = [], [n_pairs_perbatch]
            pruned_loss_batch2, cnt_batch2 = [], [2*n_pairs_perbatch]
            pruned_loss_batch3, cnt_batch3 = [], [3*n_pairs_perbatch]
            pruned_loss_batch4, cnt_batch4 = [], [4*n_pairs_perbatch]
            pruned_loss_batch5, cnt_batch5 = [], [5*n_pairs_perbatch]
            pruned_loss_batch6, cnt_batch6 = [], [6*n_pairs_perbatch]
            pruned_loss_batch7, cnt_batch7 = [], [7*n_pairs_perbatch]

            if self.batch_oracle == 0:
                for pair in pruned_index_pairs[0:n_pairs_perbatch]:
                    self.one_prune_iter(pair, cnt_batch0, pruned_loss_batch0, n_pairs)           
            elif self.batch_oracle == 1:
                for pair in pruned_index_pairs[n_pairs_perbatch:2*n_pairs_perbatch]:
                    self.one_prune_iter(pair, cnt_batch1, pruned_loss_batch1, n_pairs)
            elif self.batch_oracle == 2:
                for pair in pruned_index_pairs[2*n_pairs_perbatch:3*n_pairs_perbatch]:
                    self.one_prune_iter(pair, cnt_batch2, pruned_loss_batch2, n_pairs)
            elif self.batch_oracle == 3:
                for pair in pruned_index_pairs[3*n_pairs_perbatch:4*n_pairs_perbatch]:
                    self.one_prune_iter(pair, cnt_batch3, pruned_loss_batch3, n_pairs)
            elif self.batch_oracle == 4:
                for pair in pruned_index_pairs[4*n_pairs_perbatch:5*n_pairs_perbatch]:
                    self.one_prune_iter(pair, cnt_batch4, pruned_loss_batch4, n_pairs)
            elif self.batch_oracle == 5:
                for pair in pruned_index_pairs[5*n_pairs_perbatch:6*n_pairs_perbatch]:
                    self.one_prune_iter(pair, cnt_batch5, pruned_loss_batch5, n_pairs)
            elif self.batch_oracle == 6:
                for pair in pruned_index_pairs[6*n_pairs_perbatch:7*n_pairs_perbatch]:
                    self.one_prune_iter(pair, cnt_batch6, pruned_loss_batch6, n_pairs)
            elif self.batch_oracle == 7:
                for pair in pruned_index_pairs[7*n_pairs_perbatch:]:
                    self.one_prune_iter(pair, cnt_batch7, pruned_loss_batch7, n_pairs)
        else:
            ValueError("Do NOT support other batches number yet!")

        import sys
        sys.exit("exit after oracle pruning")