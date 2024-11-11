import time
from smilelogging import red, blue, yellow
from smilelogging.utils import (
    parse_prune_ratio_vgg,
    strlist_to_list,
    strdict_to_dict,
    check_path,
    isfloat,
)


def add_args(parser):
    parser.add_argument(
        f"--prune_method",
        type=str,
        default="",
        help='pruning method name; default is "", implying the original training without any pruning',
    )
    parser.add_argument(f"--pruner", type=str, default="", help="pruner name")
    parser.add_argument(
        "--stage_pr",
        "--layerwise_pr",
        dest="stage_pr",
        type=str,
        default="",
        help="to assign layer-wise pruning ratio",
    )
    parser.add_argument(
        "--skip_layers",
        type=str,
        default="",
        help="layer id to skip_layers when pruning",
    )
    parser.add_argument(
        "--compare_mode",
        type=str,
        default="local",
        choices=["global", "local"],
        help="global pruning sorts weights from different learnable_layers; local sorts weight within a layer",
    )
    parser.add_argument(
        "--index_layer",
        type=str,
        default="name_matching",
        help="the rule to index learnable_layers in a network by its name; used in designating pruning ratio",
    )
    parser.add_argument(
        "--align_constrained",
        action="store_true",
        help="make constrained learnable_layers have the same pruned indices",
    )
    parser.add_argument(
        "--reinit_layers",
        type=str,
        default="",
        help="learnable_layers to reinit (not inherit pretrained weights)",
    )
    parser.add_argument(
        "--lr_ft", type=str, default="0:0.01,30:0.001,60:0.0001,75:0.00001"
    )
    parser.add_argument(
        "--wg",
        type=str,
        default="filter",
        choices=["filter", "maskfilter", "channel", "weight"],
    )
    parser.add_argument(
        "--pick_pruned",
        type=str,
        default="min",
        help="the criterion to select weights to prune",
    )
    parser.add_argument(
        "--reinit",
        type=str,
        default="",
        choices=[
            "",
            "default",
            "pth_reset",
            "xavier_uniform",
            "kaiming_normal",
            "orth",
            "exact_isometry_from_scratch",
            "exact_isometry_based_on_existing",
            "exact_isometry_based_on_existing_delta",
            "approximate_isometry",
            "AI",
            "approximate_isometry_from_scratch",
            "AI_scratch",
            "data_dependent",
        ],
        help="before finetuning, the pruned model will be reinited",
    )
    parser.add_argument("--reinit_scale", type=float, default=1.0)
    parser.add_argument(
        "--base_pr_model",
        type=str,
        default="",
        help="the model that provides layer-wise pr",
    )
    parser.add_argument(
        "--inherit_pruned",
        type=str,
        default="",
        choices=["", "index", "pr"],
        help="when --base_pr_model is provided, we can choose to "
        "inherit the pruned index or only the pruning ratio (pr)",
    )
    parser.add_argument("--oracle_pruning", action="store_true")
    parser.add_argument("--ft_in_oracle_pruning", action="store_true")
    parser.add_argument(
        "--last_n_epoch",
        type=int,
        default=5,
        help="in correlation analysis, collect the last_n_epoch loss and average them",
    )
    parser.add_argument(
        "--lr_AI", type=float, default=0.001, help="lr in approximate_isometry_optimize"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="",
        help="print some intermediate results like gradients",
    )
    parser.add_argument(
        "--prune_schedule",
        type=str,
        default="simu",
        help="scheme to decide how to schedule the pruning",
    )
    parser.add_argument(
        "--prune_with_hooks",
        action="store_true",
        help="new filter pruning implementation",
    )
    parser.add_argument(
        "--not_prune_cnst",
        action="store_true",
        help="not prune constrained learnable_layers",
    )
    return parser


def check_args(args):
    args.skip_layers = strlist_to_list(args.skip_layers, str)
    args.verbose = strlist_to_list(args.verbose, str)

    if args.wg == "weight" and args.align_constrained:
        print(
            red(
                "Error: --wg weight and --align_constrained used at the same time. "
                "This combination is meaningless, please double-check and remove one "
                "(probably --align_constrained)."
            )
        )
        exit(1)

    try:
        args.stage_pr = check_path(args.stage_pr)  # Use a ckpt to provide pr
        if args.compare_mode == "global":
            print(
                red(
                    "Error: When --stage_pr is a path of ckpt, --compare_mode MUST be local. "
                    "Please use --compare_mode local and rerun"
                )
            )
            exit(1)
    except ValueError:
        if isfloat(args.stage_pr):
            args.stage_pr = float(
                args.stage_pr
            )  # Global pruning: only the global sparsity ratio is given
        else:
            assert args.index_layer == "name_matching"
            args.stage_pr = strdict_to_dict(args.stage_pr, float)
    args.reinit_layers = strlist_to_list(args.reinit_layers, str)

    # Set up finetune lr
    assert args.lr_ft, "--lr_ft must be provided"
    args.lr_ft = strdict_to_dict(args.lr_ft, float)

    return args
