import os
import glob

from smilelogging import argparser as parser
from smilelogging.utils import strdict_to_dict, check_path
from smilelogging.slutils import blue, yellow

from pruner.prune_utils import set_up_prune_args


def add_args(parser):
    # Model related args
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
        help="model architecture name",
    )
    parser.add_argument(
        "--conv_type",
        type=str,
        default="default",
        choices=["default", "wn"],
        help="convolution layer type",
    )
    parser.add_argument(
        "--not_use_bn",
        dest="use_bn",
        default=True,
        action="store_false",
        help="if use BN in the network",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="activation function",
        choices=["relu", "leaky_relu", "linear", "tanh", "sigmoid"],
    )

    # Data related args
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="imagenet", help="dataset name")
    parser.add_argument(
        "--dataset_dir", type=str, default=None, help="path of dataset folder"
    )

    # Training related args
    parser.add_argument(
        "--init", type=str, default="default", help="parameter initialization scheme"
    )
    parser.add_argument("--lr", type=str, default="0:0.1")
    parser.add_argument(
        "-b",
        "--batch-size",
        "--batch_size",
        default=256,
        type=int,
        help="mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Optimizer momentum (default: 0.9)"
    )
    parser.add_argument("--weight_decay", "--wd", type=float, default=0.0001)
    parser.add_argument(
        "--solver", "--optim", type=str, default="SGD", help="optimizer type"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--no_ddp",
        dest="ddp",
        action="store_false",
        default=True,
        help="if use ddp in training; default: True",
    )
    parser.add_argument(
        "--print_interval",
        "--i_print",
        type=int,
        default=100,
        help="interval to print logs during training",
    )
    parser.add_argument("--test_interval", type=int, default=2000)
    parser.add_argument("--plot_interval", type=int, default=100000000)
    parser.add_argument(
        "--save_model_interval", type=int, default=-1, help="the interval to save model"
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="",
        help="supposed to replace the original 'resume' feature",
    )
    parser.add_argument(
        "--pretrained_ckpt", type=str, default=None, help="path of pretrained ckpt"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--directly_ft_weights",
        type=str,
        default="",
        help="the path to a pretrained model",
    )
    parser.add_argument(
        "--test_pretrained", action="store_true", help="test the pretrained model"
    )
    parser.add_argument(
        "--test_pretrained_only", action="store_true", help="test the pretrained model"
    )
    parser.add_argument(
        "--save_init_model",
        action="store_true",
        help="save the model after initialization",
    )

    # GPU/DP/DDP related args
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="num of steps for gradient accumulation",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="mixed precision training",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=12,
        type=int,
        help="number of data loading workers (default: 12)",
    )

    # Advanced LR scheduling related
    parser.add_argument("--advanced_lr.ON", action="store_true")
    parser.add_argument(
        "--advanced_lr.lr_decay",
        type=str,
        choices=["step", "cos", "cos_v2", "linear", "schedule"],
    )
    parser.add_argument("--advanced_lr.warmup_epoch", type=int, default=0)
    parser.add_argument("--advanced_lr.min_lr", type=float, default=1e-5)

    # This code base also serves to quick-check properties of deep neural networks.
    # These functionalities are summarized here.
    parser.add_argument("--utils.ON", action="store_true")
    parser.add_argument("--utils.check_kernel_spatial_dist", action="store_true")
    parser.add_argument("--utils.check_grad_norm", action="store_true")
    parser.add_argument("--utils.check_grad_stats", action="store_true")
    parser.add_argument("--utils.check_grad_history", action="store_true")
    parser.add_argument("--utils.check_weight_stats", action="store_true")

    # Other args for analysis
    parser.add_argument("--rescale", type=str, default="")
    parser.add_argument(
        "--jsv_loop",
        type=int,
        default=0,
        help="num of batch loops when checking Jacobian singuar values",
    )
    parser.add_argument(
        "--jsv_interval", type=int, default=-1, help="the interval of printing jsv"
    )
    parser.add_argument(
        "--jsv_rand_data",
        action="store_true",
        help="if use data in random order to check JSV",
    )
    parser.add_argument("--test_trainset", action="store_true")
    parser.add_argument("--ema", type=float, default=0)
    parser.add_argument("--batch_oracle", type=int, default=0)
    parser.add_argument("--random", type=str, default="no")
    parser.add_argument("--save_combination", type=str, default="no")
    parser.add_argument(
    "--no_scp", action="store_true", help="not scp experiment to hub"
    )
    parser.add_argument("--num_missions", type=int, default=1000, help="num of missions")
    parser.add_argument("--num_batches", type=int, default=8, help="num of batches")
    return parser


def check_args(args):
    args.lr = strdict_to_dict(args.lr, float)

    # Check pretrained ckpt; fetch it if it is unavailable locally
    if args.pretrained_ckpt:
        print(
            f"==> Checking pretrained_ckpt at path {yellow(args.pretrained_ckpt)}",
            end="",
            flush=True,
        )
        candidates = glob.glob(args.pretrained_ckpt)
        if len(candidates) == 0:
            print(", not found it. Fetching it...", end="", flush=True)
            folder, file = os.path.split(args.pretrained_ckpt)
            script = f"sh scripts/set_up_pretrained_models.sh {folder} {file}"
            os.system(script)
            print(", fetch it done!", flush=True)
        elif len(candidates) == 1:
            print(", found it!", flush=True)
        elif len(candidates) > 1:
            print(
                ", found more than 1 ckpt candidates; please check --pretrained_ckpt",
                flush=True,
            )
            exit(1)
        args.pretrained_ckpt = check_path(args.pretrained_ckpt)

    if args.dataset_dir is None:
        args.dataset_dir = args.dataset

    if "linear" in args.arch.lower():
        args.activation = "linear"

    # Check arch name
    if (
        args.dataset in ["cifar10", "cifar100"]
        and args.arch.startswith("vgg")
        and not args.arch.endswith("_C")
    ):
        print(
            f'==> Error: Detected CIFAR dataset used while the VGG net names do not end with "_C". Fix this, e.g., '
            f"change vgg19 to vgg19_C"
        )
        exit(1)

    return args


def check_unknown(unknown, debug):
    if len(unknown):
        print(f"Unknown args. Please check in case of unexpected setups: {unknown}")

        # Check unknown args in case of wrong setups
        # TODO-@mst: this is a bit ad-hoc, a better solution?
        if "--base_model_path" in unknown:
            print(
                f'Error: "--base_model_path" is retired, use "--pretrained_ckpt" instead'
            )
        if "--wd" in unknown:
            print(f'Error: "--wd" is retired, use "--weight_decay" instead')

        if not debug:
            exit(1)


parser = add_args(parser)  # These args are those independent of pruning
args, unknown = set_up_prune_args(
    parser
)  # If a pruning method is used, its args will not be added to parser.

check_unknown(unknown, args.debug)
args = check_args(args)
