import argparse

parser = argparse.ArgumentParser(description="Transformer-based Model Pruning -- SichengFeng")

parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet18", help="model architecture name")
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--experiment_name", type=str, default="test", help="unqiue name for the experiment")


# Data related args
parser.add_argument("--data_path", type=str, default="./data")
parser.add_argument("--dataset", type=str, default="imagenet", help="dataset name")
parser.add_argument("--dataset_dir", type=str, default=None, help="path of dataset folder")

parser.add_argument("-b", "--batch_size", default=128, type=int, help="mini-batch size")
parser.add_argument("-j", "--workers", default=12, type=int, help="number of data loading workers (default: 12)")


# args about pruning
parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
parser.add_argument('--global_pruning', default=False, action='store_true', help='global pruning')
parser.add_argument('--pruning_ratio', default=0.5, type=float, help='prune ratio')
parser.add_argument('--pruning_type', default='random', type=str, help='pruning type', choices=['random', 'l1', 'oracle'])
parser.add_argument('--oracle_seed', default=42, type=int, help='oracle seed')
parser.add_argument('--retrain_epoch', default=90, type=int, help='retrain epoch')
parser.add_argument('--retrain_lr', default=0.01, type=float, help='retrain init lr')
parser.add_argument('--lr_decay', default='linear', type=str, help='lr decay')
parser.add_argument("--weight_decay", "--wd", type=float, default=0.03)
parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
parser.add_argument("--label_smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
parser.add_argument("--mixup_alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
parser.add_argument("--cutmix_alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
parser.add_argument("--use_v2", action="store_true", help="Use V2 transforms")
parser.add_argument("--clip_grad_norm", default=1.0, type=float, help="clip grad norm")

args = parser.parse_args()
args.dataset_dir = args.dataset