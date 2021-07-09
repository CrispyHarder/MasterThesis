import argparse
from util.models.generators.layer_wise.evaluation import check_mean_std_generated

parser = argparse.ArgumentParser()

parser.add_argument('-path', type=str, help='path to the generator')
parser.add_argument('-number_samples', type=int, default=5,
                    help='how many samples should be taken')

args = parser.parse_args()

check_mean_std_generated(args.path,args.number_samples)
