import argparse
from tests.data import DataloaderTests, DatasetTests

parser = argparse.ArgumentParser(description='argparser for tests')
parser.add_argument('--datasets',default=False, action='store_true')
parser.add_argument('--dataloader',default=False, action='store_true')
parser.add_argument('-all',default=False, action='store_true')
parser.add_argument('--spec',default=False, action='store_true')

def main():
    global args
    args = parser.parse_args()
    if args.spec:
        dl_tests = DataloaderTests()
        dl_tests.test_resnet_cifar10_layer_parameters()
        return
    if args.all:
        ds_tests = DatasetTests()
        ds_tests.test_all()

        dl_tests = DataloaderTests()
        dl_tests.test_all()
        return
    if args.datasets:
        ds_tests = DatasetTests()
        ds_tests.test_all()
    if args.dataloader:
        dl_tests = DataloaderTests()
        dl_tests.test_all()

if __name__ == "__main__":
    main()
