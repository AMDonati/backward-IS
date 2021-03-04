import argparse
from data_provider.dataset import Dataset
from scripts.train_algo import RNNAlgo

def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-dataset", type=str, default='weather', help='dataset selection')
    parser.add_argument("-data_path", type=str, default='data/weather', help="path for uploading the dataset")
    parser.add_argument("-max_samples", type=int, default=50000, help="max samples for train dataset")
    # model parameters:
    parser.add_argument("-num_layers", type=int, default=1, help="number of layers in the network")
    parser.add_argument("-hidden_size", type=int, default=32, help="number of rnn units")
    # training params.
    parser.add_argument("-bs", type=int, default=32, help="batch size")
    parser.add_argument("-ep", type=int, default=20, help="number of epochs")
    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
    # output_path params.
    parser.add_argument("-output_path", type=str, required=True, help="path for output folder")
    parser.add_argument("-save_path", type=str, help="path for saved model folder (if loading ckpt)")

    return parser

def run(args):

    # -------------------------------- Upload dataset ----------------------------------------------------------------------------------
    dataset = Dataset(data_path=args.data_path, name=args.dataset,
                          max_samples=args.max_samples)

    algo = RNNAlgo(dataset=dataset, args=args)

    if args.ep > 0:
        algo.train()
    else:
        print("skipping training...")
    algo.test()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
