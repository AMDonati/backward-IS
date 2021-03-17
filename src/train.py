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
    parser.add_argument("-save_path", type=str, help="path for saved model folder (if load checkpoint)")
    # generate arguments
    parser.add_argument("-bs_test", type=int, help="batch size for generating observations")
    parser.add_argument("-data_samples", type=int, default=3000, help="number of data samples in the test dataset to generate observations on.")
    parser.add_argument("-num_samples", type=int, default=100,
                        help="number of samples for each observation.")
    parser.add_argument("-sigma_init", type=float,
                        help="covariance matrix for initial hidden state")
    parser.add_argument("-sigma_h", type=float,
                        help="covariance matrix for the internal gaussian noise for the transition function.")
    parser.add_argument("-sigma_y", type=float,
                        help="covariance matrix for the internal gaussian noise for the observation model.")

    return parser

def run(args):

    # -------------------------------- Upload dataset ----------------------------------------------------------------------------------
    dataset = Dataset(data_path=args.data_path, name=args.dataset,
                          max_samples=args.max_samples, max_size_test=args.data_samples)

    algo = RNNAlgo(dataset=dataset, args=args)

    if args.ep > 0:
        algo.train()
        algo.save_model()
    else:
        print("skipping training...")
    algo.test()
    if args.sigma_init is not None and args.sigma_h is not None and args.sigma_y is not None:
        algo.generate_observations(sigma_init=args.sigma_init, sigma_h=args.sigma_h, sigma_y=args.sigma_y, num_samples=args.num_samples, num_data_samples=args.data_samples)
    else:
        list_sigmas_init = [0.001, 0.01, 0.05, 0.1]
        list_sigmas_h = [0.001, 0.01, 0.05, 0.1]
        list_sigmas_y = [0.001, 0.01, 0.05, 0.1]
        for sigma_h, sigma_init, sigma_y in zip(list_sigmas_init, list_sigmas_h, list_sigmas_y):
            algo.logger.info("sigma_init: {} - sigma_h: {} - sigma_y:{}".format(sigma_init, sigma_h, sigma_y))
            algo.generate_observations(sigma_init=sigma_init, sigma_h=sigma_h, sigma_y=sigma_y,
                                       num_samples=args.num_samples, num_data_samples=args.data_samples)
            algo.logger.info("--------------------------------------------------------------------------------")



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
