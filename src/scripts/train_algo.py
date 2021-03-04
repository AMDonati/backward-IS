import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.stochastic_RNN import OneLayerRNN
import datetime
from scripts.utils import create_logger, saving_training_history
import matplotlib.pyplot as plt


class RNNAlgo:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.bs = args.bs
        self.output_path = args.output_path
        self.save_path = args.save_path
        self.train_loader, self.val_loader, self.test_loader = self.dataset.data_to_dataset(args=args)
        self.lr = args.lr
        self.EPOCHS = args.ep
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnn = OneLayerRNN(input_size=self.dataset.input_size, hidden_size=args.hidden_size, output_size=self.dataset.output_size)
        self.optimizer = optim.Adam(self.rnn.parameters(), lr=self.lr)
        self.out_folder = self._create_out_folder(args=args)
        self.logger = self._create_logger()
        self.ckpt_path = self._create_ckpt_path()
        _, _ = self._load_ckpt()
        self._save_hparams(args=args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_logger(self):
        out_file_log = os.path.join(self.out_folder, 'training_log.log')
        logger = create_logger(out_file_log=out_file_log)
        return logger

    def _create_ckpt_path(self):
        checkpoint_path = os.path.join(self.out_folder, "checkpoints")
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        return checkpoint_path

    def _save_hparams(self, args):
        dict_hparams = vars(args)
        dict_hparams = {key: str(value) for key, value in dict_hparams.items()}
        config_path = os.path.join(self.out_folder, "config.json")
        with open(config_path, 'w') as fp:
            json.dump(dict_hparams, fp, sort_keys=True, indent=4)

    def _create_out_folder(self, args):
        if args.save_path is not None:
            return args.save_path
        else:
            out_file = 'RNN_h{}'.format(args.hidden_size)
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            output_folder = os.path.join(args.output_path, out_file, datetime_folder)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            return output_folder

    def eval(self, data_loader):
        self.rnn.eval()
        losses, preds = [], []
        with torch.no_grad():
            for (X, y) in data_loader:
                # forward pass on val_loss.
                X = X.to(self.device)
                y = y.to(self.device)
                preds_batch, _ = self.rnn(X)
                preds.append(preds_batch)
                loss_batch = self.criterion(preds_batch, y)
                losses.append(loss_batch.cpu().numpy())
        mse = np.mean(losses)
        preds = torch.stack(preds, dim=0) # (num_batch, batch_size, 1, S, output_size)
        preds = preds.view(-1, preds.size(2), preds.size(-2), preds.size(-1))
        return mse, preds

    def _save_ckpt(self, EPOCH, loss):
        ckpt_path = os.path.join(self.ckpt_path, "RNN")
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': self.rnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(ckpt_path, 'model.pt'))

    def _load_ckpt(self):
        ckpt_path = os.path.join(self.ckpt_path, "RNN")
        if os.path.isdir(ckpt_path):
            checkpoint = torch.load(os.path.join(ckpt_path, 'model.pt'), map_location=self.device)
            self.rnn.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            self.logger.info("loading checkpoint for epoch {} from {}".format(epoch, ckpt_path))
        else:
            epoch = 0
            loss = None
        self.start_epoch = epoch
        return epoch, loss

    def train(self):
        print("starting training ...")
        train_mse_history, val_mse_history = [], []
        for epoch in range(self.start_epoch, self.EPOCHS):
            train_loss = 0.
            for i, (datapoints, labels) in enumerate(self.train_loader):
                datapoints, labels = datapoints.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                preds, _ = self.rnn(datapoints) # (B,S,P,D)
                loss = self.criterion(preds, labels)
                train_loss += loss
                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / len(self.train_loader)
            train_mse_history.append(train_loss)
            val_mse, _ = self.eval(data_loader=self.val_loader)
            val_mse_history.append(val_mse)

            self.logger.info("Epoch: {}/{}".format(str(epoch + 1), self.EPOCHS))
            self.logger.info("Train-Loss: {:.4f}".format(train_loss.detach().numpy()))
            self.logger.info("Val-loss: {:.4f}".format(val_mse))
            self._save_ckpt(EPOCH=epoch, loss=loss)

        # storing history of losses and accuracies in a csv file
        keys = ['train mse', 'val mse']
        values = [train_mse_history, val_mse_history]
        csv_fname = 'rnn_history.csv'
        saving_training_history(keys=keys,
                                values=values,
                                output_path=self.out_folder,
                                csv_fname=csv_fname,
                                logger=self.logger,
                                start_epoch=self.start_epoch)
        self.logger.info("-------------------------------------end of training--------------------------------------------")

    def test(self):
        test_loss, test_preds = self.eval(data_loader=self.test_loader)
        self.logger.info("TEST LOSS:{}".format(test_loss))
        for _ in range(10):
            self.plot_preds_targets()

    def plot_preds_targets(self):
        inputs, targets = next(iter(self.test_loader))
        with torch.no_grad():
            predictions_test, _ = self.rnn(inputs)
            inputs = inputs.squeeze()
            targets = targets.squeeze()
            predictions_test = predictions_test.squeeze()
            index = np.random.randint(inputs.shape[0])
            inp, tar = inputs[index], targets[index]
            mean_pred = predictions_test[index, :, 0].numpy()
            inp = inp[:, 0].numpy()
            x = np.linspace(1, self.dataset.seq_len, self.dataset.seq_len)
            plt.plot(x, mean_pred, 'red', lw=2, label='predictions for sample: {}'.format(index))
            # plt.plot(x, tar, 'blue', lw=2, label='targets for sample: {}'.format(index))
            plt.plot(x, inp, 'cyan', lw=2, label='ground-truth for sample: {}'.format(index))
            plt.legend(fontsize=10)
            plt.savefig(os.path.join(self.out_folder, "plot_test_preds_targets_sample{}".format(index)))
            plt.close()
