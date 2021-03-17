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
        self.rnn = OneLayerRNN(input_size=self.dataset.input_size, hidden_size=args.hidden_size,
                               output_size=self.dataset.output_size)
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
            out_file = 'RNN_h{}_ep{}_bs{}_maxsamples{}'.format(args.hidden_size, args.ep, args.bs, args.max_samples)
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            output_folder = os.path.join(args.output_path, out_file, datetime_folder)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            return output_folder

    def save_model(self):
        with open(os.path.join(self.out_folder, "model.pt"), 'wb') as f:
            torch.save(self.rnn, f)

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
        preds = torch.stack(preds, dim=0)  # (num_batch, batch_size, 1, S, output_size)
        preds = preds.view(-1, preds.size(2), preds.size(-2), preds.size(-1))
        return mse, preds

    def _save_ckpt(self, EPOCH, loss):
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': self.rnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(self.ckpt_path, 'model.pt'))

    def _load_ckpt(self):
        if os.path.exists(os.path.join(self.ckpt_path, "model.pt")):
            checkpoint = torch.load(os.path.join(self.ckpt_path, 'model.pt'), map_location=self.device)
            self.rnn.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            self.logger.info("loading checkpoint for epoch {} from {}".format(epoch, self.ckpt_path))
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
                preds, _ = self.rnn(datapoints)  # (B,S,P,D)
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
        self.logger.info(
            "-------------------------------------end of training--------------------------------------------")

    def test(self):
        test_loss, test_preds = self.eval(data_loader=self.test_loader)
        self.logger.info("TEST LOSS:{}".format(test_loss))
        inputs, _ = next(iter(self.test_loader))
        with torch.no_grad():
            predictions_test, _ = self.rnn(inputs)
            inputs = inputs.squeeze().cpu().numpy()
            predictions_test = predictions_test.squeeze().numpy()
        if len(predictions_test.shape) > 2:
            for _ in range(10):
                self.plot_preds_targets(inputs=inputs, predictions=predictions_test, out_folder=self.out_folder)

    def generate_observations(self, sigma_init, sigma_h, sigma_y, num_samples, num_data_samples):
        observations_folder = os.path.join(self.out_folder,
                                           "observations_samples{}_sigmainit{}_sigmah{}_sigmay{}".format(num_data_samples, sigma_init, sigma_h,
                                                                                               sigma_y))
        if not os.path.isdir(observations_folder):
            os.makedirs(observations_folder)
        observations, hidden = [], []
        mse = 0.
        for batch, (inputs, targets) in enumerate(self.test_loader):
            initial_input = inputs[:, :, 0, :].unsqueeze(dim=-2)
            seq_len = self.dataset.seq_len
            observations_batch, hidden_batch = self.rnn.generate_observations(initial_input=initial_input, seq_len=seq_len, sigma_init=sigma_init,
                                           sigma_h=sigma_h, sigma_y=sigma_y, num_samples=num_samples)
            observations.append(observations_batch)
            hidden.append(hidden_batch)
            # check correctness of observations:
            mean_observations_batch = observations_batch.mean(dim=1).unsqueeze(1)
            loss_batch = self.criterion(mean_observations_batch, targets)
            mse += loss_batch
            if batch <= 10:
               self.plot_preds_targets(inputs=inputs.squeeze(dim=1), predictions=mean_observations_batch.squeeze(dim=1), out_folder=observations_folder, batch=batch)
        mse = mse / len(self.test_loader)
        self.logger.info("MSE BETWEEN MEAN OBSERVATIONS AND GROUND TRUTH: {}".format(mse))
        # saving observations and states
        observations = torch.stack(observations, dim=0) # (num_batch, batch_size, num_samples, seq_len, F_y)
        observations = observations.view(-1, observations.size(2), observations.size(-2), observations.size(-1))
        hidden = torch.stack(hidden, dim=0)  # (num_batch, batch_size, num_samples, seq_len, F_y)
        hidden = hidden.view(-1, hidden.size(2), hidden.size(-2), hidden.size(-1))
        np.save(os.path.join(observations_folder, "observations.npy"), observations.cpu().numpy())
        np.save(os.path.join(observations_folder, "states.npy"), hidden.cpu().numpy())

    def plot_preds_targets(self, inputs, predictions, out_folder, batch=None):
            index = np.random.randint(inputs.shape[0])
            inp = inputs[index, :, 0]
            mean_pred = predictions[index, :, 0]
            x = np.linspace(1, self.dataset.seq_len, self.dataset.seq_len)
            plt.plot(x, mean_pred, 'red', lw=2, label='predictions for sample: {}'.format(index))
            plt.plot(x, inp, 'cyan', lw=2, label='ground-truth for sample: {}'.format(index))
            plt.legend(fontsize=10)
            out_file = "plot_test_preds_targets_sample{}".format(index) if batch is None else "plot_test_preds_targets_batch{}_sample{}".format(batch, index)
            plt.savefig(os.path.join(out_folder, out_file))
            plt.close()
