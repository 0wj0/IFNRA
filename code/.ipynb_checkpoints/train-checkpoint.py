import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from sklearn import metrics
from data_utils import DatasetReader, BucketIterator
from models import IFNRA
from transformers import BertTokenizer
from torch.optim import AdamW


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        train_dataset = DatasetReader(dataset=opt.datasets['train'], tokenizer=opt.tokenizer)
        test_dataset = DatasetReader(dataset=opt.datasets['dev'], tokenizer=opt.tokenizer)

        self.train_data_loader = BucketIterator(data=train_dataset.data, batch_size=opt.batch_size,
                                                shuffle=True, sort=False)
        self.test_data_loader = BucketIterator(data=test_dataset.data, batch_size=opt.eval_batch_size,
                                               shuffle=False)

        self.model = IFNRA(opt.pretrained_path, gru_layer_num=1).to(opt.device)
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for n, p in self.model.named_parameters():
            if 'bert' not in n and 'resnet' not in n and p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer):
        max_test_acc = 0
        max_test_f1 = 0
        max_test_f1_m = 0
        global_step = 0
        continue_not_increase = 0
        
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                '''targets.shape = [batch_size]（情感极性）'''
                targets = sample_batched['label'].to(self.opt.device)
                outputs = self.model(inputs)
                
                loss = criterion(outputs, targets)
            
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, test_f1, test_f1_m = self._evaluate_acc_f1()
                    if test_f1_m > max_test_f1_m:
                        max_test_f1_m = test_f1_m
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            save_directory = '../my_trained/' + 'IFNRA' + '_' + self.opt.dataset + '/'
                            if not os.path.exists(save_directory):
                                os.mkdir(save_directory)
                            self.opt.tokenizer.save_pretrained(save_directory)
                            torch.save(self.model.state_dict(),
                                       save_directory + 'IFNRA' + '_' + self.opt.dataset + '.pkl')
                            print('>>> best model saved.')
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}, test_f1_m: {:.4f}'.format(
                        loss.item(), train_acc, test_acc, test_f1, test_f1_m))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        return max_test_acc, max_test_f1, max_test_f1_m

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['label'].to(opt.device)
                t_outputs = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        f1_mi = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                                 average='micro')
        f1_m = 0.5 * (f1 + f1_mi)

        return test_acc, f1, f1_m

    def run(self, repeats=12):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        # _params = filter(lambda p: p.requires_grad, self.model.parameters())
        param_optimizer = filter(lambda np: np[1].requires_grad, self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.opt.l2reg},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        # optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.learning_rate, eps=self.opt.adam_epsilon)
        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/' + 'BertClassifier' + '_' + self.opt.dataset + '_val.txt', 'w', encoding='utf-8')

        max_test_acc_global = 0
        max_test_f1_global = 0
        max_test_f1_m_global = 0
        for i in range(repeats):
            print('repeat: ', (i + 1))
            f_out.write('repeat: ' + str(i + 1) + '\n')
            self._reset_params()
            max_test_acc, max_test_f1, max_test_f1_m = self._train(criterion, optimizer)
            print('max_test_acc: {0}     max_test_f1: {1}     max_test_f1_m: {2}'.format(max_test_acc, max_test_f1,
                                                                                         max_test_f1_m))
            f_out.write('max_test_acc: {0}, max_test_f1: {1}, max_test_f1_m: {2}\n'.format(max_test_acc, max_test_f1,
                                                                                         max_test_f1_m))
            if max_test_acc > max_test_acc_global:
                max_test_acc_global = max_test_acc
            if max_test_f1 > max_test_f1_global:
                max_test_f1_global = max_test_f1
            if max_test_f1_m > max_test_f1_m_global:
                max_test_f1_m_global = max_test_f1_m
            print('#' * 100)
        print("max_test_acc_global:", max_test_acc_global)
        print("max_test_f1_global:", max_test_f1_global)
        print('max_test_f1_m_global:', max_test_f1_m_global)

        f_out.close()
        return max_test_acc_global, max_test_f1_global, max_test_f1_m_global


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tw15', type=str, help='tw15, tw17')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--adam_epsilon', default=1e-6, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=15, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=64, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument("--pretrained_path", default="../pre-trained_model/bert_uncased_L-12_H-768_A-12/")
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input_colses = ['input_ids', 'attention_mask', 'img_roi_features', 'img_roi_msk', 'aspect_msk']
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    datasets = {
        'train': opt.dataset+'_train_rst',
        'dev': opt.dataset + '_dev_rst',
        'test': opt.dataset + '_test_rst'
    }
    opt.datasets = datasets
    opt.inputs_cols = input_colses
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    opt.tokenizer = BertTokenizer.from_pretrained(opt.pretrained_path)
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run(repeats=5)
    print(f'max_global_f1:{ins.global_f1}')
