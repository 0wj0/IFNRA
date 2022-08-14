import argparse
import torch
from sklearn import metrics
from data_utils import DatasetReader, BucketIterator
from models import IFNRA
from transformers import BertTokenizer


def evaluate_acc_f1(model, test_data_loader, opt):
    # switch model to evaluation mode
    model.eval()
    n_test_correct, n_test_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    with torch.no_grad():
        for t_batch, t_sample_batched in enumerate(test_data_loader):
            t_inputs = [t_sample_batched[col].to(opt.device) for col in \
                        ['input_ids', 'attention_mask', 'img_roi_features', 'img_roi_msk', 'aspect_msk']]
            t_targets = t_sample_batched['label'].to(opt.device)
            t_outputs = model(t_inputs)

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


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tw15_test_rst', type=str)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument("--my_trained_path", default="../my_trained/IFNRA_tw15/IFNRA_tw15.pkl")
    parser.add_argument("--pretrained_path", default="../pre-trained_model/bert_uncased_L-12_H-768_A-12/")
    parser.add_argument('--note', default='BUA_batch32_lr2e-5_seed666', type=str)
    opt = parser.parse_args()

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    model = IFNRA(opt.pretrained_path, gru_layer_num=1).to(opt.device)
    model.load_state_dict(torch.load(opt.my_trained_path))

    opt.tokenizer = BertTokenizer.from_pretrained(opt.pretrained_path)
    test_dataset = DatasetReader(dataset=opt.dataset, tokenizer=opt.tokenizer)
    test_data_loader = BucketIterator(data=test_dataset.data, batch_size=64, shuffle=False)

    test_acc, test_f1, test_f1_m = evaluate_acc_f1(model, test_data_loader, opt)
    print(f'test_acc:{test_acc}, test_f1:{test_f1}, test_f1_m:{test_f1_m}')
    f_out = open('tst_rst/' + opt.note + opt.dataset + '.txt', 'w', encoding='utf-8')
    f_out.write(f'test_acc:{test_acc}, test_f1:{test_f1}, test_f1_m:{test_f1_m}\n')
    f_out.close()
