# coding: utf-8
# 2021/4/23 @ zengxiaonan

import logging

import numpy as np
import torch
import tqdm
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

from EduKTM import KTM


class Net(nn.Module):
    def __init__(self, num_questions, hidden_size, num_layers):
        '''
        :param num_questions:知识点数
        :param hidden_size:
        :param num_layers: RNN有几层全连接网络
        '''
        super(Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        # batch_first: True or False 表示是否input data的第一维表示的是batch_size
        self.rnn = nn.RNN(num_questions * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, num_questions)


    def forward(self, x):
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # num_directions: 1-单向RNN 2-双向RNN
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        out, _ = self.rnn(x, h0)
        res = torch.sigmoid(self.fc(out))
        return res


def process_raw_pred(raw_question_matrix, raw_pred, num_questions: int) -> tuple:
    '''
    :param raw_question_matrix: 训练集的one-hot向量 sequence_len * skill_num
    :param raw_pred: DKT预测的学生认知状态 sequence_len * skill_num
    :param num_questions: 知识点数 skill_num
    :return:
    Key statemtn:
    raw_question_matrix[:, 0: num_questions]:试题考察的知识点 如果学生在这道题答对，这个向量是有1的
    raw_question_matrix[:, num_questions:]:学生的作答情况 如果学生在这道题答错，这个向量是有1的
    '''

    question_matrix = raw_question_matrix[:, 0: num_questions] + raw_question_matrix[:, num_questions:]  # 类似q-matrix
    answer_pred_matrix = raw_pred[: -1].mm(question_matrix[1:].t())  # 矩阵相乘 （sequence_len-1 * 123）x（123 * sequence_len-1）
    index = torch.LongTensor([range(answer_pred_matrix.shape[0])])  # 0-(sequnece_len-2)
    pred = answer_pred_matrix.gather(0, index)[0]  #
    truth = (((raw_question_matrix[:, 0: num_questions] - raw_question_matrix[:, num_questions:]).sum(1) + 1) // 2)[1:]
    # //整除
    return pred, truth


class DKT(KTM):
    def __init__(self, num_questions, hidden_size, num_layers):
        super(DKT, self).__init__()
        self.num_questions = num_questions  # 知识点数 不是试题数
        self.dkt_model = Net(num_questions, hidden_size, num_layers)

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002) -> ...:
        loss_function = nn.BCEWithLogitsLoss()  # 把BCELoss(交叉熵)和sigmoid融合了
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr)

        for e in range(epoch):
            losses = []
            for batch in tqdm.tqdm(train_data, "Epoch %s" % e):
                '''RNN的input应是三维的[batch_size, sequence_len, input_size]'''
                integrated_pred = self.dkt_model(batch)  # batch_size *  result(sequence_len * skill_num)
                batch_size = batch.shape[0]
                loss = torch.Tensor([0.0])
                for student in range(batch_size):
                    '''student:batch中的一个
                       batch[student]:训练集的数据 (one-hot向量)
                       integrated_pred[student]：预测的学生的认知状态
                       self.num_question:知识点数
                    '''
                    pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
                    loss += loss_function(pred[pred > 0], truth[pred > 0])

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                auc = self.eval(test_data)
                print("[Epoch %d] auc: %.6f" % (e, auc))

    def eval(self, test_data) -> float:
        self.dkt_model.eval()
        y_pred = torch.Tensor([])
        y_truth = torch.Tensor([])
        for batch in tqdm.tqdm(test_data, "evaluating"):
            integrated_pred = self.dkt_model(batch)
            batch_size = batch.shape[0]
            for student in range(batch_size):
                pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
                y_pred = torch.cat([y_pred, pred[pred > 0]])
                y_truth = torch.cat([y_truth, truth[pred > 0]])

        return roc_auc_score(y_truth.detach().numpy(), y_pred.detach().numpy())

    def save(self, filepath):
        torch.save(self.dkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
