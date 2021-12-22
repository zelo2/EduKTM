# coding: utf-8
# 2021/12/22 @ zelo2

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LPKTNet(nn.Module):
    def __init__(self, exercise_num, skill_num, stu_num, ans_time_num, interval_time_num, d_k, d_a, d_e, q_matrix):
        '''
        :param exercise_num: 试题数量
        :param skill_num: 知识点数量
        :param stu_num: 学生数量
        :param ans_time_num: 做答时间数量
        :param interval_time_num: 相邻Learning cell之间的interval time的数量
        :param d_a: Dimension of the answer (0-All Zero, 1-All One)
        :param d_e: Dimension of the exercise
        :param d_k: Dimension of the skill
        '''
        super(LPKTNet, self).__init__()
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.exercise_num = exercise_num
        self.skill_num = skill_num
        self.stu_num = stu_num
        self.ans_time_num = ans_time_num
        self.interval_time_num = interval_time_num
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


        self.gamma = 0.03  # for those values of the Q-matrix are zero
        self.q_matrix = q_matrix
        self.q_matrix[self.q_matrix == 0] = self.gamma  # Enhanced Q-matrix

        '''Dropout layer'''
        self.dropout = nn.Dropout(0.2)  # follow the original paper

        '''Exercise Embedding'''
        self.exercise_embed = nn.Embedding(self.exercise_num, self.d_e)


        '''Time Embedding'''
        self.ans_time_embed = nn.Embedding(self.ans_time_num, self.d_k)
        self.interval_time_embed = nn.Embedding(self.interval_time_num, self.d_k)


        '''Knowledge Embedding'''
        self.stu_mastery_embed = nn.Embedding(self.stu_num, self.d_k)


        '''MLP Construction'''
        # Learning gain Embedding
        # 这里的Embedding的input没有加入试题所考察的知识点向量 存疑？
        self.learning_gain_embed_layer = nn.Linear(self.d_e+self.d_k+self.d_a, self.d_k)  # input = exercise + answer time+ answer
        torch.nn.init.xavier_normal(self.learning_gain_embed_layer.weight)  # follow the original paper

        # Learning Obtain Layer
        self.learning_layer = nn.Linear(self.d_k * 4, self.d_k)  # input = l(t-1) + interval time + l(t) + h(t-1)
        torch.nn.init.xavier_normal(self.learning_layer.weight)

        # Learning Judge Layer
        self.learning_gate = nn.Linear(self.d_k * 4, self.d_k)  # input = l(t-1) + interval time + l(t) + h(t-1)
        torch.nn.init.xavier_normal(self.learning_gate.weight)

        # Forgetting Layer
        self.forgetting_gate = nn.Linear(self.d_k * 3, self.d_k)  # input = h(t-1) + learning gain (t) + interval time
        torch.nn.init.xavier_normal(self.forgetting_gate.weight)

        # Predicting Layer
        self.predicting_layer = nn.Linear(self.d_k * 2, self.d_k)  # input = exercise (t+1) + h (t)


    def forward(self, exercise_id, skill_id, stu_id, answer_value, ans_time, interval_time):
        '''
        :param exercise_id: 试题id序列
        :param skill_id: 知识点id序列
        :param stu_id: 学生id
        :param answer: 试题得分序列
        :param ans_time: 回答时间序列
        :param interval_time: 两次回答间隔时间序列 长度=前面的序列长度-1
        :return: Prediction
        E.g: stu_id-0
             exercise_id- 1, 2, 3, 4, 6
             skill_id- 1, 1, 1, 4, 5, 5
             answer- 1, 1, 0, 0, 0, 0
             ans_time- 5, 10, 15, 5, 20
             interval_time- 1000, 20000, 5000, 400
        '''

        batch_size, sequence_len = exercise_id.size(0), exercise_id.size(1)

        '''Supposing the units of the answer time and the interval time are both Second (s)'''
        interval_time /= 60  # discretize by minutes
        ans_time /= 1  # discretize by seconds

        '''Obtain the Embedding of each element'''
        exercise = self.exercise_embed(exercise_id)  # batch_size * sequence * d_e
        stu_mastery = self.stu_mastery_embed(stu_id)  # 1 x skill_num
        ans_time = self.ans_time_embed(ans_time)  # batch_size * sequence * d_k
        interval_time = self.interval_time_embed(interval_time)  # batch_size * sequence * d_k

        # process the answer
        answer = answer_value.view(-1, 1)  # (batch_size * sequence) * 1
        answer = answer.repeat(1, self.d_a)  # (batch_size * sequence) * d_a
        answer = answer.view(batch_size, -1, self.d_a)  # batch_size * sequence * d_a

        # initial the learning gain
        # 使用torch.cat((A,B),dim)时，除拼接维数dim数值可不同外其余维数数值需相同，方能对齐
        learning_gain = self.learning_gain_embed_layer(torch.cat((exercise, ans_time, answer), 2))

        '''Batch size train'''
        # 每个作答序列，我们都需要两两拿出来进行训练
        for echo in range(sequence_len - 1):
            exercise_vector = exercise[:, echo]  # batch_size * d_e
            answer_time_vector = ans_time[:, echo]  # batch_size * d_k
            answer_vector = ans_time[:, echo]  # batch_size * d_a






        return 0