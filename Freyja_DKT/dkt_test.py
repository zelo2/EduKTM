import random
import pandas as pd
import tqdm
import numpy as np

data = pd.read_csv(
    '../data/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv',
    usecols=['order_id', 'user_id', 'sequence_id', 'skill_id', 'correct']).fillna(0)  # 给缺省的skill_id填为0
# print(data.isnull().sum())  # 缺省信息

raw_question = data.skill_id.unique().tolist()  # 知识点id
num_skill = len(raw_question)  # 124 skills, 因为把缺省的知识点全部转换为0

def question_id_transfer(question):
    '''
    :param question: 知识点id矩阵
    :return: 知识点id矩阵，每个知识点对应的id的一个字典
    '''
    id2question = [p for p in raw_question]
    question2id = {}
    for i, p in enumerate(raw_question):
        question2id[p] = i

    return id2question, question2id

id2question, question2id = question_id_transfer(raw_question)


def parse_all_seq(students):
    '''
    :param students: 学生的id（user_id）这里是unique的，没有重复的学生id。
    :return:
    '''
    all_sequences = []
    for student_id in tqdm.tqdm(students, 'parse student sequence:\t'):  # tqdm 进度条
        student_sequence = parse_student_seq(data[data.user_id == student_id])  # 将每个学生的答题记录进行预处理
        all_sequences.extend(student_sequence)  # 在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
    return all_sequences

def parse_student_seq(student):
    '''
    :param student: 一个学生的答题记录，包含'order_id', 'user_id', 'sequence_id', 'skill_id', 'correct'
    :return:
    '''
    student = student.drop_duplicates(subset='order_id')  # 将这列对应值相同的行进行去重（除去重复答题记录）
    sequence_ids = student.sequence_id.unique()  # 学生的答题集（sequence）的id集合
    sequences = []  # list [(知识点id，对应的得分)] length:学生的答题集(sequence)的个数
    for seq_id in sequence_ids:  # 遍历sequence
        seq = student[student.sequence_id == seq_id].sort_values('order_id')  # 按照order_id大小进行排序得到学生在某一问题集上的答题情况
        questions = [question2id[id] for id in seq.skill_id.tolist()]  # 知识点的id
        answers = seq.correct.tolist()
        sequences.append((questions, answers))
    return sequences  # 长度为学生作答的sequence的长度，每个元素都是tuple类型，里面有两个list，代表考察的知识点和对应的答题情况

'''得到每个学生在每个sequnce上的得分情况
   length:学生数 x 每个学生做过的sequnce数 = 59874'''
sequences = parse_all_seq(data.user_id.unique())

def train_test_split(data, train_size=.7, shuffle=True):
    '''
    :param data: 学生在每个sequence上的得分情况，每个元素是一个tuple([skill_id], [score profiles])
    :param train_size: 训练机比例
    :param shuffle: 是否打乱
    :return:测试集与训练集
    '''
    if shuffle:
        random.shuffle(data)
    boundary = round(len(data) * train_size)
    return data[: boundary], data[boundary:]

train_sequences, test_sequences = train_test_split(sequences)  # 得到训练集和测试集

def sequences2tl(sequences, trgpath):
    with open(trgpath, 'a', encoding='utf8') as f:
        for seq in tqdm.tqdm(sequences, 'write into file: '):
            # seq:([skill_id], [score profiles])
            questions, answers = seq
            seq_len = len(questions)
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(q) for q in questions]) + '\n')
            f.write(','.join([str(a) for a in answers]) + '\n')
"""
   得到的数据每三行作为一组数据（Triple Line Format data）
   第一行：学生在每个问题集（sequence）上的答题次数
   第二行：每个子问题考察的知识点
   第三行：对应的答题情况（0-答错，1-答对）
"""
# save triple line format for other tasks
sequences2tl(train_sequences, '../../data/2009_skill_builder_data_corrected/train.txt')
sequences2tl(test_sequences, '../../data/2009_skill_builder_data_corrected/test.txt')

MAX_STEP = 50  # sequence length
NUM_QUESTIONS = num_skill  # 知识点个数

'''复现DKT原文，将得分情况编码为one_hot作为输入'''
def encode_onehot(sequences, max_step, num_questions):
    question_sequences = np.array([])
    answer_sequences = np.array([])
    onehot_result = []

    for questions, answers in tqdm.tqdm(sequences, 'convert to onehot format: '):
        # seq: ([skill_id], [score profiles])
        # questions: [skill_id]
        # answers: [score profiles]
        length = len(questions)

        # append questions' and answers' length to an integer multiple of max_step
        mod = 0 if length % max_step == 0 else (max_step - length % max_step)
        fill_content = np.zeros(mod) - 1  # -1填充，保证向量长度都是max_step(50)的倍数？
        questions = np.append(questions, fill_content)  # 类似list.extend()
        answers = np.append(answers, fill_content)

        # one-hot
        q_seqs = questions.reshape([-1, max_step])
        a_seqs = answers.reshape([-1, max_step])
        for (i, q_seq) in enumerate(q_seqs):
            # q_seq: max_step
            onehot = np.zeros(shape=[max_step, 2 * num_questions])  # max_step x skill_num
            for j in range(max_step):
                # 核心句：如果学生答对该试题（得分大于0），index为该知识点的id所在的位置。
                # 如果答错或者没有记录，index为at（学生作答情况向量）上对应知识点id所在位置。
                index = int(q_seq[j] if a_seqs[i][j] > 0 else q_seq[j] + num_questions)
                onehot[j][index] = 1
            onehot_result = np.append(onehot_result, onehot)

    return onehot_result.reshape(-1, max_step, 2 * num_questions)


# reduce the amount of data for example running faster
percentage = 0.05
train_data = encode_onehot(train_sequences[: int(len(train_sequences) * percentage)], MAX_STEP, NUM_QUESTIONS)
test_data = encode_onehot(test_sequences[: int(len(test_sequences) * percentage)], MAX_STEP, NUM_QUESTIONS)