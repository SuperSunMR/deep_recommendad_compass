import utils
import numpy as np
from sklearn.metrics import roc_auc_score

train_file = './data/train.yzx.txt'
test_file = './data/test.yzx.txt'

input_dim = utils.INPUT_DIM

train_data = utils.read_data(train_file)
train_data = utils.shuffle(train_data)
test_data = utils.read_data(test_file)

if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')

train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(utils.FIELD_SIZES)

min_round = 1
num_round = 40
early_stop_round = 2
batch_size = 512

field_sizes = utils.FIELD_SIZES
field_offsets = utils.FIELD_OFFSETS

def train(model):
    history_score = []
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            # for j in range(train_size / batch_size + 1):
            for j in range(train_size // batch_size + 1):
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = utils.slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        train_preds = model.run(model.y_prob, utils.slice(train_data)[0])
        test_preds = model.run(model.y_prob, utils.slice(test_data)[0])
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                        -1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
                break