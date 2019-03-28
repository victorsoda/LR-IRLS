import numpy as np

data_path = './a9a/'
train_file = data_path + 'a9a'
test_file = data_path + 'a9a.t'
feature_num = 123


def read_data_from_file(filename):
    X = []
    y = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split(' ')
            y.append(int(tmp[0]))
            x = np.zeros(feature_num)
            for i in range(1, len(tmp)-1):
                x[int(tmp[i].split(':')[0])-1] = 1
            X.append(x)
    y = np.array(y)
    y[y < 0] = 0
    X = np.array(X)
    # X = X[:2000]
    # y = y[:2000]
    return X, y


def sigmoid(z):
    ex = np.exp(z)
    return ex / (1 + ex)


def get_evaluations(X, y, w):
    mu = sigmoid(X@w)
    mu[mu < 0.5] = 0
    mu[mu >= 0.5] = 1
    sub = mu - y
    accuracy = 1 - np.sum(np.abs(sub)) / y.shape[0]
    TP = np.sum(mu * y)
    FP = np.sum(sub[sub == 1])
    FN = np.abs(np.sum(sub[sub == -1]))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1_score


def L2_norm(w):
    return np.linalg.norm(w)


def IRLS(X, y, lam, maxiter=10, break_thres=0.001):
    """

    :param X: training features X
    :param y: training labels y
    :param lam: lambda for l2-norm penalty of w
    :param maxiter:
    :param break_thres: break the IRLS when || w_{t+1} - w_t ||_1 < break_thres
    :return:
    """
    N = y.shape[0]
    R = np.diag(np.repeat(1, N))
    # z = np.linalg.inv(R)@y
    # H = X.T@R@X + lam * np.eye(feature_num)
    # w = np.linalg.pinv(H)@(X.T@R@z)  # 为什么要这样初始化呢？
    w = np.random.normal(0, 0.01, feature_num)
    print("Start iterations..")
    for it in range(maxiter):
        Xw = X@w
        mu = sigmoid(Xw)
        r = np.multiply(mu, 1 - mu)
        R = np.diag(r)
        R_inv = np.diag(1 / r)
        XR = X.T@R
        H = XR@X + lam * np.eye(feature_num)
        z = Xw - R_inv@(mu - y)
        w_next = np.linalg.pinv(H)@XR@z    # H求伪逆
        if np.sum(np.abs(w_next - w)) < break_thres:
            return w_next
        else:
            # print("Thres =", np.sum(np.abs(w_next - w)))
            w = w_next
            eva = get_evaluations(X, y, w)
            print("Iter =", it, ", accuracy =", eva[0], ", l2_norm =", L2_norm(w))
    return w


if __name__ == '__main__':
    train_X, train_y = read_data_from_file(train_file)
    test_X, test_y = read_data_from_file(test_file)
    print("y.shape =", train_y.shape)
    with open("result.txt", 'w', encoding='utf-8') as f:
        for lam in [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]:
        # for lam in [0.1]:
            w_final = IRLS(train_X, train_y, lam)
            eva = get_evaluations(test_X, test_y, w_final)
            print("When lambda = %f test accuracy = %f, precision = %f, recall = %f, f1_score = %f, l2_norm = %f"
                  % (lam, eva[0], eva[1], eva[2], eva[3], L2_norm(w_final)))
            f.write("%f,%f,%f,%f,%f,%f\n" % (lam, eva[0], eva[1], eva[2], eva[3], L2_norm(w_final)))

