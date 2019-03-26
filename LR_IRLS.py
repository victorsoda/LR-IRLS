import numpy as np

data_path = './a9a/'
train_file = data_path + 'a9a'
test_file = data_path + 'a9a.t'
feature_num = 123
lam = 1


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
    X = X[:2000]
    y = y[:2000]
    return X, y


def sigmoid(z):
    ex = np.exp(z)
    return ex / (1 + ex)


def get_accuracy(X, y, w):
    mu = sigmoid(X@w)
    mu[mu < 0.5] = 0
    mu[mu >= 0.5] = 1
    tmp = np.abs(mu - y)
    acc = 1 - np.sum(tmp) / y.shape[0]
    return acc


def L2_norm(w):
    return np.linalg.norm(w)


def IRLS(X, y, maxiter=10, break_thres=0.001):
    N = y.shape[0]
    R = np.diag(np.repeat(1, N))
    z = np.linalg.inv(R)@y
    H = X.T@R@X + lam * np.eye(feature_num)
    w = np.linalg.pinv(H)@(X.T@R@z)  # 为什么要这样初始化呢？
    for it in range(maxiter):
        Xw = X@w
        mu = sigmoid(Xw)
        R = np.diag(np.multiply(mu, 1 - mu))
        XR = X.T@R
        H = XR@X + lam * np.eye(feature_num)
        z = Xw - np.linalg.inv(R)@(mu - y)
        w_next = np.linalg.pinv(H)@XR@z    # H求伪逆
        # print(w_next)
        if np.sum(np.abs(w_next - w)) < break_thres:
            return w_next
        else:
            print("Thres =", np.sum(np.abs(w_next - w)))
            w = w_next
            acc = get_accuracy(X, y, w)
            print("Iter =", it, ", accuracy =", acc, ", l2_norm =", L2_norm(w))


if __name__ == '__main__':
    train_X, train_y = read_data_from_file(train_file)
    test_X, test_y = read_data_from_file(test_file)
    print("y.shape =", train_y.shape)
    w_final = IRLS(train_X, train_y)

    print("Test accuracy =", get_accuracy(test_X, test_y, w_final), ", l2_norm =", L2_norm(w_final))


