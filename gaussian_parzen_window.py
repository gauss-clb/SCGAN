import numpy as np
import argparse
import utils

# refer to https://github.com/goodfeli/adversarial
# log p(x)
# = log int p(z, x) dz
# = log int p(z) p(x |z) dz
# = log E_z p(x|z)
# = log (1/m) sum_z p(x|z)
# = log (1/m) sum_z prod_i sqrt(1/(2 pi sigma^2)) exp( -0.5 (x_i-g(z)_i)^2 / sigma^2)
# = log  sqrt(1/(2 pi sigma^2))^d (1/m) sum_z prod_iexp( -0.5 (x_i-g(z)_i)^2 / sigma^2)
# = log  sqrt(1/(2 pi sigma^2))^d (1/m) sum_z exp( sum_i -0.5 (x_i-g(z)_i)^2 / sigma^2)
# = log  sqrt(1/(2 pi sigma^2))^d + log (1/m) sum_z exp( sum_i -0.5 (x_i-g(z)_i)^2 / sigma^2)
# = 0.5 d log  1/(2 pi sigma^2) + log (1/m) sum_z exp( sum_i -0.5 (x_i-g(z)_i)^2 / sigma^2)
# = -0.5 d log  (2 pi sigma^2) + log (1/m) sum_z exp( sum_i -0.5 (x_i-g(z)_i)^2 / sigma^2)


def log_mean_exp(t):
    # prevent overflow
    # \sum{i=1}^n e^{x_i} = e^\max{x_j} \sum{i=1}^n e^{x_i-\max{x_j}}
    max_ = t.max(1, keepdims=True)
    return np.squeeze(max_) + np.log(np.mean(np.exp(t - max_), 1)) # log-likelihood


def parzen_batch(x, mu, sigma):
    '''
        x: [batch_size, z_dim]
        mu: [n, z_dim]
        sigma: float scalar
    '''

    x = np.expand_dims(x, axis=1)
    mu = np.expand_dims(mu, axis=0)
    t = (((x - mu) / sigma)**2).sum(2)*(-0.5)
    E = log_mean_exp(t)
    Z = mu.shape[2] * np.log(sigma*np.sqrt(2*np.pi))
    return E - Z


def get_lls(x, mu, sigma, batch_size=10):
    lls = np.array([])
    num_samples = x.shape[0]
    num_batches = (num_samples+batch_size-1)//batch_size
    for i in range(num_batches):
        ll = parzen_batch(x[(i*batch_size):min((i+1)*batch_size, num_samples)], mu, sigma)
        lls = np.append(lls, ll)
    return lls


def cross_validate_sigma(x, mu, sigmas, batch_size=10):
    '''
        x: validation data
        mu: generative data
        sigmas: sigma sequence
    '''
    lls = []
    for sigma in sigmas:
        ll = get_lls(x, mu, sigma, batch_size=batch_size)
        lls.append(ll.mean())
        print('Sigma: {}, LL: {}'.format(sigma, ll.mean()))
    return sigmas[np.argmax(lls)]
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Gaussian parzen window, negative log-likelihood estimator.')
    parser.add_argument('-d', '--data_dir', default='/home/clb/dataset/mnist',  help='Directory to load mnist.')
    parser.add_argument('-g', '--gen_data_path', default='result/scgan_mnist/scgan_mnist.npy', help='Path to load generative data.')
    parser.add_argument('-l', '--limit_size', default=1000, type=int, help='The number of samples in validation.')
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    parser.add_argument('-c', '--cross_val', default=10, type=int,
                            help="Number of cross valiation folds")
    parser.add_argument('--sigma_start', default=-1, type=float)
    parser.add_argument('--sigma_end', default=0., type=float)
    parser.add_argument('--file', default='cgan_mnist.txt', help='File to save mean and std of log-likelihood.')
    args = parser.parse_args()

    # load mnist
    trainX, trainY, testX, testY = utils.load_mnist(args.data_dir)
    trainX = trainX.reshape([-1, 784]).astype(np.float32)/255.
    testX = testX.reshape([-1, 784]).astype(np.float32)/255.

    x = trainX[60000-args.limit_size:]
    mu = np.load(args.gen_data_path).astype(np.float32)/255.

    sigmas = np.logspace(args.sigma_start, args.sigma_end, args.cross_val)
    sigma = cross_validate_sigma(x, mu, sigmas, args.batch_size)
    print('Using Sigma: {}'.format(sigma))
    lls = get_lls(testX, mu, sigma, args.batch_size)
    print('Negative Log-Likelihood of Test Set = {}, Std: {}'.format(lls.mean(), lls.std()/np.sqrt(testX.shape[0])))
    with open(args.file, 'w') as file:
        file.write('Negative Log-Likelihood of Test Set = {}, Std: {}\n'.format(lls.mean(), lls.std()/np.sqrt(testX.shape[0]))) 