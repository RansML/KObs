"""
Occupancy Mapping
Ransalu Senanayake
"""

import numpy as np
import matplotlib.pyplot as pl
import pickle as pk
import util
import hilbert
import sys

def load_parameters(case):
    parameters = \
        {
         'uiuc1': \
             {
             'train': 'datasets/uiuc1/uiuc1_train.csv', \
             'test':'', \
             'laser_max_distance':30, \
             'lims': [-25, 25, -5, 25], \
             'gamma': 0.1, \
             'alpha': 0.00001
              },

        'uiuc2': \
            {
                'train': 'datasets/uiuc2/uiuc2_train.csv', \
                'test': '', \
                'laser_max_distance': 30, \
                'lims': [-25, 25, -5, 25], \
                'gamma': 0.1, \
                'alpha': 0.00001
            },
        }

    return parameters[case]

def get_mesh_grid(resolution=1, limits=[-120, 120, -20, 120]):
   x_spaced = np.arange(limits[0], limits[1], resolution )
   y_spaced = np.arange(limits[2], limits[3], resolution)
   xx, yy = np.meshgrid(x_spaced, y_spaced)
   X_plot = np.vstack((xx.flatten(),yy.flatten())).T
   return X_plot, xx, yy

def get_weights(dataset='uiuc1'):
    print('Fitting individual static maps...')

    #load parameters
    paras = load_parameters(dataset)

    #load data
    #col0=scan index; col1=longitude; col2=latitude; col3=occupied/unoccupied
    X_all, Y_all = util.read_txy_csv(paras['train'])
    print('X_shape={}, Y_shape={}'.format(X_all.shape, Y_all.shape))
    X_plot, xx, yy = get_mesh_grid(resolution=0.5, limits=paras['lims'])

    #estimate individual map weights
    weights = []
    for ith_scan in range(np.max(X_all[:,0]).astype(np.int16)):
        ith_ind = X_all[:,0] == ith_scan
        X_t, Y_t = X_all[ith_ind,1:], Y_all[ith_ind,:]
        shm = hilbert.HilbertMap(gamma=paras['gamma'], grid=None, cell_resolution=(5, 5), cell_max_min=paras['lims'],
                                 X=X_t, alpha=paras['alpha'])
        shm.paras = paras
        shm.fit(X_t, Y_t)
        print('{}th fitted'.format(ith_scan))
        Y_plot = shm.predict_prob(X_plot).reshape(xx.shape)

        weights.append(shm.classifier.coef_[0,:].ravel())

        pl.close('all')
        pl.figure(figsize=(12,3))
        pl.subplot(131)
        pl.scatter(X_t[:,0], X_t[:,1], s=1, c=Y_t, cmap='jet')
        pl.colorbar()
        pl.scatter(shm.grid[:, 0], shm.grid[:, 1], c='k', marker='x')
        pl.xlim(paras['lims'][:2]); pl.ylim(paras['lims'][2:])
        pl.xticks(np.arange(-25, 26, 5))
        pl.title('Raw Data')
        pl.subplot(132)
        pl.xlim(paras['lims'][:2]); pl.ylim(paras['lims'][2:])
        pl.xticks(np.arange(-25, 26, 5))
        pl.scatter(X_plot[:,0], X_plot[:,1], c=Y_plot, s=10, cmap='jet')
        pl.colorbar()
        pl.title('Static Occupancy Map')
        pl.subplot(133)
        pl.scatter(shm.grid[:, 0], shm.grid[:, 1], c=shm.classifier.coef_[0][1:].reshape(shm.xx.shape), cmap='jet')
        pl.colorbar()
        pl.xlim(paras['lims'][:2]); pl.ylim(paras['lims'][2:])
        pl.xticks(np.arange(-25, 26, 5))
        pl.title('Weight Map')
        #pl.show()
        pl.savefig('output/{}/static_images/shm_{}.png'.format(dataset,ith_scan))

    sanity_check_W, sanity_check_T = 3, 4
    #shm.all_weights = np.array(weights).T[:sanity_check_W,:sanity_check_T] #weights vetically and time horizontally
    shm.all_weights = np.array(weights).T # weights vetically and time horizontally
    pk.dump(shm, open('output/{}/{}_shm.p'.format(dataset,dataset), 'wb'))

def solve_for_A(dataset='uiuc1'):
    print('\n\nSolving for A...')

    #load individal weights with the model
    shm = pk.load(open('output/{}/{}_shm.p'.format(dataset,dataset), 'rb'))
    W, xx, yy = shm.all_weights, shm.xx, shm.yy
    print('Individual map weight shape={}\n full W shape={}'.format(xx.shape, W.shape))

    #weight matrices
    W1 = np.hstack((np.zeros(W.shape[0])[:, None], W[:, :-1]))  # 0,t0,t1,t2,...,t(T-1)
    W2 = W[:,:] #t0,t1,t2,t3,...,tT
    print('W=\n{} \nW1\n={} \nW2\n={}\n'.format(W, W1, W2))

    #solve for A
    A_hat = (np.linalg.pinv(W1.dot(W1.T)).dot(W1.dot(W2.T))).T
    print('\nA_hat=\n',A_hat)

    #update the class
    shm.A_hat = A_hat
    pk.dump(shm, open('output/{}/{}_shm.p'.format(dataset,dataset), 'wb'))

    #save A_hat image
    pl.close('all')
    pl.figure()
    pl.imshow(A_hat, interpolation='None', cmap='jet')
    pl.colorbar()
    pl.title('A_hat')
    pl.savefig('output/{}/{}_A_hat.png'.format(dataset,dataset))


def predict(dataset='uiuc1'):
    print('\n\nPredicting for A...')

    #load A_hat and individal weights with the model
    shm = pk.load(open('output/{}/{}_shm.p'.format(dataset,dataset), 'rb'))

    max_t = shm.all_weights.shape[1] - 1
    for t in range(25):
        if t == 0:
            W1 = shm.all_weights[:, t][:, None]
        else:
            W1 = W2_hat
        print('W1=\n', W1)

        W2_hat = shm.A_hat.dot(W1)
        print('\nP=\n', W2_hat)
        if t < max_t:
            print('\nMSE=', np.average((W2_hat - shm.all_weights[:, t + 1][:, None])**2))

        #plot
        pl.close('all')
        pl.figure(figsize=(15,8))
        pl.suptitle('$t={}$'.format(t), fontsize=20)
        pl.subplot(231)
        pl.scatter(shm.xx.ravel(), shm.yy.ravel(), c=W1[1:].reshape(shm.xx.shape), cmap='jet')
        pl.colorbar()
        pl.title('True $W(t)$ | intercept={}'.format(str(np.round(W1[0],2))))

        pl.subplot(233)
        pl.scatter(shm.xx.ravel(), shm.yy.ravel(), c=W2_hat[1:].reshape(shm.xx.shape), cmap='jet')
        pl.colorbar()
        pl.title('Pred $W(t+1)$ | intercept={}'.format(str(np.round(W2_hat[0],2))))
        paras = shm.paras
        X_plot, xx, yy = get_mesh_grid(resolution=0.5, limits=paras['lims'])
        pl.subplot(234)
        shm.classifier.coef_ = W1.T
        Y_plot = shm.predict_prob(X_plot)
        pl.xlim(paras['lims'][:2]); pl.ylim(paras['lims'][2:])
        pl.xticks(np.arange(-25, 26, 5))
        pl.scatter(X_plot[:,0], X_plot[:,1], c=Y_plot, s=10, cmap='jet')
        pl.colorbar()
        pl.title('True Occu$(t)$')
        pl.subplot(236)
        shm.classifier.coef_ = W2_hat.T
        Y_plot = shm.predict_prob(X_plot)
        pl.xlim(paras['lims'][:2]); pl.ylim(paras['lims'][2:])
        pl.xticks(np.arange(-25, 26, 5))
        pl.scatter(X_plot[:,0], X_plot[:,1], c=Y_plot, s=10, cmap='jet')
        pl.colorbar()
        pl.title('Pred Occu$(t+1)$')

        #plot grund truth
        pl.subplot(232)
        if t < max_t:
            pl.scatter(shm.xx.ravel(), shm.yy.ravel(), c=shm.all_weights[1:, t + 1].reshape(shm.xx.shape), cmap='jet')
            pl.colorbar()
            pl.title('True $W(t+1)$ | intercept={}'.format(str(np.round(shm.all_weights[0, t + 1], 2))))
        pl.subplot(235)
        pl.title('True Occu$(t+1)$')
        pl.xlim(paras['lims'][:2]);   pl.ylim(paras['lims'][2:])
        pl.xticks(np.arange(-25, 26, 5))
        if t < max_t:
            shm.classifier.coef_ = shm.all_weights[:, t + 1][None,:]
            Y_plot = shm.predict_prob(X_plot)
            pl.scatter(X_plot[:,0], X_plot[:,1], c=Y_plot, s=10, cmap='jet')
            pl.colorbar()
        #pl.show()
        pl.savefig('output/{}/pred_images/shm_{}.png'.format(dataset, t))


if __name__ == '__main__':
    dataset = 'uiuc2'
    get_weights(dataset)
    solve_for_A(dataset)
    predict(dataset)
