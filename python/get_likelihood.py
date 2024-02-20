import argparse
import os
import pandas as pd
import pyro.distributions as dist
import torch

import dgp
import illustrative

dgp_names = ['bn_5', 'bn_10', 'bn_50', 'bn_100', 'cassini', 'pawelczyk', 'two_sines']
illustrative_dgps = {
    'pawelczyk': illustrative.ModelPawelczyk(),
    'two_sines': illustrative.TwoSines(),
    'cassini': illustrative.Cassini()
}


dgpname = 'bn_5'
cf_path = '../cfs/bn5_cfs.csv'
dgps_path = 'synthetic/dgps/'

parser = argparse.ArgumentParser()
parser.add_argument('dgpname')
parser.add_argument('cf_path')
parser.add_argument('--dgpspath', default='python/synthetic/dgps/')


args = parser.parse_args()
dgpname = args.dgpname
cf_path = args.cf_path
dgps_path = args.dgpspath
dgp_path = dgps_path + dgpname + '/'

if not dgpname in dgp_names:
    raise ValueError('DGP name muss be one of {}'.format(dgp_names))
if not os.path.exists(cf_path):
    raise ValueError('cf path does not exist')

cfs = pd.read_csv(cf_path)
assert 'y' not in cfs.columns
cfss_y = []
for ii in [0, 1]:
    cfs_y = cfs.copy()
    cfs_y['y'] = ii
    cfss_y.append(cfs_y)

if dgpname in illustrative_dgps.keys():
    model = illustrative_dgps[dgpname]
    log_probs = model.get_log_likelihood(cfs)
else:
    if not os.path.exists(dgp_path):
        raise ValueError('DGP path does not exist')
    if not dgps_path.endswith('/'):
        raise ValueError('cf path must end with / (should be a folder)')
    model = dgp.DGP.load(dgps_path, dgpname, cfs.shape[0])
    log_probs_0 = model.log_prob(cfss_y[0])
    log_probs_1 = model.log_prob(cfss_y[1])
    log_probs = torch.logsumexp(torch.stack([log_probs_0, log_probs_1]), dim=0)

cfs['log_probs'] = log_probs

cfs.to_csv(cf_path.replace('.csv', '_with_log_probs.csv'), index=False)
