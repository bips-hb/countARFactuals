import argparse
import os
import pandas as pd
import pyro.distributions as dist
import torch

import dgp_v2
import dgp as dgp_v1
import illustrative
from visualize import plot_pairs, get_savepath

v1_dgps = ['bn_5', 'bn_10', 'bn_50']
illustrative_dgps = {
    'pawelczyk': illustrative.ModelPawelczyk(),
    'two_sines': illustrative.TwoSines(),
    'cassini': illustrative.Cassini()
}
v2_dgps = ['bn_10_v2', 'bn_20', 'bn_100', 'bn_5_v2', 'bn_10_v2']
dgp_names = [] + v1_dgps + list(illustrative_dgps.keys()) + v2_dgps

dgps_root = 'python/synthetic/'
dgps_v2_root = 'python/synthetic_v2/'
dgps_path = dgps_root + 'dgps/'
dgps_v2_path = dgps_v2_root + '/dgps/'

dgpname = 'bn_5'
cf_path = 'cfs/bn5_cfs.csv'

parser = argparse.ArgumentParser()
parser.add_argument('dgpname')
parser.add_argument('cf_path')


args = parser.parse_args()
dgpname = args.dgpname
cf_path = args.cf_path
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
    if dgpname in v2_dgps:
        model = dgp_v2.DGP.load(dgps_v2_path, dgpname, cfs.shape[0])
    else:
        model = dgp_v1.DGP.load(dgps_path, dgpname, cfs.shape[0])
    log_probs_0 = model.log_prob(cfss_y[0])
    log_probs_1 = model.log_prob(cfss_y[1])
    log_probs = torch.logsumexp(torch.stack([log_probs_0, log_probs_1]), dim=0)

foreground = cfs.copy()
cfs['log_probs'] = log_probs

try:
    if dgpname in list(illustrative_dgps.keys()) + v1_dgps:
        background = pd.read_csv(dgps_root + '{}.csv'.format(dgpname))
    elif dgpname in v2_dgps:
        background = pd.read_csv(dgps_v2_root + '{}.csv'.format(dgpname))
    foreground['y'] = 'cf'
    foreground = foreground.sample(50)
    background = background.sample(150)
    df_visualize = pd.concat([background, foreground])
    plot_pairs(df_visualize, '', sample=False,
               savepath=cf_path.replace('.csv', ''))
except Exception as e:
    print('Could not visualize')
    print(e)

cfs.to_csv(cf_path.replace('.csv', '_with_log_probs.csv'), index=False)
