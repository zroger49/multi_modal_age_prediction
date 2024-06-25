import os
import torch
#  ---------------------------------------
# Parameters
# ---------------------------------------
import yaml
def get_parameters(path):
    with open(path) as params:
        params_dict = yaml.safe_load(params)
    return params_dict

params__global = get_parameters("./conf/params__global.yml")
params__model  = get_parameters("./conf/params__model.yml")

TISSUE_TYPE = params__global['TISSUE_TYPE']
OUTPUT_DIR = params__global['OUTPUT_DIR'].format(TISSUE_TYPE)
DEVICE = torch.device(params__global['DEVICE'])
path__test_sample = params__global["PATH_TestSample"].format(TISSUE_TYPE)

reweight_opt  = params__model['reweight_opt' ]
normalize_opt = params__model['normalize_opt']
custom_loss   = params__model['custom_loss'  ]
hipt_features = params__model['hipt_features']
optimizer_opt = params__model['optimizer_opt']
learning_rate = params__model['learning_rate']
n_layers      = params__model['n_layers'     ]
n_nodes       = params__model['n_nodes'      ]
dropout_rate  = params__model['dropout_rate' ]
n_epochs      = params__model['n_epochs'     ]
batch         = params__model['batch'        ]
lds_ks        = params__model['lds_ks'       ]
lds_sigma     = params__model['lds_sigma'    ]
lds_kernel    = params__model['lds_kernel'   ]

n_feat_meth = params__model['n_feat_meth']
feat_select_meth = params__model['feat_select_meth']


feat_select_gene = params__model['feat_select_gene']
n_feat_gene = params__model['n_feat_gene']
path__histology = params__global['PATH__Histology']

path__methylation = params__global['PATH__Methylation']
path__methylation_meta = os.path.join(path__methylation, 'methylation_epic_v1.0b5.csv')
path__methylation = os.path.join(path__methylation, TISSUE_TYPE)

path__methylation_raw = os.path.join(path__methylation, f'raw/methylation_{TISSUE_TYPE.lower()}.csv')
path__methylation_tensor = os.path.join(path__methylation, f'tensor/tensor_{TISSUE_TYPE.lower()}.pt')
path__methylation_probes = os.path.join(path__methylation, f'probes/probes.pkl')
path__methylation_subjects = os.path.join(path__methylation, f'subjects/subjects.pkl')
path__methylation_features_linear = os.path.join(path__methylation, f'features/features__linear.pkl')
path__methylation_features_dml = os.path.join(path__methylation, f'features/features__dml.pkl')