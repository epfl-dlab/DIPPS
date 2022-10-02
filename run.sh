# This bash script allows for rerunning the experiments from the paper.
# The experiment parameters can be set in this script itself.

# Random seed used
seed=1111

# Feature dimension of synthetic Gaussian mixture data, only used in 
# synthetic experiments
feat_dim=20

# Latent dimension for synthetic Gaussian mixture data, only used
# in synthetic experiments
n_component=5

# Epsilon of differentially private mechanism
epsilon=1

# Configuration file specifying the settings of the autoencoder used
# to generate the cluster, when using an autoencoder instead of a GMM
config_file=./config/config_all.cfg

# Source of the dataset
data_dir=./data/web_visits

# Number of latent components of the autoencoder. When 0, the number gets
# selected by an automatic implementation of ELBOW
n_latent_component=0

python3 run.py  --random_seed ${seed} \
                    --feat_dim ${feat_dim}  \
                    --n_component ${n_component} \
                    --epsilon ${epsilon} \
                    --config_file ${config_file}  \
                    --data_dir  ${data_dir}  \
                    --n_latent_component ${n_latent_component}
