# Meta-MAPG

Complete the missing 2-agent HalfCheetah part of the code.

### Pretraining in 2-agent HalfCheetah environment

SAC implemented in [tianshou](https://github.com/thu-ml/tianshou) is used to train 2 cooperative agents in HalfCheetah-v2. LSTM policy does not work well in this environment (reward is low), so we choose to use normal MLP policy instead. See codes in `pretrain_mujoco`. Pretrained models were moved to `pretrain_model/HalfCheetah-v2`.

### Meta-MAPG

Add MLP policy to adapt to MLP peer and simplify meta-agent training.

