import torch as th
from sb3_contrib import RecurrentPPO
from kh_env import KHEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import argparse
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

def lr_schedule(progress):
    return 4e-4 * progress  # goes from 1e-4 -> 0

parser = argparse.ArgumentParser()

parser.add_argument('-NumberOfEpochs', type=int, required=True, default=10)
parser.add_argument('-NumberOfTrainingEnvs', type=int, required=False, default=10)
parser.add_argument('--Seed', type=int, required=False,default=0)
parser.add_argument('-NumberOfEvalEnvs', type=int, required=False, default=10)
parser.add_argument('-LogsDirectory', type=str, required=False, default='model_folder')
parser.add_argument('-EvalFreq', type=int, required=False, default=1000)
parser.add_argument('-NumberOfEvalEpisodes', type=int, required=False, default=10)
parser.add_argument('-ModelName', type=str, required=False, default="model")


args = parser.parse_args()


def make_env(rank, log_dir=None):
    def _init():
        env = KHEnv(data_folder=rank)
        if log_dir is not None:
            env = Monitor(env, filename=f"{log_dir}/env_{rank}")
        else:
            env = Monitor(env)
        return env
    return _init


if __name__ == '__main__':

    env_fns = [make_env(i, "./MHDControlLog/MLP model_1") for i in range(1, args.NumberOfTrainingEnvs+1)]
    env = SubprocVecEnv(env_fns)
    
    model_dir = args.LogsDirectory 
    model_name = 'MLP' + ' ' + args.ModelName

    checkpoint_callback = CheckpointCallback(
        save_freq=200,             # Save every 2k steps
        save_path="./checkpoints/",  # Directory where models are saved
        name_prefix="ppo_mhd_control"  # Prefix for saved model files
    )

    
    if  model_dir in os.listdir() and model_name + '.zip' in os.listdir(model_dir): 
        print('Model Found:')
        model = RecurrentPPO.load(os.getcwd() + '/' + model_dir + '/' + model_name, env=env, print_system_info=True)
    else: 
        print('No model found -- initialising new one')

        n_steps = int(2000/(args.NumberOfTrainingEnvs))
        print(n_steps, model_name, model_dir)

        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[128, 128], lstm_hidden_size=32, n_lstm_layers=1, log_std_init= -.85, ortho_init=True)
        model = RecurrentPPO("MlpLstmPolicy", env=env, 
                    n_steps=n_steps,  
                    gamma=.98,
                    learning_rate=lr_schedule, 
                    batch_size=256, #256 original
                    n_epochs=5, 
                    ent_coef=0.0,
                    verbose=1, 
                    tensorboard_log="./MHDControlLog/",
                    policy_kwargs=policy_kwargs, 
                    use_sde=False, 
                    clip_range=0.2, 
                    normalize_advantage=True,
                    vf_coef=0.25,
                    max_grad_norm=0.5,
                    clip_range_vf=0.2
                    )

    model.learn(total_timesteps=2000*args.NumberOfEpochs, 
                tb_log_name=model_name, 
                reset_num_timesteps=False, 
                callback=checkpoint_callback
                )
    model.save(os.getcwd() + '/' + model_dir + '/' + model_name)