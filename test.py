import numpy as np
import os
from kh_env import KHEnv
from sb3_contrib import RecurrentPPO
import time
from utilities import image_initialiser, plotter, stats_printer



if __name__ == '__main__':
    
    env = KHEnv()
    model_dir = "model_folder"
    model_name = "MLP paper_J_02.zip"
    model_name = os.path.join(model_dir, model_name)
    model = RecurrentPPO.load(model_name, env=env, print_system_info=True)


    obs, info = env.reset()
    
    cont = []
    state = None
    term = False    
    fig, img, strat, x = image_initialiser(rows=2, cols=1)
    tot_rew = []
    i = 0

    action, state = model.predict(obs, state=state, episode_start=True,  deterministic=True)
    start = time.time()
    while not term:
        plotter(env.matrix, fig, img, strat, x,cont)
        #print(np.mean(np.abs(env.vx[env.index])))
        #action = np.random.uniform(0,1, size=(1,))#np.array([0.5])
        cont.append(env.action)
        obs, reward, term, trun, info = env.step(action)
        action, state = model.predict(obs, state=state,  deterministic=True)
        tot_rew.append(reward)

        i+=1
    env.proc.terminate()
    stats_printer(env, start=start)