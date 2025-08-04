import numpy as np
import subprocess
import gymnasium as gym
import os


# NEW IDEA: REMOVE THE REWARD COMING FROM THE VX, JUST CONSIDER AN OBSTACLE AVOIDANCE PROBLEM WITH A L1 PENALISATION.
class KHEnv(gym.Env):

    metadata={"render_mode":"human",
              "output_folder":"snapshots",
              "save":False, "output_dt":.1}
    act_metadata={'Nx':128, 'Nz':64, "num_actions":1, 'penalty':0.0}
    sim_metadata={"Lx":2, "Lz":1, 'Nx':12, 'Nz':16}


    def __init__(self, seed=0, render_mode=None, data_folder=1):

        #np.random.seed(seed)
        #self.seed = seed
        self.eps = np.random.uniform(0.001, 0.1) # amplitude of the perturbation
        self.phase = np.random.uniform(np.pi/2, np.pi* 3/2) # phase of the perturbation
        self.observation_space = gym.spaces.Box(-100, 100, shape=(1, 1, (1 + self.sim_metadata['Nx']*self.sim_metadata['Nz'])))
        self.action_space = gym.spaces.Box(0, 1, shape=(self.act_metadata['num_actions'],))

        self.data_folder = "data/data_"+str(data_folder)+"/"
        self.terminated =  False

        self.update_parameter(self.data_folder + "BOUT.inp", "nx", self.act_metadata['Nx'])
        self.update_parameter(self.data_folder + "BOUT.inp", "nz", self.act_metadata['Nz'])

        self.probes_z = np.linspace(0, 64, 16, endpoint=False).astype(int)
        self.probes_x = np.linspace(42, 86, 12, endpoint=True).astype(int)#np.array([56, 58, 60, 62, 64, 66, 68, 70])#
        self.index = np.ix_(self.probes_x, self.probes_z)

        self.dx = self.sim_metadata['Lx']/self.act_metadata['Nx']
        self.dz = self.sim_metadata['Lz']/self.act_metadata['Nz']
        self.controls = []
        self.rewards = []
        self.vx_energy = []
        self.action = [0.0]


    def _d3_init(self):
        
        self._printer_problem_started()
        # necessary to update the parameters
        self.eps = np.random.uniform(0.1, 0.3)#np.random.uniform(.1, .5)
        self.phase = np.random.uniform(np.pi/2, np.pi* 3/2)
        self.update_parameter(self.data_folder + "BOUT.inp", "epsilon", self.eps)
        self.update_parameter(self.data_folder + "BOUT.inp", "phase", self.phase)

        self.x0 = np.random.uniform(0.4, 0.6)
        self.update_parameter(self.data_folder + "BOUT.inp", "x0", self.x0)
        self.proc = subprocess.Popen(
        ["mpirun","-np", "4", "build/./BOUT_KH","-d",self.data_folder],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,                # handle text I/O (strings)
        bufsize=1                 # line-buffered
        )


    def update_parameter(self, filename, param_name, new_value):
        with open(filename, 'r') as file:
            lines = file.readlines()

        updated_lines = []
        for line in lines:
            if line.strip().startswith(param_name + " ="):
                updated_line = f"{param_name} = {new_value}\n"
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)

        with open(filename, 'w') as file:
            file.writelines(updated_lines)


    def _compute_reward(self):
        #dist = self.compute_difference_in_norm(self.stable_config, self.matrix)
        #cont = np.array(self.controls)

        vx_mean = np.mean(np.abs(self.vx[self.index]))
        stability_reward =  np.exp(-100*(vx_mean - 0.015))/np.exp(1.5)  # Soft penalty before 0.02 threshold
        
        # Control effort penalty (L2 for smoothness)
        control_penalty = -3*self.controls[-1]
        # if len(self.controls) > 6:
        #     recent_actions = np.array(self.controls[-5:])
        #     action_std = np.std(recent_actions)
        #     pulsation_reward = 0.05 * action_std - 0.1 * np.mean(np.abs(np.diff(recent_actions)))
        # else:
        #     pulsation_reward = 0

        return 0.001*(self.iteration - 5*np.sum(self.controls)) - self.controls[-1]*0.1#control_penalty #0.01*(self.iteration - 2*np.sum(self.controls)) 
        #return (stability_reward + control_penalty) * 0.05


    def _extract_observation(self):

        Nx = self.sim_metadata['Nx']
        Nz = self.sim_metadata['Nz']

        obs = np.reshape(self.matrix[self.index], (Nx * Nz, ))
        obs = np.hstack((obs, self.action))  # Add the last action to the observation
        obs = np.reshape(obs, (1, 1, Nx*Nz+1))

        if np.isnan(obs).any(): 
            self.nan = True
            return np.ones_like(obs)

        return obs/np.max(np.abs(obs))
    

    def _printer_problem_started(self): 
        print(' '*30 + '-------------------------------------')
        print(' '*30 + '--------PROBLEM INITIALIZED----------')
        print(' '*30 + '-------------------------------------')
    

    def reset(self, seed=None, options={}):
        super().reset()

        # all that follow commented out is for the mpi version which still needs troubleshooting
        #if self.terminated and hasattr(self, "proc"):
        #    print('Found a terminated process, cleaning up...')
        #    self.proc.terminate()
            # try:
            #     self.proc.wait(timeout=5)
            # except subprocess.TimeoutExpired:
            #     print("Process didn't terminate in time. Forcing kill...")
            #     self.proc.kill()
            #     self.proc.wait()
        #    subprocess.run(["pkill", "-f", "BOUT_KH"], check=False)
        #    subprocess.run(["pkill", "-f", "mpiexec"], check=False)
        #    subprocess.run(["pkill", "-f", "prterun"], check=False)
        if self.terminated:
            self.proc.terminate()
        subprocess.Popen(["./cleaner_env.sh",f"{self.data_folder}"])
        self.terminated = False
        self._d3_init()
        self.iteration = 1
        self.rewards = []
        self.controls = []
        self.vx_energy = []
        self.nan = False

        while True:
            line = self.proc.stdout.readline()
            if "INPUT" and 'time = 0.000000e+00' in line:
                self.stable_config =  np.fromfile(self.data_folder+"data.bin", dtype=np.float64).reshape((self.act_metadata['Nx'],self.act_metadata['Nz']))
                self.matrix        =  self.stable_config   
                self.vx0 =  np.fromfile(self.data_folder+"vx.bin", dtype=np.float64).reshape((self.act_metadata['Nx'],self.act_metadata['Nz']))
                self.vx        =  self.vx0

                self.proc.stdin.write(f"{0}\n")# 0 action to proceed with c++ sim. 
                self.proc.stdin.flush()     
                break

        obs = self._extract_observation()
        info = {'Obtained data'}

        return obs, info
    

    def compute_difference_in_norm(self, matrix1, matrix2):
        """
        Compute norm of the two matrices"
        """
        integrand = np.abs(matrix1 - matrix2)
        #integrand = np.abs(matrix2)
        return np.sqrt(np.sum(integrand*(self.dx*self.dz))) #- 1.16261 #baseline
    

    def _compute_vx_energy(self):
        integrand = np.abs(self.vx ** 2)
        #print(np.mean(np.abs(self.vx[self.index])))
        return np.max(self.vx)#np.sum(integrand*(self.dx*self.dz))


    def step(self, action=0.0):
        action *= .6
        terminated=False
        truncated=terminated
        while not terminated:
            line = self.proc.stdout.readline()
            if line == '' and self.proc.poll() is not None:
                terminated = True
                break
            if "INPUT" in line:
                try:       
                    self.matrix = np.fromfile(self.data_folder+"data.bin", dtype=np.float64).reshape(
                        (self.act_metadata['Nx'],self.act_metadata['Nz']))
                    self.vx =  np.fromfile(self.data_folder+"vx.bin", dtype=np.float64).reshape(
                        (self.act_metadata['Nx'],self.act_metadata['Nz']))
                    self.controls.append(action)
                    self.proc.stdin.write(f"{float(action[0])}\n")
                    self.proc.stdin.flush()
                    self.action = action
                    self.iteration+=1
                    break

                except FileNotFoundError:
                    print("data.bin not found, skipping...")
                    continue


        self.vx_energy.append(self._compute_vx_energy())
        obs = self._extract_observation()

        val = 0.02 #when 0.3 on omega
        
        if np.mean(np.abs(self.vx[self.index])) >= val or self.nan == True:
            self.terminated = True
            terminated = True
            truncated = True
            reward = -5.0
        # elif np.mean(np.abs(self.vx[self.index])) >= 0.015:  # Warning zone
        #     reward = self._compute_reward().item() - .5  
        else:
            reward = self._compute_reward().item()

        if self.iteration == 100:
            self.terminated = True
            terminated = True
            truncated = False

        info = {'reward':reward}

        self.rewards.append(reward)
        if terminated: print(f"Total reward: {np.sum(self.rewards)}, Final action: {action}, Truncate: {truncated}, final reward: {reward}, steps:{self.iteration}")
        return (obs, reward, terminated, truncated, info)