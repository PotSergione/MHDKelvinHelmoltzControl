import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from kh_env import KHEnv
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
from utilities import image_initialiser, plotter, stats_printer

args = argparse.ArgumentParser(description="PID Controller Optimization")
args.add_argument('--NumEvalsperStep', type=int, default=1, help='Number of evaluations per step', required=False)
args.add_argument('--NumSteps', type=int, default=25, help='Number of optimization steps', required=False)
args.add_argument('--test', type=bool, default=False, help='Run in test mode', required=True)

args = args.parse_args()


env = KHEnv()

# --- PID Controller Class ---
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.reset()

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return np.clip(output, 0, 1)


# --- PID Evaluation Function ---
def evaluate_pid(Kp, Ki, Kd, render=False):
    n_eval = args.NumEvalsperStep
    controller = PIDController(Kp, Ki, Kd)


    total_rewards = []
    start = time.time()
    for _ in range(n_eval):  # Average over a few runs

        if render:
                fig, img, strat, x = image_initialiser(rows=2, cols=1)

        cont  = []
        env.reset()
        controller.reset()
        done = False
        dt = 0.1  # 10Hz control loop
        reward = 0
        while not done:
            if render:
                plotter(env.matrix, fig, img, strat, x, cont)
            error = np.mean(np.abs(env.vx[env.index])) 
            action = np.array([controller.compute(error, dt)])
            cont.append(action)
            _, reward, term, trun, _ = env.step(action)
            done = term or trun
        if env.iteration < 100:
            total_rewards.append(50)
        else:
            total_rewards.append(np.sum(cont)) # we are minimizing the total control power
        
        if render:
            plt.close(fig)

    env.proc.terminate()
    print(f"Evaluation took {time.time() - start:.2f} seconds, mean current: {np.mean(total_rewards):.2f}")
    return np.mean(total_rewards)  # We're minimizing the total control power 


# --- Optimizer ---
def optimise(x0):
    # --- Define Search Space ---
    space = [
        # PID parameters: Kp, Ki, Kd
        Real(-1000.0, 1000.0, name='Kp'),
        Real(-1000.0, 1000.0, name='Ki'),
        Real(-1000.0, 1000.0, name='Kd'),
    ]

    @use_named_args(space)
    def objective(**params):
        return evaluate_pid(**params)

    # --- Run Bayesian Optimization ---
    print("Starting Bayesian optimization...")
    result = gp_minimize(
        func=objective,
        dimensions=space,
        x0=x0,
        n_calls=args.NumSteps,             # Total evaluations
        acq_func="gp_hedge",          # Expected improvement
        random_state=0,
        callback= lambda res: print(f"Current best: {res.fun:.2f} with params {res.x}"),
        verbose=True
        #x0 = np.array([98.799, 10.0, 50.0])  # Initial guess from previous shenaningans
    )

    best_Kp, best_Ki, best_Kd = result.x
    print(f"\nBest PID values found:")
    print(f"Kp = {best_Kp:.2f}, Ki = {best_Ki:.2f}, Kd = {best_Kd:.2f}")
    print(f"Best average reward: {-result.fun:.2f}")
    params = np.array(result.x_iters)
    # --- Output Best Result ---
    return best_Kp, best_Ki, best_Kd, params, result.func_vals



if __name__ == "__main__":

    if False:
        try:
            df = pd.read_csv('gp_minimize_results.csv')
            best_Kp = df['Kp'].iloc[np.argmax(df['reward']).astype(int)]
            best_Ki = df['Ki'].iloc[np.argmax(df['reward']).astype(int)]
            best_Kd = df['Kd'].iloc[np.argmax(df['reward']).astype(int)] 
            best_Kp, best_Ki, best_Kd, params, rewards = optimise([best_Kp, best_Ki, best_Kd])
            temp = pd.DataFrame(params, columns=['Kp', 'Ki', 'Kd']) 
            temp['reward'] = rewards
            df = pd.concat([df, temp], ignore_index=True)
            df.to_csv('gp_minimize_results.csv', index=False)
        except FileNotFoundError:
            print("No previous database of results found, starting optimization...")
            best_Kp, best_Ki, best_Kd, params, rewards = optimise()  # Initial guess
            df = pd.DataFrame(params, columns=['Kp', 'Ki', 'Kd']) 
            df['reward'] = rewards
            df.to_csv('gp_minimize_results.csv', index=False)
    else:
        df = pd.read_csv('gp_minimize_results.csv')
        best_Kp = df['Kp'].iloc[np.argmax(df['reward']).astype(int)]
        best_Ki = df['Ki'].iloc[np.argmax(df['reward']).astype(int)]
        best_Kd = df['Kd'].iloc[np.argmax(df['reward']).astype(int)] 
        start = time.time()
        evaluate_pid(best_Kp, best_Ki, best_Kd, render=True)
        stats_printer(env, start=start)