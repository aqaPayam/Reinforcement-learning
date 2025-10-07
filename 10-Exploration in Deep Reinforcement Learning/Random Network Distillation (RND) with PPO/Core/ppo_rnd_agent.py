# ppo_rnd_agent.py
# Core module for PPO policy, Random Network Distillation (RND), and training logic.

import torch
import numpy as np
from torch.optim import Adam
from Core.model import PolicyModel, PredictorModel, TargetModel
from Common.utils import mean_of_list, RunningMeanStd

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

class Brain:
    def __init__(self, **config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === Models ===
        self.current_policy = PolicyModel(config["state_shape"], config["n_actions"]).to(self.device)
        self.predictor_model = PredictorModel(config["obs_shape"]).to(self.device)
        self.target_model    = TargetModel(config["obs_shape"]).to(self.device)
        # Freeze target network parameters
        for p in self.target_model.parameters():
            p.requires_grad = False

        # === Optimizer ===
        params = list(self.current_policy.parameters()) + list(self.predictor_model.parameters())
        self.optimizer = Adam(params, lr=config["lr"])

        # === Running statistics for normalization ===
        self.state_rms = RunningMeanStd(shape=config["obs_shape"])
        self.int_reward_rms = RunningMeanStd(shape=(1,))
        self.mse_loss = torch.nn.MSELoss()

    def get_actions_and_values(self, obs_tensor, hidden_state):
        obs_tensor   = obs_tensor.to(self.device)
        hidden_state = hidden_state.to(self.device)
        with torch.no_grad():
            dist, int_v, ext_v, probs, new_hidden = self.current_policy(obs_tensor, hidden_state)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
        return (
            action.cpu(),
            int_v.cpu(),
            ext_v.cpu(),
            log_prob.cpu(),
            probs.cpu(),
            new_hidden.cpu()
        )

    def calculate_int_rewards(self, next_obs, batch=True):
        """
        Compute the RND intrinsic reward for each next observation:
        the mean squared error between predictor and target features.
        Returns a NumPy array of shape (batch_size,).
        """
        if not batch:
            next_obs = np.expand_dims(next_obs, 0)

        # normalize observations
        norm = (next_obs - self.state_rms.mean) / np.sqrt(self.state_rms.var + 1e-8)
        norm = np.clip(norm, -5, 5).astype(np.float32)
        obs_tensor = torch.from_numpy(norm).to(self.device)

        with torch.no_grad():
            pred_feat = self.predictor_model(obs_tensor)
            targ_feat = self.target_model(obs_tensor)

        # per-sample MSE over feature dims
        mse = (pred_feat - targ_feat).pow(2).mean(dim=1)
        return mse.cpu().numpy()

    def normalize_int_rewards(self, int_rewards):
        gamma = self.config["int_gamma"]
        returns = []
        for seq in int_rewards:
            discounted, acc = [], 0.0
            for r in reversed(seq):
                acc = r + gamma * acc
                discounted.insert(0, acc)
            returns.append(discounted)

        flat = np.array(returns).reshape(-1, 1)
        self.int_reward_rms.update(flat)
        return int_rewards / (np.sqrt(self.int_reward_rms.var) + 1e-8)

    def get_gae(self, rewards, values, next_values, dones, gamma):
        lam = self.config["lambda"]
        advantages = []
        for r, v, nv, d in zip(rewards, values, next_values, dones):
            gae_seq = []
            gae = 0.0
            for t in reversed(range(len(r))):
                delta = r[t] + gamma * nv[t] * (1 - d[t]) - v[t]
                gae = delta + gamma * lam * (1 - d[t]) * gae
                gae_seq.insert(0, gae)
            advantages.append(gae_seq)
        return np.array(advantages)

    @mean_of_list
    def train(
        self, states, actions, int_rewards, ext_rewards, dones,
        int_values, ext_values, log_probs, next_int_values,
        next_ext_values, total_next_obs, hidden_states
    ):
        # --- Compute returns & advantages ---
        int_ret = self.get_gae(
            [int_rewards], [int_values], [next_int_values],
            [np.zeros_like(dones)], self.config["int_gamma"]
        )[0]
        ext_ret = self.get_gae(
            [ext_rewards], [ext_values], [next_ext_values],
            [dones], self.config["ext_gamma"]
        )[0]

        advs = (
            (ext_ret - ext_values) * self.config["ext_adv_coeff"] +
            (int_ret - int_values) * self.config["int_adv_coeff"]
        )
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        ext_ret = torch.tensor(ext_ret, dtype=torch.float32, device=self.device)
        int_ret = torch.tensor(int_ret, dtype=torch.float32, device=self.device)

        # --- Prepare tensors ---
        states        = states.to(self.device)
        actions       = actions.to(self.device)
        log_probs     = log_probs.to(self.device)
        hidden_states = hidden_states.to(self.device)
        next_obs      = torch.tensor(total_next_obs, dtype=torch.float32).to(self.device)

        pg_losses, ext_v_losses, int_v_losses, rnd_losses, entropies = [], [], [], [], []

        for _ in range(self.config["n_epochs"]):
            dist, int_v, ext_v, _, _ = self.current_policy(states, hidden_states)
            entropy = dist.entropy().mean()
            new_logp = dist.log_prob(actions)
            ratio = (new_logp - log_probs).exp()

            # PPO policy loss
            surr1 = ratio * advs
            surr2 = torch.clamp(
                ratio,
                1 - self.config["clip_range"],
                1 + self.config["clip_range"]
            ) * advs
            pg_loss = -torch.min(surr1, surr2).mean()

            # Critic losses
            loss_ext = self.mse_loss(ext_v.squeeze(), ext_ret)
            loss_int = self.mse_loss(int_v.squeeze(), int_ret)
            critic_loss = 0.5 * (loss_ext + loss_int)

            # RND loss
            rnd_loss = self.calculate_rnd_loss(next_obs)

            # Total loss & backprop
            total_loss = pg_loss + critic_loss + rnd_loss - self.config["ent_coeff"] * entropy
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.current_policy.parameters()) + list(self.predictor_model.parameters()),
                self.config["max_grad_norm"]
            )
            self.optimizer.step()

            # Logging
            pg_losses.append(pg_loss.item())
            ext_v_losses.append(loss_ext.item())
            int_v_losses.append(loss_int.item())
            rnd_losses.append(rnd_loss.item())
            entropies.append(entropy.item())

        return (
            pg_losses,
            ext_v_losses,
            int_v_losses,
            rnd_losses,
            entropies,
            int_values,
            int_ret,
            ext_values,
            ext_ret
        )

    def calculate_rnd_loss(self, obs):
        """
        Compute masked RND loss.
        Clamps predictor_proportion into [0,1] before sampling.
        """
        # 1) Feature extraction
        pred_feat = self.predictor_model(obs)
        with torch.no_grad():
            targ_feat = self.target_model(obs)

        # 2) Per-sample MSE
        mse = (pred_feat - targ_feat).pow(2).mean(dim=1)

        # 3) Clamp probability
        p = float(self.config.get("predictor_proportion", 1.0))
        p = max(0.0, min(1.0, p))

        # 4) Mask and avoid zero division
        probs = torch.full_like(mse, p, device=self.device)
        mask = torch.bernoulli(probs)
        valid = mask.sum().clamp(min=1.0)

        # 5) Final masked loss
        rnd_loss = (mse * mask).sum() / valid
        return rnd_loss

    def set_from_checkpoint(self, checkpoint):
        self.current_policy.load_state_dict(checkpoint["current_policy_state_dict"])
        self.predictor_model.load_state_dict(checkpoint["predictor_model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state_rms.mean  = checkpoint["state_rms_mean"]
        self.state_rms.var   = checkpoint["state_rms_var"]
        self.state_rms.count = checkpoint["state_rms_count"]
        self.int_reward_rms.mean  = checkpoint["int_reward_rms_mean"]
        self.int_reward_rms.var   = checkpoint["int_reward_rms_var"]
        self.int_reward_rms.count = checkpoint["int_reward_rms_count"]

    def set_to_eval_mode(self):
        self.current_policy.eval()
