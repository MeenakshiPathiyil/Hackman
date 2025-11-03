import os
import random
import numpy as np
from src.agent import HangmanDQNAgent
from src.hangman_env import HangmanEnv
from src.hmm_oracle import HMMOracle
from src.hmm_trainer import HMMTrainer
import torch

with open("data/corpus.txt", "r") as f:
    all_words = [line.strip().upper() for line in f if line.strip()]

train_words = all_words[:40_000]
test_words = all_words[40_000:]

print(f"Training on {len(train_words)} words")
print(f"Testing on {len(test_words)} words")

# Ensure HMM model exists; train if missing
hmm_model_path = "models/hmm_model.pkl"
if not os.path.exists(hmm_model_path):
    print("No HMM model found — training one from corpus...")
    trainer = HMMTrainer(hmm_model_path)
    trainer.train("data/corpus.txt", n_states=30, min_word_len=2)
    print(f"HMM trained and saved to {hmm_model_path}")

hmm_oracle = HMMOracle(hmm_model_path)

def get_hmm_probs(mask, guessed):
    return hmm_oracle.predict(mask, guessed)

agent = HangmanDQNAgent(
    max_word_len=20,
    alpha=0.001,
    gamma=0.95,
    epsilon_start=0.8,
    epsilon_min=0.15,
    epsilon_decay=0.999,
    batch_size=64
)

agent.load()
print("Loaded DQN model." if os.path.exists("models/dqn_agent.pt") else "No saved DQN model found. Starting fresh.")

def play_game(env: HangmanEnv, agent, hmm):
    """Evaluation helper: HMM-greedy policy (aligns with main_evaluate)."""
    env.reset()
    while not env.done:
        mask = env.get_masked_word()
        probs = hmm.predict(mask, env.guessed)
        available = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in env.guessed]
        if not available:
            break
        avail_idx = [ord(c) - ord('a') for c in available]
        best_idx_local = avail_idx[int(np.argmax(probs[avail_idx]))]
        guess = chr(best_idx_local + ord('a'))
        env.step(guess)
    return "_" not in env.get_masked_word()

print("\n=== TRAINING STARTED ===")
for episode in range(1, 2001):
    word = random.choice(train_words)

    env = HangmanEnv(word=word.lower(), max_lives=6)
    env.reset()

    while not env.done:
        mask = env.get_masked_word()
        guessed = env.guessed
        lives = env.lives_left

        probs = get_hmm_probs(mask, guessed)
        state = agent.get_state_vector(mask, guessed, probs, lives)
        action = agent.choose_action(state, guessed, probs)
        if action is None:
            break
        guess = agent.idx_to_letter[action]

        next_mask, reward, done, _ = env.step(guess)

        next_probs = get_hmm_probs(next_mask, env.guessed)
        next_state = agent.get_state_vector(next_mask, env.guessed, next_probs, env.lives_left)
        next_avail = [i for i in range(26) if agent.idx_to_letter[i] not in env.guessed]
        
        agent.update(state, action, reward, next_state, done, next_avail)

    agent.decay_epsilon()

    if episode % 200 == 0:
        wins = 0
        for _ in range(50):
            test_word = random.choice(test_words)
            test_env = HangmanEnv(word=test_word.lower(), max_lives=6)
            wins += play_game(test_env, agent, hmm_oracle)
        win_rate = wins / 50
        print(
            f"Ep {episode:4d} | ε={agent.epsilon:.3f} | "
            f"Win-rate (50 games) = {win_rate:.1%}"
        )

os.makedirs("models", exist_ok=True)
agent.save("models/dqn_agent.pt")
print(f"\nTRAINING COMPLETE")
print(f"   Final ε        : {agent.epsilon:.3f}")
print(f"   Model saved to : models/dqn_agent.pt")




# # main_train.py
# # FINAL RL + HMM Training (5000 episodes) – optimized for better learning
# import os
# import random
# import numpy as np
# from src.agent import HangmanTabularAgent
# from src.hangman_env import HangmanEnv
# from src.hmm_oracle import HMMOracle

# # =====================================================
# # 1. Load word list
# # =====================================================
# with open("data/corpus.txt", "r") as f:
#     all_words = [line.strip().upper() for line in f if line.strip()]

# train_words = all_words[:40_000]
# test_words = all_words[40_000:]

# print(f"Training on {len(train_words)} words")
# print(f"Testing on {len(test_words)} words")

# # =====================================================
# # 2. HMM Oracle
# # =====================================================
# hmm_oracle = HMMOracle("models/hmm_model.pkl")

# def get_hmm_probs(mask, guessed):
#     return hmm_oracle.predict(mask, guessed)

# # =====================================================
# # 3. Initialize Agent
# # =====================================================
# agent = HangmanTabularAgent(
#     alpha=0.1,
#     gamma=0.95,
#     epsilon_start=0.7,
#     epsilon_min=0.1,  # Increased to maintain exploration
#     epsilon_decay=0.999  # Slower decay
# )

# agent.load()
# print(f"Starting with {len(agent.q_table)} states")

# # =====================================================
# # 4. Helper: Play one game (for win-rate)
# # =====================================================
# def play_game(env: HangmanEnv, agent, hmm):
#     env.reset()
#     while not env.done:
#         mask = env.get_masked_word()
#         probs = hmm.predict(mask, env.guessed)
#         key = agent.get_state_key(mask, env.guessed, probs, env.lives_left)
#         action = agent.choose_action(key, env.guessed)
#         if action is None:
#             break
#         guess = agent.idx_to_letter[action]
#         env.step(guess)
#     return "_" not in env.get_masked_word()

# # =====================================================
# # 5. TRAINING LOOP (5000 episodes)
# # =====================================================
# print("\n=== TRAINING STARTED ===")
# for episode in range(1, 5001):  # Extended to 5000 episodes
#     word = random.choice(train_words)

#     # NEW ENV PER EPISODE → no state leak
#     env = HangmanEnv(word=word.lower(), max_lives=6)
#     env.reset()

#     while not env.done:
#         mask = env.get_masked_word()
#         guessed = env.guessed
#         lives = env.lives_left

#         # ---------- HMM suggestion ----------
#         probs = get_hmm_probs(mask, guessed)

#         # ---------- Agent decision ----------
#         state_key = agent.get_state_key(mask, guessed, probs, lives)
#         action = agent.choose_action(state_key, guessed)
#         if action is None:
#             break
#         guess = agent.idx_to_letter[action]

#         # ---------- Execute step ----------
#         next_mask, reward, done, _ = env.step(guess)

#         # ---------- Q-update ----------
#         next_probs = get_hmm_probs(next_mask, env.guessed)
#         next_key = agent.get_state_key(
#             next_mask, env.guessed, next_probs, env.lives_left
#         )
#         next_avail = [
#             i for i in range(26) if agent.idx_to_letter[i] not in env.guessed
#         ]
#         agent.update(state_key, action, reward, next_key, next_avail)

#     agent.decay_epsilon()

#     # ---------- Win-rate every 200 episodes ----------
#     if episode % 200 == 0:
#         wins = 0
#         for _ in range(50):
#             test_word = random.choice(train_words)
#             test_env = HangmanEnv(word=test_word.lower(), max_lives=6)
#             wins += play_game(test_env, agent, hmm_oracle)
#         win_rate = wins / 50
#         print(
#             f"Ep {episode:4d} | ε={agent.epsilon:.3f} | "
#             f"Win-rate (50 games) = {win_rate:.1%} | States = {len(agent.q_table):,}"
#         )

# # =====================================================
# # 6. SAVE FINAL MODEL
# # =====================================================
# os.makedirs("models", exist_ok=True)
# agent.save("models/rl_agent.h5")
# print(f"\nTRAINING COMPLETE")
# print(f"   States learned : {len(agent.q_table):,}")
# print(f"   Final ε        : {agent.epsilon:.3f}")
# print(f"   Model saved to : models/rl_agent.h5")


# # # # main_train.py
# # # # FINAL RL + HMM Training (1200 episodes) – debug print fixed
# # # import os
# # # import random
# # # import numpy as np
# # # from src.agent import HangmanTabularAgent
# # # from src.hangman_env import HangmanEnv
# # # from src.hmm_oracle import HMMOracle

# # # # =====================================================
# # # # 1. Load word list
# # # # =====================================================
# # # with open("data/corpus.txt", "r") as f:
# # #     all_words = [line.strip().upper() for line in f if line.strip()]

# # # train_words = all_words[:40_000]
# # # test_words  = all_words[40_000:]

# # # print(f"Training on {len(train_words)} words")
# # # print(f"Testing  on {len(test_words)} words")

# # # # =====================================================
# # # # 2. HMM Oracle
# # # # =====================================================
# # # hmm_oracle = HMMOracle("models/hmm_model.pkl")

# # # def get_hmm_probs(mask, guessed):
# # #     return hmm_oracle.predict(mask, guessed)

# # # # =====================================================
# # # # 3. Initialize Agent
# # # # =====================================================
# # # agent = HangmanTabularAgent(
# # #     alpha=0.1,
# # #     gamma=0.95,
# # #     epsilon_start=0.7,
# # #     epsilon_min=0.05,
# # #     epsilon_decay=0.995
# # # )

# # # agent.load()
# # # print(f"Starting with {len(agent.q_table)} states")

# # # # =====================================================
# # # # 4. Helper: Play one game (for win-rate)
# # # # =====================================================
# # # def play_game(env: HangmanEnv, agent, hmm):
# # #     env.reset()
# # #     while not env.done:
# # #         mask = env.get_masked_word()
# # #         probs = hmm.predict(mask, env.guessed)
# # #         key = agent.get_state_key(mask, env.guessed, probs, env.lives_left)
# # #         action = agent.choose_action(key, env.guessed)
# # #         if action is None:
# # #             break
# # #         guess = agent.idx_to_letter[action]
# # #         env.step(guess)
# # #     return "_" not in env.get_masked_word()

# # # # =====================================================
# # # # 5. TRAINING LOOP (1200 episodes)
# # # # =====================================================
# # # print("\n=== TRAINING STARTED ===")
# # # for episode in range(1, 1201):                     # ← change to 1201 for full run
# # #     word = random.choice(train_words)

# # #     # NEW ENV PER EPISODE → no state leak
# # #     env = HangmanEnv(word=word.lower(), max_lives=6)
# # #     env.reset()

# # #     print(f"\n--- EPISODE {episode} – WORD = {word} ---")

# # #     while not env.done:
# # #         mask   = env.get_masked_word()
# # #         guessed = env.guessed
# # #         lives  = env.lives_left

# # #         # ---------- HMM suggestion ----------
# # #         probs = get_hmm_probs(mask, guessed)
# # #         hmm_top = chr(97 + np.argmax(probs))

# # #         # ---------- Agent decision ----------
# # #         state_key = agent.get_state_key(mask, guessed, probs, lives)
# # #         action = agent.choose_action(state_key, guessed)
# # #         if action is None:
# # #             print("  (no legal action – break)")
# # #             break
# # #         guess = agent.idx_to_letter[action]

# # #         # ---------- Execute step ----------
# # #         next_mask, reward, done, _ = env.step(guess)

# # #         # ---------- DEBUG PRINT ----------
# # #         print(
# # #             f"  → {guess.upper():1} | R:{reward:+3} | "
# # #             f"{mask} → {next_mask} | Lives:{lives} | "
# # #             f"HMM top: {hmm_top}"
# # #         )

# # #         # ---------- Q-update ----------
# # #         next_probs = get_hmm_probs(next_mask, env.guessed)
# # #         next_key = agent.get_state_key(
# # #             next_mask, env.guessed, next_probs, env.lives_left
# # #         )
# # #         next_avail = [
# # #             i for i in range(26) if agent.idx_to_letter[i] not in env.guessed
# # #         ]
# # #         agent.update(state_key, action, reward, next_key, next_avail)

# # #     agent.decay_epsilon()

# # #     # ---------- Win-rate every 200 episodes ----------
# # #     if episode % 200 == 0:
# # #         wins = 0
# # #         for _ in range(50):
# # #             test_word = random.choice(train_words)
# # #             test_env = HangmanEnv(word=test_word.lower(), max_lives=6)
# # #             wins += play_game(test_env, agent, hmm_oracle)
# # #         win_rate = wins / 50
# # #         print(
# # #             f"Ep {episode:4d} | ε={agent.epsilon:.3f} | "
# # #             f"Win-rate={win_rate:.1%} | States={len(agent.q_table):,}"
# # #         )

# # # # =====================================================
# # # # 6. SAVE FINAL MODEL
# # # # =====================================================
# # # os.makedirs("models", exist_ok=True)
# # # agent.save("models/rl_agent.h5")
# # # print(f"\nTRAINING COMPLETE")
# # # print(f"   States learned : {len(agent.q_table):,}")
# # # print(f"   Final ε        : {agent.epsilon:.3f}")
# # # print(f"   Model saved to : models/rl_agent.h5")


# # import os
# # import random
# # import numpy as np
# # from src.agent import HangmanTabularAgent
# # from src.hangman_env import HangmanEnv
# # from src.hmm_oracle import HMMOracle

# # with open("data/corpus.txt", "r") as f:
# #     all_words = [line.strip().upper() for line in f if line.strip()]

# # train_words = all_words[:40_000]   # 80 %
# # test_words  = all_words[40_000:]   # 20 %

# # print(f"Training on {len(train_words)} words")
# # print(f"Testing  on {len(test_words)} words")

# # hmm_oracle = HMMOracle("models/hmm_model.pkl")

# # # # ---- MOCK HMM (fallback) ---------------------------------------
# # # class MockHMMOracle:
# # #     def predict(self, masked_word, guessed_letters):
# # #         probs = np.zeros(26)
# # #         common = ['E','T','A','O','I','N','S','H','R','D','L','U']
# # #         for c in common:
# # #             idx = ord(c) - 65
# # #             if c not in guessed_letters:
# # #                 probs[idx] = 1.0
# # #         probs += np.random.random(26) * 0.05
# # #         probs /= probs.sum() + 1e-8
# # #         return probs

# # # hmm_oracle = MockHMMOracle()
# # # # ------------------------------------------------------------------

# # def get_hmm_probs(mask, guessed):
# #     """Convenient wrapper – works with both mock & real oracle."""
# #     return hmm_oracle.predict(mask, guessed)

# # env = HangmanEnv(word="dummy", max_lives=6)

# # agent = HangmanTabularAgent(
# #     alpha=0.1,
# #     gamma=0.95,
# #     epsilon_start=0.7,
# #     epsilon_min=0.05,
# #     epsilon_decay=0.995
# # )

# # # Load a previously saved Q-table (if any)
# # agent.load()               
# # print(f"Starting with {len(agent.q_table)} states")

# # def play_one_game(word: str) -> bool:
# #     """Return True if the agent wins the given word."""
# #     env.word = word.lower()
# #     env.reset()
# #     while not env.done:
# #         mask = env.get_masked_word()
# #         probs = get_hmm_probs(mask, env.guessed)
# #         key = agent.get_state_key(mask, env.guessed, probs, env.lives_left)
# #         action = agent.choose_action(key, env.guessed)
# #         if action is None:
# #             break
# #         guess = agent.idx_to_letter[action]
# #         env.step(guess)
# #     return "_" not in env.get_masked_word()

# # print("\n=== TRAINING STARTED ===")
# # for episode in range(1, 1201):           
# #     word = random.choice(train_words)
# #     env.word = word.lower()
# #     env.reset()

# #     while not env.done:
# #         mask   = env.get_masked_word()
# #         guessed = env.guessed
# #         lives  = env.lives_left

# #         probs = get_hmm_probs(mask, guessed)
# #         state_key = agent.get_state_key(mask, guessed, probs, lives)

# #         action = agent.choose_action(state_key, guessed)
# #         if action is None:
# #             break
# #         guess = agent.idx_to_letter[action]

# #         next_mask, reward, done, _ = env.step(guess)

# #         next_probs = get_hmm_probs(next_mask, env.guessed)
# #         next_key   = agent.get_state_key(next_mask, env.guessed, next_probs, env.lives_left)
# #         next_avail = [i for i in range(26) if agent.idx_to_letter[i] not in env.guessed]
# #         agent.update(state_key, action, reward, next_key, next_avail)

# #     agent.decay_epsilon()

# #     if episode % 200 == 0:
# #         win_count = sum(1 for _ in range(50) if play_one_game(random.choice(train_words)))
# #         win_rate  = win_count / 50
# #         print(f"Ep {episode:4d} | ε={agent.epsilon:.3f} | "
# #               f"Win-rate (50 games) = {win_rate:.1%} | "
# #               f"States = {len(agent.q_table):,}")

# # os.makedirs("models", exist_ok=True)
# # agent.save("models/rl_agent.h5")
# # print("\nTRAINING COMPLETE")
# # print(f"   States learned : {len(agent.q_table):,}")
# # print(f"   Final ε        : {agent.epsilon:.3f}")
# # print(f"   Model saved to : models/rl_agent.h5")