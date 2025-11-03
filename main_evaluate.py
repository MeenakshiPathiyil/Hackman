import os
import random
import numpy as np
from src.agent import HangmanDQNAgent
from src.hangman_env import HangmanEnv
from src.hmm_oracle import HMMOracle
from src.hmm_trainer import HMMTrainer
import torch

with open("data/test.txt", "r") as f:
    all_words = [line.strip().upper() for line in f if line.strip()]
test_words = all_words
print(f"Evaluating on {len(test_words)} test words")

hmm_model_path = "models/hmm_model.pkl"
if not os.path.exists(hmm_model_path):
    print("No HMM model found — training one from corpus for evaluation...")
    trainer = HMMTrainer(hmm_model_path)
    trainer.train("data/corpus.txt", n_states=30, min_word_len=2)
    print(f"HMM trained and saved to {hmm_model_path}")
hmm_oracle = HMMOracle(hmm_model_path)

def get_hmm_probs(mask, guessed):
    return hmm_oracle.predict(mask, guessed)


agent = HangmanDQNAgent(
    max_word_len=20,
    alpha=0.0005,
    gamma=0.95,
    epsilon_start=0.0,
    epsilon_min=0.0,
    epsilon_decay=0.999,
    batch_size=128
)
agent.load("models/dqn_agent.pt")
print("Loaded DQN model")


def play_game(env: HangmanEnv, agent, hmm):
    # Reset the environment to start a new game
    env.reset()
    wrong_guesses = 0
    repeated_guesses = 0
    # Continue playing until the game is over
    while not env.done:
        # Get the current masked word
        mask = env.get_masked_word()
        probs = get_hmm_probs(mask, env.guessed)
        # Find all letters that have not been guessed yet
        available = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in env.guessed]
        if not available:
            break
        avail_idx = [ord(c) - ord('a') for c in available]

        state = agent.get_state_vector(mask, env.guessed, probs, env.lives_left)
        with torch.no_grad():
            q_values = agent.model(state).detach().cpu().numpy()
        q_avail = q_values[avail_idx]
        q_avail = q_avail - np.max(q_avail)
        q_soft = np.exp(q_avail / 1.5)
        q_soft = q_soft / (q_soft.sum() + 1e-12)
        hmm_avail = probs[avail_idx]
        hmm_avail = hmm_avail / (hmm_avail.sum() + 1e-12)

        # Blend DQN and HMM probabilities
        alpha = 0.6
        blend = alpha * q_soft + (1.0 - alpha) * hmm_avail

        # Choose the letter with the highest combined probability
        chosen_local = int(np.argmax(blend))
        best_idx_local = avail_idx[chosen_local]
        guess = chr(best_idx_local + ord('a'))

        # Play the chosen letter in the environment
        prev_mask = env.get_masked_word()
        next_mask, reward, done, _ = env.step(guess)
        base_reward = reward + 2 

        # Track wrong and repeated guesses
        if base_reward == -10:
            repeated_guesses += 1
        elif base_reward in [-15, -100]:
            wrong_guesses += 1

    # Check if the agent successfully revealed the whole word
    won = "_" not in env.get_masked_word()
    return won, wrong_guesses, repeated_guesses


wins = 0
total_wrong = 0
total_repeated = 0
for i in range(2000):
    test_word = random.choice(test_words)
    test_env = HangmanEnv(word=test_word.lower(), max_lives=6)
    won, wrong, repeated = play_game(test_env, agent, hmm_oracle)
    wins += won
    total_wrong += wrong
    total_repeated += repeated


success_rate = wins / 2000
final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
print(f"\nEVALUATION RESULTS (2000 games)")
print(f"Success Rate: {success_rate:.1%}")
print(f"Total Wrong Guesses: {total_wrong}")
print(f"Total Repeated Guesses: {total_repeated}")
print(f"Final Score: {final_score:.1f}")






# import os
# import random
# import numpy as np
# from src.agent import HangmanDQNAgent
# from src.hangman_env import HangmanEnv
# from src.hmm_oracle import HMMOracle

# with open("data/corpus.txt", "r") as f:
#     all_words = [line.strip().upper() for line in f if line.strip()]
# test_words = all_words[40_000:]
# print(f"Evaluating on {len(test_words)} test words")

# USE_HMM = True  
# if USE_HMM:
#     hmm_oracle = HMMOracle("models/hmm_model.pkl")
#     def get_hmm_probs(mask, guessed):
#         return hmm_oracle.predict(mask, guessed)
# else:
#     from collections import defaultdict
#     letter_freq = defaultdict(int)
#     for word in test_words:
#         for c in word.lower():
#             letter_freq[c] += 1
#     total = sum(letter_freq.values())
#     letter_probs = np.zeros(26)
#     for c in 'abcdefghijklmnopqrstuvwxyz':
#         letter_probs[ord(c) - ord('a')] = letter_freq.get(c, 0) / total
#     def get_hmm_probs(mask, guessed):
#         probs = letter_probs.copy()
#         for c in guessed:
#             probs[ord(c) - ord('a')] = 0
#         return probs / (probs.sum() + 1e-12) if probs.sum() > 0 else np.ones(26) / 26

# agent = HangmanDQNAgent(
#     max_word_len=20,
#     alpha=0.0005,
#     gamma=0.95,
#     epsilon_start=0.15,
#     epsilon_min=0.15,
#     epsilon_decay=0.999,
#     batch_size=128
# )
# agent.load("models/dqn_agent.pt")
# print("Loaded DQN model")

# def play_game(env: HangmanEnv, agent, hmm):
#     env.reset()
#     wrong_guesses = 0
#     repeated_guesses = 0
#     while not env.done:
#         mask = env.get_masked_word()
#         probs = get_hmm_probs(mask, env.guessed)
#         state = agent.get_state_vector(mask, env.guessed, probs, env.lives_left)
#         action = agent.choose_action(state, env.guessed, probs)
#         if action is None:
#             break
#         guess = agent.idx_to_letter[action]
#         prev_mask = env.get_masked_word()
#         next_mask, reward, done, _ = env.step(guess)
#         if reward == -10 or (reward < -2 and prev_mask == next_mask and not done):  # Repeated guess
#             repeated_guesses += 1
#         elif reward in [-15, -100] or (reward < -2 and prev_mask == next_mask and done):  # Wrong guess
#             wrong_guesses += 1
#     won = "_" not in env.get_masked_word()
#     return won, wrong_guesses, repeated_guesses

# wins = 0
# total_wrong = 0
# total_repeated = 0
# for i in range(2000):
#     test_word = random.choice(test_words)
#     test_env = HangmanEnv(word=test_word.lower(), max_lives=6)
#     won, wrong, repeated = play_game(test_env, agent, hmm_oracle if USE_HMM else None)
#     wins += won
#     total_wrong += wrong
#     total_repeated += repeated

# success_rate = wins / 2000
# final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
# print(f"\nEVALUATION RESULTS (2000 games)")
# print(f"Success Rate: {success_rate:.1%}")
# print(f"Total Wrong Guesses: {total_wrong}")
# print(f"Total Repeated Guesses: {total_repeated}")
# print(f"Final Score: {final_score:.1f}")







# # evaluate.py
# # Evaluates the DQN agent on the test set
# import os
# import random
# import numpy as np
# from src.agent import HangmanDQNAgent
# from src.hangman_env import HangmanEnv
# from src.hmm_oracle import HMMOracle

# # Load word list
# with open("data/corpus.txt", "r") as f:
#     all_words = [line.strip().upper() for line in f if line.strip()]
# test_words = all_words[40_000:]
# print(f"Evaluating on {len(test_words)} test words")

# # Initialize HMM or frequency baseline
# USE_HMM = True  # Set to False to use frequency baseline
# if USE_HMM:
#     hmm_oracle = HMMOracle("models/hmm_model.pkl")
#     def get_hmm_probs(mask, guessed):
#         return hmm_oracle.predict(mask, guessed)
# else:
#     from collections import defaultdict
#     letter_freq = defaultdict(int)
#     for word in test_words:
#         for c in word.lower():
#             letter_freq[c] += 1
#     total = sum(letter_freq.values())
#     letter_probs = np.zeros(26)
#     for c in 'abcdefghijklmnopqrstuvwxyz':
#         letter_probs[ord(c) - ord('a')] = letter_freq.get(c, 0) / total
#     def get_hmm_probs(mask, guessed):
#         probs = letter_probs.copy()
#         for c in guessed:
#             probs[ord(c) - ord('a')] = 0
#         return probs / (probs.sum() + 1e-12) if probs.sum() > 0 else np.ones(26) / 26

# # Initialize agent
# agent = HangmanDQNAgent(
#     max_word_len=20,  # Ensure consistency with training
#     alpha=0.0005,
#     gamma=0.95,
#     epsilon_start=0.15,
#     epsilon_min=0.15,
#     epsilon_decay=0.999,
#     batch_size=128
# )
# agent.load("models/dqn_agent.pt")
# print("Loaded DQN model")

# # Play one game and track metrics
# def play_game(env: HangmanEnv, agent, hmm):
#     env.reset()
#     wrong_guesses = 0
#     repeated_guesses = 0
#     while not env.done:
#         mask = env.get_masked_word()
#         probs = get_hmm_probs(mask, env.guessed)
#         state = agent.get_state_vector(mask, env.guessed, probs, env.lives_left)
#         action = agent.choose_action(state, env.guessed, probs)
#         if action is None:
#             break
#         guess = agent.idx_to_letter[action]
#         prev_mask = env.get_masked_word()
#         next_mask, reward, done, _ = env.step(guess)
#         if reward == -10:
#             repeated_guesses += 1
#         elif reward == -15 or reward == -100:
#             wrong_guesses += 1
#     won = "_" not in env.get_masked_word()
#     return won, wrong_guesses, repeated_guesses

# # Evaluate on 2000 games
# wins = 0
# total_wrong = 0
# total_repeated = 0
# for i in range(2000):
#     test_word = random.choice(test_words)
#     test_env = HangmanEnv(word=test_word.lower(), max_lives=6)
#     won, wrong, repeated = play_game(test_env, agent, hmm_oracle if USE_HMM else None)
#     wins += won
#     total_wrong += wrong
#     total_repeated += repeated

# # Compute final score
# success_rate = wins / 2000
# final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
# print(f"\nEVALUATION RESULTS (2000 games)")
# print(f"Success Rate: {success_rate:.1%}")
# print(f"Total Wrong Guesses: {total_wrong}")
# print(f"Total Repeated Guesses: {total_repeated}")
# print(f"Final Score: {final_score:.1f}")



# # evaluate.py
# # Evaluates the DQN agent on the test set
# import os
# import random
# import numpy as np
# from src.agent import HangmanDQNAgent
# from src.hangman_env import HangmanEnv
# from src.hmm_oracle import HMMOracle

# # Load word list
# with open("data/corpus.txt", "r") as f:
#     all_words = [line.strip().upper() for line in f if line.strip()]
# test_words = all_words[40_000:]
# print(f"Evaluating on {len(test_words)} test words")

# # Initialize HMM or frequency baseline
# USE_HMM = True  # Set to False to use frequency baseline
# if USE_HMM:
#     hmm_oracle = HMMOracle("models/hmm_model.pkl")
#     def get_hmm_probs(mask, guessed):
#         return hmm_oracle.predict(mask, guessed)
# else:
#     from collections import defaultdict
#     letter_freq = defaultdict(int)
#     for word in test_words:
#         for c in word.lower():
#             letter_freq[c] += 1
#     total = sum(letter_freq.values())
#     letter_probs = np.zeros(26)
#     for c in 'abcdefghijklmnopqrstuvwxyz':
#         letter_probs[ord(c) - ord('a')] = letter_freq.get(c, 0) / total
#     def get_hmm_probs(mask, guessed):
#         probs = letter_probs.copy()
#         for c in guessed:
#             probs[ord(c) - ord('a')] = 0
#         return probs / (probs.sum() + 1e-12) if probs.sum() > 0 else np.ones(26) / 26

# # Initialize agent
# agent = HangmanDQNAgent(
#     max_word_len=20,
#     alpha=0.0005,
#     gamma=0.95,
#     epsilon_start=0.15,
#     epsilon_min=0.15,
#     epsilon_decay=0.999,
#     batch_size=128
# )
# agent.load("models/dqn_agent.pt")
# print("Loaded DQN model")

# # Play one game and track metrics
# def play_game(env: HangmanEnv, agent, hmm):
#     env.reset()
#     wrong_guesses = 0
#     repeated_guesses = 0
#     while not env.done:
#         mask = env.get_masked_word()
#         probs = get_hmm_probs(mask, env.guessed)
#         state = agent.get_state_vector(mask, env.guessed, probs, env.lives_left)
#         action = agent.choose_action(state, env.guessed, probs)
#         if action is None:
#             break
#         guess = agent.idx_to_letter[action]
#         prev_mask = env.get_masked_word()
#         next_mask, reward, done, _ = env.step(guess)
#         if reward == -10:
#             repeated_guesses += 1
#         elif reward == -15 or reward == -100:
#             wrong_guesses += 1
#     won = "_" not in env.get_masked_word()
#     return won, wrong_guesses, repeated_guesses

# # Evaluate on 2000 games
# wins = 0
# total_wrong = 0
# total_repeated = 0
# for i in range(2000):
#     test_word = random.choice(test_words)
#     test_env = HangmanEnv(word=test_word.lower(), max_lives=6)
#     won, wrong, repeated = play_game(test_env, agent, hmm_oracle if USE_HMM else None)
#     wins += won
#     total_wrong += wrong
#     total_repeated += repeated

# # Compute final score
# success_rate = wins / 2000
# final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
# print(f"\nEVALUATION RESULTS (2000 games)")
# print(f"Success Rate: {success_rate:.1%}")
# print(f"Total Wrong Guesses: {total_wrong}")
# print(f"Total Repeated Guesses: {total_repeated}")
# print(f"Final Score: {final_score:.1f}")





# # import joblib
# # from src.hmm_trainer import HMMTrainer
# # from src.hangman_env import HangmanEnv

# # trainer = HMMTrainer()
# # env = HangmanEnv()


# # # Run 2000 games
# # total_games = 2000
# # success, wrong, repeated = 0, 0, 0

# # for i in range(total_games):
# #     word = env.reset()
# #     done = False
# #     while not done:
# #         prefix = env.get_masked_prefix()
# #         probs = trainer.next_letter_probs(prefix)
# #         guess = max(probs, key=probs.get)
# #         result = env.step(guess)
# #         if result == "repeated":
# #             repeated += 1
# #         elif result == "wrong":
# #             wrong += 1
# #         elif result == "win":
# #             success += 1
# #             done = True

# # # Final Score
# # final_score = (success * 2000) - (wrong * 5) - (repeated * 2)
# # print("Success:", success, "Wrong:", wrong, "Repeated:", repeated)
# # print("Final Score:", final_score)
