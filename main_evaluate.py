import joblib
from src.hmm_trainer import HMMTrainer
from hangman_env import HangmanEnv

# Load model
trainer = HMMTrainer()

# Initialize game environment
env = HangmanEnv()

# Run 2000 games
total_games = 2000
success, wrong, repeated = 0, 0, 0

for i in range(total_games):
    word = env.reset()
    done = False
    while not done:
        prefix = env.get_masked_prefix()
        probs = trainer.next_letter_probs(prefix)
        guess = max(probs, key=probs.get)
        result = env.step(guess)
        if result == "repeated":
            repeated += 1
        elif result == "wrong":
            wrong += 1
        elif result == "win":
            success += 1
            done = True

# Final Score
final_score = (success * 2000) - (wrong * 5) - (repeated * 2)
print("Success:", success, "Wrong:", wrong, "Repeated:", repeated)
print("Final Score:", final_score)
