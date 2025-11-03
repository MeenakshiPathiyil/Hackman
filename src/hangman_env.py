# src/hangman_env.py
import numpy as np

class HangmanEnv:
    """
    Reinforcement-learning environment for Hangman.
    Handles word masking, guesses, rewards, and game termination.
    """

    def __init__(self, word: str, max_lives: int = 6):
        self.word = word.lower()
        self.max_lives = max_lives
        self.reset()

    # --- Core API ---
    def reset(self):
        """Resets environment for a new game."""
        self.guessed = set()
        self.lives_left = self.max_lives
        self.done = False
        self.reward = 0
        return self.get_masked_word()

    def get_masked_word(self):
        """Returns masked string like '_pp_e'."""
        return "".join(c if c in self.guessed else "_" for c in self.word)

    def step(self, guess: str):
        """
        Executes one action (letter guess).
        Returns (new_mask, reward, done, info)
        """
        if self.done:
            return self.get_masked_word(), 0, True, {"reason": "finished"}

        guess = guess.lower()
        info = {}

        # repeated guess penalty
        if guess in self.guessed:
            self.reward = -5
            info["reason"] = "repeat"
            return self.get_masked_word(), self.reward, False, info

        self.guessed.add(guess)

        # correct guess
        if guess in self.word:
            occurrences = self.word.count(guess)
            self.reward = 10 * occurrences
        else:
            # wrong guess
            self.lives_left -= 1
            self.reward = -10

        masked = self.get_masked_word()

        # terminal checks
        if "_" not in masked:
            self.done = True
            self.reward += 100   # win bonus
            info["outcome"] = "win"

        elif self.lives_left <= 0:
            self.done = True
            self.reward -= 50    # lose penalty
            info["outcome"] = "lose"

        return masked, self.reward, self.done, info

    def render(self):
        """Optional: prints game state."""
        print(f"Word: {self.get_masked_word()}  |  Lives: {self.lives_left}  |  Guessed: {sorted(self.guessed)}")
