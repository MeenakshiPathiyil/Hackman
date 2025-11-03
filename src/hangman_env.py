import numpy as np

class HangmanEnv:
    def __init__(self, word: str, max_lives: int = 6):
        self.word = word.lower()
        self.max_lives = max_lives
        self.reset()

    def reset(self):
        self.guessed = set()
        self.lives_left = self.max_lives
        self.done = False
        return self.get_masked_word()

    def get_masked_word(self):
        return "".join(c if c in self.guessed else "_" for c in self.word)
    
    def step(self, guess):
        if guess in self.guessed:
            return self.get_masked_word(), -10, self.done, {}  # Stronger repeat penalty

        self.guessed.add(guess)
        count = self.word.count(guess)
        
        # Base step reward
        if count > 0:
            reward = 15 * count  # +15 per occurrence
        else:
            reward = -15        # -15 for wrong guess
            self.lives_left -= 1

        # Terminal conditions (override base reward)
        if "_" not in self.get_masked_word():
            self.done = True
            reward = 200          # Strong win bonus
        elif self.lives_left <= 0:
            self.done = True
            reward = -100         # Strong loss penalty
        
        reward -= 2  # Per-step penalty for efficiency
        return self.get_masked_word(), reward, self.done, {}

    def render(self):
        print(f"Word: {self.get_masked_word()} | Lives: {self.lives_left} | Guessed: {sorted(self.guessed)}")
