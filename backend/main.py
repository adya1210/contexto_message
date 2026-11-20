from similarity_scorer import SimilarityScorer
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GAME_STATE_FILE = BASE_DIR / "game_state.json"


def load_word():
    try:
        with open(GAME_STATE_FILE, "r") as f:
            data = json.load(f)
            word = data.get("user_picked_word")
            return word.strip() if isinstance(word, str) and word.strip() else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_word(word):
    with open(GAME_STATE_FILE, "w") as f:
        json.dump({"user_picked_word": word}, f)


def play_game(scorer):
    print("Welcome! Pick an option: A) Set a word to send to someone else B) Normal play C) Exit")
    option = input("Enter your choice (A, B, or C): ").strip().upper()
    if option == 'A':
        word = input("Enter the word you want to send: ").strip()
        if not word:
            print("The word cannot be empty.")
        else:
            save_word(word)
            print(f"You have set the word: {word}. Now, other players guess it.")
    elif option == 'B':
        word = load_word()
        if word is None:
            print("No word has been set yet. Choose option A first.")
        else:
            input_word = input("Enter your guess: ").strip()
            while input_word and input_word.lower() != word.lower():
                print(scorer.similarity(word, input_word))
                input_word = input("Enter your guess: ").strip()
            if input_word:
                print("Correct!")
    elif option == 'C':
        print("Exiting the game. Goodbye!")
        return
    else:
        print("Please enter A, B, or C.")
    return play_game(scorer)


if __name__ == "__main__":
    word = load_word()
    if word is None:
        print("No word is set in game_state.json. Use option A to set one.")
    scorer = SimilarityScorer()
    play_game(scorer)