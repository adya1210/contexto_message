import base64
import json
import math
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from cryptography.fernet import Fernet, InvalidToken
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

if __package__:
    from .similarity_scorer import SimilarityScorer, parts_of_speech
else:
    from similarity_scorer import SimilarityScorer, parts_of_speech

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
GAME_STATE_FILE = BASE_DIR / "game_state.json"
GITHUB_API = "https://api.github.com"
MAX_MESSAGE_LENGTH = 500
MAX_GUESS_LENGTH = 40
MAX_SENDER_LENGTH = 60
MAX_HINT_GUESSES = 2000
FIRST_HINT_RANK = 300

STOP_WORDS = {
    "a", "about", "all", "and", "are", "as", "at", "be", "because", "been",
    "before", "but", "by", "can", "could", "did", "do", "does", "for", "from",
    "guess", "had", "has", "have", "he", "her", "here", "him", "his", "how",
    "i", "in", "is", "it", "its", "just", "like", "may", "me", "my", "of",
    "on", "or", "our", "please", "she", "so", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "to", "too", "try",
    "up", "us", "want", "was", "we", "were", "what", "when", "where", "which",
    "who", "with", "would", "you", "your"
}

app = Flask(__name__)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "https://adya1210.github.io,http://localhost:5500,"
        "http://127.0.0.1:5500,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024
scorer = SimilarityScorer()


class GitHubStateError(Exception):
    pass


def state_cipher():
    key = os.environ.get("STATE_ENCRYPTION_KEY")
    if not key:
        raise GitHubStateError("STATE_ENCRYPTION_KEY is not configured.")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as error:
        raise GitHubStateError("STATE_ENCRYPTION_KEY is invalid.") from error


def github_contents_request(method, payload=None):
    repository = os.environ.get("GITHUB_REPOSITORY", "adya1210/contexto_message")
    state_path = os.environ.get("GITHUB_STATE_PATH", "backend/game_state.json")
    branch = os.environ.get("GITHUB_STATE_BRANCH", "main")
    path = quote(state_path, safe="/")
    url = f"{GITHUB_API}/repos/{quote(repository, safe='/')}/contents/{path}"
    if method == "GET":
        url += f"?ref={quote(branch)}"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "message-contexto",
    }
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    api_request = Request(url, data=body, headers=headers, method=method)

    try:
        with urlopen(api_request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if method == "GET" and error.code == 404:
            return None
        raise GitHubStateError(
            f"GitHub returned HTTP {error.code} while accessing game state."
        ) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise GitHubStateError("Could not read or write game state on GitHub.") from error


def decrypt_state_field(data, encrypted_name, legacy_name):
    encrypted_value = data.get(encrypted_name)
    if encrypted_value is None:
        return data.get(legacy_name)
    try:
        return state_cipher().decrypt(encrypted_value.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeEncodeError, UnicodeDecodeError) as error:
        raise GitHubStateError("Could not decrypt the shared game state.") from error


def clean_text(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def clean_state(word, message, game_id, sender):
    """Return (secret, message, game_id, sender). Older secrets like "kids" map to "kid"."""
    word = clean_text(word)
    secret = scorer.normalize(word) if word else None
    return secret, clean_text(message), game_id, clean_text(sender)


def load_state():
    if os.environ.get("GITHUB_TOKEN"):
        file_data = github_contents_request("GET")
        if file_data is None:
            return clean_state(None, None, None, None)
        try:
            contents = base64.b64decode(file_data["content"]).decode("utf-8")
            data = json.loads(contents)
        except (KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GitHubStateError("The game-state file on GitHub is invalid.") from error
        if not isinstance(data, dict):
            raise GitHubStateError("The game-state file on GitHub is invalid.")
        word = decrypt_state_field(data, "encrypted_user_picked_word", "user_picked_word")
        message = decrypt_state_field(data, "encrypted_message", "message")
        sender = decrypt_state_field(data, "encrypted_sender", "sender")
        return clean_state(word, message, data.get("game_id"), sender)

    try:
        with open(GAME_STATE_FILE, "r") as file:
            data = json.load(file)
            if not isinstance(data, dict):
                raise GitHubStateError("The local game-state file is invalid.")
            return clean_state(
                data.get("user_picked_word"), data.get("message"),
                data.get("game_id"), data.get("sender"),
            )
    except (FileNotFoundError, json.JSONDecodeError):
        return clean_state(None, None, None, None)


def load_word():
    return load_state()[0]


def save_word(word, message, sender=None):
    game_id = os.urandom(16).hex()
    if os.environ.get("GITHUB_TOKEN"):
        existing_file = github_contents_request("GET")
        cipher = state_cipher()
        content = base64.b64encode(json.dumps({
            "encrypted_user_picked_word": cipher.encrypt(word.encode("utf-8")).decode("ascii"),
            "encrypted_message": cipher.encrypt(message.encode("utf-8")).decode("ascii"),
            "encrypted_sender": cipher.encrypt((sender or "").encode("utf-8")).decode("ascii"),
            "game_id": game_id,
        }).encode("utf-8")).decode("ascii")
        payload = {
            "message": "Update shared Message Contexto word",
            "content": content,
            "branch": os.environ.get("GITHUB_STATE_BRANCH", "main"),
        }
        if existing_file is not None:
            payload["sha"] = existing_file["sha"]
        github_contents_request("PUT", payload)
        return

    with open(GAME_STATE_FILE, "w") as file:
        json.dump({"user_picked_word": word, "message": message, "sender": sender, "game_id": game_id}, file)


# Filler words that are real dictionary words but make poor answers.
FILLER_WORDS = {
    "actually", "almost", "already", "also", "always", "anything", "anyway", "basically",
    "bit", "couple", "day", "doing", "else", "enough", "even", "ever", "every", "everyone",
    "everything", "feel", "get", "getting", "go", "going", "gonna", "good", "got", "great",
    "guy", "instead", "kind", "know", "least", "let", "literally", "lot", "lots", "make",
    "maybe", "much", "need", "never", "nothing", "now", "okay", "one", "pretty", "probably",
    "quite", "rather", "really", "right", "said", "say", "see", "something", "someone",
    "sometimes", "sort", "still", "stuff", "sure", "take", "tell", "thing", "think", "though",
    "time", "today", "together", "tomorrow", "tonight", "totally", "very", "way", "well",
    "whatever", "yeah", "yes", "yesterday", "yet",
    # interjections
    "aww", "haha", "hello", "hey", "lol", "miss", "nah", "omg", "please", "sorry", "thank",
    "thanks", "wow", "yay",
}
POS_SCORES = {"NOUN": 1.0, "ADJ": 0.55, "VERB": 0.5, "ADV": 0.1}


def choose_word(message):
    """Pick the most guessable word: a specific, real noun beats names and filler words."""
    tokens = list(re.finditer(r"[A-Za-z]+(?:['-][A-Za-z]+)*", message))
    scored = []
    for position, match in enumerate(tokens):
        raw_word = match.group().strip("'-")
        lowered = raw_word.lower()
        if len(lowered) < 3 or lowered in STOP_WORDS:
            continue
        word = scorer.normalize(lowered)
        if not scorer.knows(word) or scorer.is_too_common(word):
            continue

        tags = parts_of_speech(lowered) | parts_of_speech(word)
        # No dictionary entry means it's almost always a name (e.g. "riya", "singh").
        score = max((POS_SCORES.get(tag, 0) for tag in tags), default=-0.6)
        # Capitalized in the middle of a sentence is probably a name too.
        start = match.start()
        sentence_start = not message[:start].strip() or message[:start].rstrip()[-1] in ".!?"
        if raw_word[:1].isupper() and not sentence_start:
            score -= 0.4
        # Very common words make dull answers; specific words are more fun.
        frequency = scorer.frequency_rank(word)
        score += 0.6 * min(1.0, math.log(frequency + 2) / math.log(3000))
        score += 0.1 * (position + 1) / len(tokens)
        if lowered in FILLER_WORDS or word in FILLER_WORDS:
            score -= 1.5
        scored.append((score, word))

    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


@app.get("/")
def index():
    return send_from_directory(PROJECT_DIR, "index.html")


@app.get("/api/state")
def state():
    try:
        word = load_word()
        return jsonify({"has_word": word is not None})
    except GitHubStateError as error:
        app.logger.error("%s", error)
        return jsonify({"error": "Could not read the shared game state."}), 503


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/message")
def set_message():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}
    message = data.get("message", "")
    message = message.strip() if isinstance(message, str) else ""
    sender = data.get("from", "")
    sender = sender.strip() if isinstance(sender, str) else ""
    if len(sender) > MAX_SENDER_LENGTH:
        return jsonify({"error": "The from line must be 60 characters or fewer."}), 400
    if not message:
        return jsonify({"error": "Enter a message."}), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": "Messages must be 500 characters or fewer."}), 400

    word = choose_word(message)
    if word is None:
        return jsonify({"error": "The message needs at least one usable word."}), 400

    try:
        save_word(word, message, sender or None)
    except GitHubStateError as error:
        app.logger.error("%s", error)
        return jsonify({"error": "Could not save the shared word to GitHub."}), 503
    return jsonify({"word": word})


@app.post("/api/guess")
def guess():
    data = request.get_json(silent=True)
    guess_word = data.get("guess", "").strip().lower() if isinstance(data, dict) else ""
    if len(guess_word) > MAX_GUESS_LENGTH:
        return jsonify({"error": "Guesses must be 40 characters or fewer."}), 400

    try:
        secret, message, _, sender = load_state()
    except GitHubStateError as error:
        app.logger.error("%s", error)
        return jsonify({"error": "Could not read the shared game state."}), 503
    if secret is None:
        return jsonify({"error": "No word has been set."}), 400
    if not guess_word:
        return jsonify({"error": "Enter a guess."}), 400

    guess_word = scorer.normalize(guess_word)
    is_correct = guess_word == secret
    if is_correct:
        rank = 1
    else:
        if scorer.is_too_common(guess_word):
            return jsonify({"error": "That word is too common, try something more specific."}), 400
        if not scorer.knows(guess_word):
            return jsonify({"error": "I don't know that word, try another one."}), 400
        rank = scorer.rank(secret, guess_word)

    return jsonify(guess_response(guess_word, rank, is_correct, message, sender))


def guess_response(word, rank, is_correct, message, sender):
    response = {
        "word": word,
        "score": round(1.0 if is_correct else scorer.score_from_rank(rank), 4),
        "rank": rank,
        "total": scorer.total,
        "correct": is_correct,
    }
    if is_correct:
        if message:
            response["message"] = message
        if sender:
            response["from"] = sender
    return response


@app.post("/api/hint")
def hint():
    """Contexto-style hint: a word halfway between your best guess and the answer."""
    data = request.get_json(silent=True)
    guesses = data.get("guesses", []) if isinstance(data, dict) else []
    if not isinstance(guesses, list) or len(guesses) > MAX_HINT_GUESSES:
        return jsonify({"error": "Invalid guess list."}), 400

    try:
        secret, message, _, sender = load_state()
    except GitHubStateError as error:
        app.logger.error("%s", error)
        return jsonify({"error": "Could not read the shared game state."}), 503
    if secret is None or not scorer.knows(secret):
        return jsonify({"error": "No word has been set."}), 400

    ranks = [
        scorer.rank(secret, scorer.normalize(guess))
        for guess in guesses
        if isinstance(guess, str) and 0 < len(guess) <= MAX_GUESS_LENGTH
    ]
    ranks = [rank for rank in ranks if rank is not None]
    best = min(ranks, default=None)
    if best == 1:
        return jsonify({"error": "You already found the word!"}), 400

    target = min(FIRST_HINT_RANK, scorer.total) if best is None else max(1, best // 2)
    if best is not None and best > FIRST_HINT_RANK * 2:
        target = FIRST_HINT_RANK
    word = scorer.word_at_rank(secret, target)
    return jsonify(guess_response(word, target, target == 1, message, sender))


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5050")),
        debug=True,
    )
