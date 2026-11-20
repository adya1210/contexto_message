import base64
import hashlib
import hmac
import json
import os
import random
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from cryptography.fernet import Fernet, InvalidToken
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

if __package__:
    from .similarity_scorer import SimilarityScorer
else:
    from similarity_scorer import SimilarityScorer

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
GAME_STATE_FILE = BASE_DIR / "game_state.json"
GITHUB_API = "https://api.github.com"
MAX_MESSAGE_LENGTH = 500
MAX_GUESS_LENGTH = 40
MAX_HINT_TOKEN_LENGTH = 2048
LOCAL_HINT_SIGNING_KEY = os.urandom(32)

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


def hint_signing_key():
    key = os.environ.get("STATE_ENCRYPTION_KEY")
    if key:
        try:
            encryption_key = base64.urlsafe_b64decode(key.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as error:
            raise GitHubStateError("STATE_ENCRYPTION_KEY is invalid.") from error
        return hmac.new(encryption_key, b"message-contexto-hints", hashlib.sha256).digest()
    return LOCAL_HINT_SIGNING_KEY


def create_hint_token(game_id, revealed_indices, started=True):
    payload = json.dumps(
        {"game_id": game_id, "revealed_indices": revealed_indices, "started": started},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    signature = hmac.new(hint_signing_key(), payload, hashlib.sha256).digest()
    encoded_payload = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    encoded_signature = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{encoded_payload}.{encoded_signature}"


def read_hint_token(token, game_id, word):
    if not isinstance(token, str) or not token or len(token) > MAX_HINT_TOKEN_LENGTH:
        raise ValueError("Invalid hint token.")
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        payload = base64.urlsafe_b64decode(encoded_payload + "=" * (-len(encoded_payload) % 4))
        signature = base64.urlsafe_b64decode(encoded_signature + "=" * (-len(encoded_signature) % 4))
        expected_signature = hmac.new(hint_signing_key(), payload, hashlib.sha256).digest()
        token_data = json.loads(payload)
    except (ValueError, UnicodeEncodeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid hint token.") from error

    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Invalid hint token.")
    if not isinstance(token_data, dict) or token_data.get("game_id") != game_id:
        return False, []

    revealed = token_data.get("revealed_indices")
    started = token_data.get("started")
    if not isinstance(started, bool):
        raise ValueError("Invalid hint token.")
    if not isinstance(revealed, list) or any(
        isinstance(index, bool) or not isinstance(index, int)
        or index < 0 or index >= len(word) or not word[index].isalpha()
        for index in revealed
    ):
        raise ValueError("Invalid hint token.")
    if len(revealed) != len(set(revealed)):
        raise ValueError("Invalid hint token.")
    return started, revealed


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


def load_state():
    if os.environ.get("GITHUB_TOKEN"):
        file_data = github_contents_request("GET")
        if file_data is None:
            return None, None
        try:
            contents = base64.b64decode(file_data["content"]).decode("utf-8")
            data = json.loads(contents)
        except (KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GitHubStateError("The game-state file on GitHub is invalid.") from error
        if not isinstance(data, dict):
            raise GitHubStateError("The game-state file on GitHub is invalid.")
        word = decrypt_state_field(data, "encrypted_user_picked_word", "user_picked_word")
        message = decrypt_state_field(data, "encrypted_message", "message")
        game_id = data.get("game_id")
        clean_word = word.strip().lower() if isinstance(word, str) and word.strip() else None
        clean_message = message.strip() if isinstance(message, str) and message.strip() else None
        return clean_word, clean_message, game_id

    try:
        with open(GAME_STATE_FILE, "r") as file:
            data = json.load(file)
            if not isinstance(data, dict):
                raise GitHubStateError("The local game-state file is invalid.")
            word = data.get("user_picked_word")
            message = data.get("message")
            game_id = data.get("game_id")
            clean_word = word.strip().lower() if isinstance(word, str) and word.strip() else None
            clean_message = message.strip() if isinstance(message, str) and message.strip() else None
            return clean_word, clean_message, game_id
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None, None


def load_word():
    return load_state()[0]


def save_word(word, message):
    game_id = os.urandom(16).hex()
    if os.environ.get("GITHUB_TOKEN"):
        existing_file = github_contents_request("GET")
        cipher = state_cipher()
        content = base64.b64encode(json.dumps({
            "encrypted_user_picked_word": cipher.encrypt(word.encode("utf-8")).decode("ascii"),
            "encrypted_message": cipher.encrypt(message.encode("utf-8")).decode("ascii"),
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
        json.dump({"user_picked_word": word, "message": message, "game_id": game_id}, file)


def choose_word(message):
    candidates = re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", message)
    scored = []
    for index, raw_word in enumerate(candidates):
        word = raw_word.strip("'-").lower()
        if len(word) < 3 or word in STOP_WORDS or scorer.is_too_common(word):
            continue

        in_vocabulary = scorer.knows(word)
        length_score = min(len(word), 12) / 12
        position_score = (index + 1) / max(len(candidates), 1)
        capitalized = raw_word[:1].isupper()

        score = length_score * 0.35 + position_score * 0.25
        # Strongly prefer words the model knows, otherwise guesses can't be ranked.
        score += 1.0 if in_vocabulary else 0.0
        score += 0.10 if capitalized else 0
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
    if os.environ.get("GITHUB_TOKEN") and not os.environ.get("GAME_ADMIN_KEY"):
        return jsonify({"error": "Game admin access is not configured on the server."}), 503
    admin_key = os.environ.get("GAME_ADMIN_KEY")
    if admin_key and not hmac.compare_digest(
        request.headers.get("X-Game-Admin-Key", ""), admin_key
    ):
        return jsonify({"error": "Only the game owner can set a new message."}), 403

    data = request.get_json(silent=True)
    message = data.get("message", "").strip() if isinstance(data, dict) else ""
    if not message:
        return jsonify({"error": "Enter a message."}), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": "Messages must be 500 characters or fewer."}), 400

    word = choose_word(message)
    if word is None:
        return jsonify({"error": "The message needs at least one usable word."}), 400

    try:
        save_word(word, message)
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
        secret, message, _ = load_state()
    except GitHubStateError as error:
        app.logger.error("%s", error)
        return jsonify({"error": "Could not read the shared game state."}), 503
    if secret is None:
        return jsonify({"error": "No word has been set."}), 400
    if not guess_word:
        return jsonify({"error": "Enter a guess."}), 400

    is_correct = guess_word == secret
    if is_correct:
        rank = 1
    else:
        if scorer.is_too_common(guess_word):
            return jsonify({"error": "That word is too common, try something more specific."}), 400
        if not scorer.knows(guess_word):
            return jsonify({"error": "I don't know that word, try another one."}), 400
        rank = scorer.rank(secret, guess_word)

    response = {
        "score": round(1.0 if is_correct else scorer.score_from_rank(rank), 4),
        "rank": rank,
        "total": scorer.total,
        "correct": is_correct,
    }
    if is_correct and message:
        response["message"] = message
    return jsonify(response)


@app.post("/api/hint")
def hint():
    data = request.get_json(silent=True)
    token = data.get("token") if isinstance(data, dict) else None

    try:
        secret, message, game_id = load_state()
    except GitHubStateError as error:
        app.logger.error("%s", error)
        return jsonify({"error": "Could not read the shared game state."}), 503
    if secret is None:
        return jsonify({"error": "No word has been set."}), 400

    if not isinstance(game_id, str) or not game_id:
        game_id = "legacy-game"

    try:
        started, revealed = read_hint_token(token, game_id, secret) if token else (False, [])
    except (GitHubStateError, ValueError):
        return jsonify({"error": "Your hint state is invalid. Refresh and try again."}), 400

    letter_indices = [index for index, character in enumerate(secret) if character.isalpha()]
    if not letter_indices:
        return jsonify({"error": "The answer has no letters to reveal."}), 400

    if not started:
        return jsonify({
            "letter_count": len(letter_indices),
            "revealed_count": 0,
            "pattern": "".join(
                character if not character.isalpha() else "_"
                for character in secret
            ),
            "token": create_hint_token(game_id, [], started=True),
            "complete": False,
        })

    remaining = [index for index in letter_indices if index not in revealed]
    if remaining:
        revealed.append(random.choice(remaining))

    pattern = "".join(
        character if not character.isalpha() or index in revealed else "_"
        for index, character in enumerate(secret)
    )
    complete = len(revealed) == len(letter_indices)
    response = {
        "letter_count": len(letter_indices),
        "revealed_count": len(revealed),
        "pattern": pattern,
        "token": create_hint_token(game_id, revealed),
        "complete": complete,
    }
    if complete and message:
        response["message"] = message
    return jsonify(response)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5050")),
        debug=True,
    )
