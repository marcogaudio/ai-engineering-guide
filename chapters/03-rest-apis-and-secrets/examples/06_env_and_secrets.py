"""API keys and secrets: environment variables, .env files and pydantic-settings.

Everything here uses a FAKE key written to a temporary folder. The example never reads your
real `.env` file and never prints the real values of your environment variables.

Run: uv run chapters/03-rest-apis-and-secrets/examples/06_env_and_secrets.py
"""

import os
import tempfile
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Not a real key: it just looks like one, so you can see where it would leak.
FAKE_KEY = "sk-demo-4f9a2c71e0b84d6fa3c5e7d9b1a20c38"


def mask(secret: str) -> str:
    """Show just enough to tell keys apart: 'sk-d…0c38'. Never print a key in full."""
    return f"{secret[:4]}…{secret[-4:]}"


# A settings class: one place that declares every config value the app needs, with its type.
class Settings(BaseSettings):
    # Read the environment variables DEMO_API_KEY, DEMO_BASE_URL and DEMO_TIMEOUT_S.
    # (A prefix keeps your app's variables apart from everything else in the environment.)
    model_config = SettingsConfigDict(env_prefix="DEMO_")

    api_key: SecretStr  # required (no default): the app refuses to start without it
    base_url: str = "https://api.example.com/v1"  # optional: has a default
    timeout_s: float = 30.0  # arrives as the text "10" and is converted to the float 10.0


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * (60 - len(title)))


def main(tmp: Path) -> None:
    section("1. environment variables")
    # Every process gets a copy of its parent's environment: a dict of name -> text value.
    # In the shell, `export DEMO_API_KEY=sk-...` sets one for every program started afterwards.
    # Here we set it from Python, which only affects this process (and processes it starts).
    os.environ["DEMO_API_KEY"] = FAKE_KEY
    print("DEMO_API_KEY is set:", mask(os.environ["DEMO_API_KEY"]))

    # Required value -> os.environ[...]: crashes right away if it's missing (fail fast).
    try:
        os.environ["DEMO_MISSING_KEY"]
    except KeyError as e:
        print("os.environ[...] on a missing variable -> KeyError:", e)
    # Optional value -> os.getenv(...): returns None, or the default you pass.
    print("os.getenv(...) with a default ->", os.getenv("DEMO_MODEL", "llama3.2"))
    del os.environ["DEMO_API_KEY"]

    section("2. a .env file")
    # A .env file holds the same NAME=value lines, so you don't have to export them by hand
    # in every new terminal. It lives in the project folder, is listed in .gitignore, and is
    # never committed. `.env.example` (same names, fake values) IS committed as documentation.
    env_file = tmp / ".env"
    env_file.write_text(
        "# local secrets: listed in .gitignore, never committed\n"
        f"DEMO_API_KEY={FAKE_KEY}\n"
        "DEMO_TIMEOUT_S=10\n"
    )
    # dotenv_values() only reads the file into a dict: the environment is untouched.
    values = dotenv_values(env_file)
    print("dotenv_values:", {k: mask(v) if "KEY" in k and v else v for k, v in values.items()})

    # load_dotenv() copies the file into os.environ. Variables that are ALREADY set win
    # (override=False), so `DEMO_TIMEOUT_S=5 uv run app.py` still beats the file.
    os.environ["DEMO_TIMEOUT_S"] = "5"
    load_dotenv(env_file)
    print("after load_dotenv: key", mask(os.environ["DEMO_API_KEY"]),
          "| timeout", os.environ["DEMO_TIMEOUT_S"], "(the real environment won over the file)")
    # Same effect without any code: `uv run --env-file .env app.py`.

    section("3. pydantic-settings")
    # Reads the environment (and optionally a .env file), converts types, validates, and
    # wraps the key in a SecretStr. Priority: real env vars > .env file > defaults.
    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]  # api_key comes from env
    print("settings:", settings)  # the key is masked: safe to print or log
    print("timeout_s:", settings.timeout_s, type(settings.timeout_s).__name__)
    print("f-string of the key:", f"{settings.api_key}")  # still masked
    # Unwrap the real value only at the point of use, e.g. to build the auth header:
    headers = {"Authorization": f"Bearer {settings.api_key.get_secret_value()}"}

    # A missing required secret stops the app at startup, with a clear message, instead of a
    # confusing 401 from the API ten minutes into a long eval run.
    for name in ["DEMO_API_KEY", "DEMO_TIMEOUT_S"]:
        del os.environ[name]
    try:
        Settings()  # type: ignore[call-arg]
    except ValidationError as e:
        print("no key anywhere ->", e.errors()[0]["loc"][0], ":", e.errors()[0]["msg"])

    section("4. never log a secret")
    # Headers are the classic leak: logging a request "for debugging" writes the key to disk.
    def redact(headers: dict[str, str]) -> dict[str, str]:
        sensitive = {"authorization", "x-api-key", "api-key"}
        return {k: "***" if k.lower() in sensitive else v for k, v in headers.items()}

    print("log this:", redact(headers | {"Content-Type": "application/json"}))


if __name__ == "__main__":
    # A temporary folder that is deleted, with our fake .env inside, when the block ends.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            main(Path(tmp))
        finally:
            # Leave the environment as we found it (in case main() stopped half-way).
            for name in [n for n in os.environ if n.startswith("DEMO_")]:
                del os.environ[name]
