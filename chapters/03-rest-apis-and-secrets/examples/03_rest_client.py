"""A REST client with httpx2: create, read, update and delete (CRUD) resources.

The "server" is a fake notes API that runs inside this process through `httpx2.MockTransport`,
so the example works offline. Against a real server the client code is identical: only the
`transport=` argument goes away. MockTransport is also how you unit-test your own API clients.

Run: uv run chapters/03-rest-apis-and-secrets/examples/03_rest_client.py
"""

import itertools
import json
from typing import Any

import httpx2

API_KEY = "demo-key-not-a-secret"  # a real key comes from the environment: see 06_env_and_secrets

# ----------------------------------------------------------------------------------------------
# The fake server. A real one would be FastAPI, Django... the HTTP it speaks is the same.
# ----------------------------------------------------------------------------------------------
NOTES: dict[int, dict[str, Any]] = {}  # the server's "database": note id -> note
IDS = itertools.count(1)  # 1, 2, 3... a new id for each created note


def notes_api(request: httpx2.Request) -> httpx2.Response:
    """Route a request to the right action, like a web framework does."""
    # Authentication first: no valid key, no data (401 Unauthorized).
    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return httpx2.Response(401, json={"error": "invalid or missing API key"})

    # The URL path says WHICH resource, the method says WHAT to do with it.
    parts = request.url.path.strip("/").split("/")  # "/notes/3" -> ["notes", "3"]
    if parts[0] != "notes" or len(parts) > 2:
        return httpx2.Response(404, json={"error": "no such endpoint"})

    if len(parts) == 1:  # the collection: /notes
        if request.method == "GET":  # list, with an optional ?limit=N query parameter
            limit = int(request.url.params.get("limit", "10"))
            return httpx2.Response(200, json={"data": list(NOTES.values())[:limit]})
        if request.method == "POST":  # create a new note from the JSON body
            body = json.loads(request.content)
            if not isinstance(body.get("text"), str):
                return httpx2.Response(422, json={"error": "field 'text' (string) is required"})
            note = {"id": next(IDS), "text": body["text"], "done": False}
            NOTES[note["id"]] = note
            # 201 Created + a Location header that says where the new resource lives.
            return httpx2.Response(201, json=note, headers={"Location": f"/notes/{note['id']}"})
        return httpx2.Response(405, json={"error": "method not allowed"})

    note_id = int(parts[1])  # a single item: /notes/{id}
    if note_id not in NOTES:
        return httpx2.Response(404, json={"error": f"note {note_id} not found"})
    if request.method == "GET":
        return httpx2.Response(200, json=NOTES[note_id])
    if request.method == "PATCH":  # partial update: change only the fields that were sent
        NOTES[note_id] |= json.loads(request.content)
        return httpx2.Response(200, json=NOTES[note_id])
    if request.method == "DELETE":
        del NOTES[note_id]
        return httpx2.Response(204)  # 204 No Content: success, and no body to send back
    return httpx2.Response(405, json={"error": "method not allowed"})


# ----------------------------------------------------------------------------------------------
# The client: this is the code you write against any REST API.
# ----------------------------------------------------------------------------------------------
def log_response(response: httpx2.Response) -> None:
    """Event hook: httpx2 calls it for every response, handy for logging and debugging."""
    request = response.request
    print(f"  {request.method:6} {request.url} -> {response.status_code} {response.reason_phrase}")


def make_client() -> httpx2.Client:
    # Settings shared by every request go on the client, so each call stays short.
    return httpx2.Client(
        base_url="https://api.notes.example",  # requests use paths relative to this
        headers={"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"},
        timeout=10,  # seconds; always set one
        event_hooks={"response": [log_response]},
        transport=httpx2.MockTransport(notes_api),  # remove this line to talk to a real server
    )


if __name__ == "__main__":
    with make_client() as client:
        print("CREATE (POST): `json=` serializes the dict and sets Content-Type for us")
        r = client.post("/notes", json={"text": "read MDN's overview of HTTP"})
        print("   sent:", r.request.headers["Content-Type"], r.request.content.decode())
        print("   got: ", r.json(), "| Location:", r.headers["Location"])
        client.post("/notes", json={"text": "put the API key in .env"})

        print("\nREAD (GET): `params=` builds the query string")
        r = client.get("/notes", params={"limit": 5})
        for note in r.json()["data"]:
            print("  ", note)

        print("\nUPDATE (PATCH): send only the fields that change")
        r = client.patch("/notes/1", json={"done": True})
        print("  ", r.json())

        print("\nDELETE: 204 means success with an empty body")
        r = client.delete("/notes/2")
        print("   body:", r.content)

        print("\nERRORS: httpx2 does NOT raise on 4xx/5xx by itself...")
        r = client.get("/notes/2")
        print("   r.is_success:", r.is_success, "| body:", r.json())
        print("...so call raise_for_status() to turn an error status into an exception")
        try:
            r.raise_for_status()
        except httpx2.HTTPStatusError as e:
            print("   HTTPStatusError:", e.response.status_code, e.response.json()["error"])

        print("\nA wrong key -> 401, a wrong body -> 422: fix the request, retrying won't help")
        # Per-request headers override the client's defaults: here, a wrong key.
        r = client.get("/notes", headers={"Authorization": "Bearer wrong-key"})
        r = client.post("/notes", json={"title": "no text field"})  # 422: the body is wrong
        print("   422 body:", r.json())
