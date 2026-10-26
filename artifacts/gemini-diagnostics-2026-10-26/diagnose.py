"""Bounded, one-shot probes of the configured model. Never retries or switches.

Run from the project root with PYTHONPATH=src. Reads the existing .env via
Settings. Archives only allowlisted/redacted response fields, never headers
containing credentials or raw error bodies. Refuses to overwrite a report.
"""

import json
import re
import time
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from cartoon_rag.settings import Settings


ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "diagnostics.json"
    if output.exists():
        raise ValueError("Diagnostic report already exists; do not overwrite historical probes")
    settings = Settings.load()
    secret = settings.gemini_api_key
    if not secret or any(not 33 <= ord(char) <= 126 for char in secret):
        raise ValueError("A validly formatted Gemini key must be configured")

    def clean(value):
        text = str(value).replace(secret, "[REDACTED]").replace(quote(secret, safe=""), "[REDACTED]")
        text = re.sub(r"projects/[A-Za-z0-9_-]+", "projects/[REDACTED]", text)
        return text[:2000]

    def redacted(value):
        if isinstance(value, str):
            return clean(value)
        if isinstance(value, dict):
            return {clean(k): redacted(v) for k, v in value.items()}
        if isinstance(value, list):
            return [redacted(v) for v in value]
        return value

    # Exercise redaction before any provider call, without persisting its input.
    assert secret not in clean("echo " + secret)
    assert quote(secret, safe="") not in clean("echo " + quote(secret, safe=""))

    report = {"started_at": datetime.now(timezone.utc).isoformat(),
              "model": settings.gemini_model, "api_version": "v1beta",
              "timeout_seconds": settings.generation_timeout_seconds,
              "automatic_retries": 0, "model_switches": 0,
              "billing_changes": False, "redaction_self_check_passed": True,
              "probes": []}

    def save():
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        assert secret not in text
        output.write_text(text)

    def probe(name, method, path, body=None):
        started = time.monotonic()
        item = {"name": name, "method": method, "path": path,
                "started_at": datetime.now(timezone.utc).isoformat()}
        headers = {"x-goog-api-key": secret, "Content-Type": "application/json"}
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        request = Request("https://generativelanguage.googleapis.com" + path,
                          headers=headers, data=data, method=method)
        try:
            with urlopen(request, timeout=settings.generation_timeout_seconds) as response:
                item["http_status"] = response.status
                result = json.load(response)
            if method == "GET":
                fields = ("name", "version", "displayName", "supportedGenerationMethods",
                          "inputTokenLimit", "outputTokenLimit")
                item["model_metadata"] = {key: result[key] for key in fields if key in result}
            else:
                candidates = result.get("candidates", [])
                item["candidate_count"] = len(candidates)
                item["finish_reasons"] = [clean(c.get("finishReason", "")) for c in candidates]
                item["has_output_text"] = any(
                    p.get("text") for c in candidates for p in c.get("content", {}).get("parts", [])
                    if not p.get("thought", False))
                if name == "structured_minimal":
                    try:
                        text = "".join(p.get("text", "") for p in candidates[0]["content"]["parts"]
                                       if not p.get("thought", False))
                        item["schema_satisfied"] = json.loads(text) == {"ok": True}
                    except (KeyError, IndexError, TypeError, ValueError):
                        item["schema_satisfied"] = False
        except HTTPError as exc:
            item["http_status"] = exc.code
            item["retry_after"] = clean(exc.headers.get("Retry-After", ""))
            try:
                error = json.loads(exc.read(65536)).get("error", {})
                item["provider_error"] = {
                    "code": error.get("code"), "status": clean(error.get("status", "")),
                    "message": clean(error.get("message", "")),
                    "reasons": [clean(d["reason"]) for d in error.get("details", [])
                                if isinstance(d, dict) and isinstance(d.get("reason"), str)],
                }
            except (ValueError, AttributeError, OSError, HTTPException):
                item["provider_error"] = {"body_parseable": False}
            finally:
                exc.close()
        except (URLError, OSError, HTTPException, ValueError, TypeError, AttributeError) as exc:
            item["transport_or_decode_error"] = type(exc).__name__
        item["elapsed_seconds"] = round(time.monotonic() - started, 3)
        # Apply a final redaction pass to all selected strings, including model metadata.
        item = redacted(item)
        report["probes"].append(item)
        save()
        print(json.dumps({"name": name, "http_status": item.get("http_status"),
                          "provider_error": item.get("provider_error")}, ensure_ascii=False), flush=True)
        return item

    model_path = "/v1beta/models/" + settings.gemini_model
    metadata = probe("model_metadata", "GET", model_path)
    if metadata.get("http_status") in (400, 401, 402, 403, 404, 429):
        report["generation_skipped"] = "Model metadata returned an access/configuration/quota error"
    else:
        config = {"temperature": 0, "candidateCount": 1,
                  "maxOutputTokens": settings.generation_max_output_tokens}
        plain = {"contents": [{"role": "user", "parts": [{"text": "Верни только слово OK."}]}],
                 "generationConfig": config}
        report["plain_request"] = plain
        status = probe("plain_minimal", "POST", model_path + ":generateContent", plain)
        if status.get("http_status") == 200 and status.get("has_output_text"):
            structured = {"contents": [{"role": "user", "parts": [{"text": 'Верни JSON: {"ok":true}.'}]}],
                          "generationConfig": {**config, "responseFormat": {"text": {
                              "mimeType": "APPLICATION_JSON", "schema": {
                                  "type": "object", "additionalProperties": False,
                                  "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
                              }}}}}
            report["structured_request"] = structured
            probe("structured_minimal", "POST", model_path + ":generateContent", structured)
        else:
            report["structured_probe_skipped"] = "Minimal unstructured generation failed; do not spend another call"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["generation_requests"] = sum(p["method"] == "POST" for p in report["probes"])
    save()


if __name__ == "__main__":
    main()
