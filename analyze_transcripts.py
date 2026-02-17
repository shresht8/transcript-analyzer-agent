"""
Step 2: Analyze transcripts using AWS Bedrock (Claude) and score them against the rubric.

Usage:
    uv run analyze_transcripts.py [--data-dir data] [--output-dir results] [--max-tokens 2048]
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

from rubric import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv()

SCORE_FIELDS = [
    "value_innovation",
    "market_potential",
    "commercial_viability",
    "technical_feasibility",
    "presentation_quality",
]


def get_bedrock_client():
    """Create a Bedrock Runtime client from environment variables."""
    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def analyze_transcript(client, model_id: str, transcript_text: str, max_tokens: int) -> dict:
    """Send a transcript to Bedrock Claude for scoring. Returns parsed JSON result."""
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT_TEMPLATE.format(transcript=transcript_text)}
                ],
            }
        ],
    }

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body),
    )

    response_body = json.loads(response["body"].read())
    raw_text = response_body["content"][0]["text"]

    # Parse JSON from the response (handle potential markdown code blocks)
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    return json.loads(cleaned)


def load_existing_results(output_path: Path) -> set[str]:
    """Load already-scored video IDs from existing JSON results."""
    if not output_path.exists():
        return set()
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {r["video_id"] for r in results}


def save_results(results: list[dict], output_dir: Path):
    """Save results as both JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save CSV
    csv_path = output_dir / "scores.csv"
    fieldnames = ["video_id", "title_or_summary"]
    for field in SCORE_FIELDS:
        fieldnames.append(f"{field}_score")
        fieldnames.append(f"{field}_justification")
    fieldnames.extend(["total_score", "overall_summary"])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                "video_id": r["video_id"],
                "title_or_summary": r.get("title_or_summary", ""),
                "total_score": r.get("total_score", ""),
                "overall_summary": r.get("overall_summary", ""),
            }
            for field in SCORE_FIELDS:
                score_data = r.get("scores", {}).get(field, {})
                row[f"{field}_score"] = score_data.get("score", "")
                row[f"{field}_justification"] = score_data.get("justification", "")
            writer.writerow(row)

    print(f"Results saved to {json_path} and {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze transcripts with AWS Bedrock")
    parser.add_argument("--data-dir", default="data", help="Directory with transcript .txt files")
    parser.add_argument("--output-dir", default="results", help="Directory for output files")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for model response")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    json_path = output_dir / "scores.json"

    model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    # Get transcript files
    transcript_files = sorted(data_dir.glob("*.txt"))
    if not transcript_files:
        print(f"No transcript files found in {data_dir}/")
        return

    print(f"Found {len(transcript_files)} transcript files.")

    # Load existing results for resume support
    done_ids = load_existing_results(json_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} already scored.")

    # Load existing results list
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Create Bedrock client
    client = get_bedrock_client()

    to_process = [f for f in transcript_files if f.stem not in done_ids]
    print(f"Analyzing {len(to_process)} transcripts with {model_id}...\n")

    for i, filepath in enumerate(to_process, 1):
        video_id = filepath.stem
        transcript_text = filepath.read_text(encoding="utf-8")

        # Truncate very long transcripts to avoid token limits (~100k chars is ~25k tokens)
        if len(transcript_text) > 100_000:
            transcript_text = transcript_text[:100_000] + "\n\n[Transcript truncated due to length]"

        try:
            print(f"[{i}/{len(to_process)}] Scoring {video_id}...", end=" ", flush=True)
            result = analyze_transcript(client, model_id, transcript_text, args.max_tokens)
            result["video_id"] = video_id
            all_results.append(result)
            print(f"total={result.get('total_score', '?')}")

            # Save after each successful analysis (crash-safe)
            save_results(all_results, output_dir)

        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

        # Small delay between API calls
        time.sleep(0.5)

    print(f"\nComplete. {len(all_results)}/{len(transcript_files)} transcripts scored.")


if __name__ == "__main__":
    main()
