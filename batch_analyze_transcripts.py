"""
Analyze transcripts using Anthropic API batch jobs for cost-efficient processing.
Batches run asynchronously at 50% of standard prices.

Usage:
    uv run batch_analyze_transcripts.py [--data-dir data] [--output-dir results] [--max-tokens 2048] [--poll-interval 60]
"""

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
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


def load_existing_results(output_path: Path) -> tuple[list[dict], set[str]]:
    """Load already-scored results and return (results_list, done_video_ids)."""
    if not output_path.exists():
        return [], set()
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results, {r["video_id"] for r in results}


def save_results(results: list[dict], output_dir: Path):
    """Save results as both JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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


def parse_response_text(raw_text: str) -> dict:
    """Parse JSON from model response, handling optional markdown code fences."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())


def build_batch_requests(
    transcript_files: list[Path],
    model_id: str,
    max_tokens: int,
) -> list[Request]:
    """Create one Anthropic batch Request per transcript file."""
    requests = []
    for filepath in transcript_files:
        video_id = filepath.stem
        transcript_text = filepath.read_text(encoding="utf-8")

        # Truncate very long transcripts to avoid token limits (~100k chars ≈ 25k tokens)
        if len(transcript_text) > 100_000:
            transcript_text = transcript_text[:100_000] + "\n\n[Transcript truncated due to length]"

        requests.append(
            Request(
                custom_id=video_id,
                params=MessageCreateParamsNonStreaming(
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": USER_PROMPT_TEMPLATE.format(transcript=transcript_text),
                        }
                    ],
                ),
            )
        )
    return requests


async def wait_for_batch(client: anthropic.Anthropic, batch_id: str, poll_interval: int):
    """Poll the batch status until it ends."""
    print(f"Waiting for batch {batch_id} to complete...")
    loop = asyncio.get_event_loop()
    while True:
        batch = await loop.run_in_executor(None, client.messages.batches.retrieve, batch_id)
        if batch.processing_status == "ended":
            counts = batch.request_counts
            print(
                f"Batch complete! succeeded={counts.succeeded} "
                f"errored={counts.errored} expired={counts.expired}"
            )
            return
        counts = batch.request_counts
        print(
            f"  [{batch.processing_status}] processing={counts.processing} "
            f"succeeded={counts.succeeded} errored={counts.errored}"
        )
        await asyncio.sleep(poll_interval)


def collect_results(
    client: anthropic.Anthropic,
    batch_id: str,
    existing_results: list[dict],
) -> list[dict]:
    """Stream batch results and append parsed scores to existing_results."""
    new_results = []
    for result in client.messages.batches.results(batch_id):
        video_id = result.custom_id
        if result.result.type == "succeeded":
            raw_text = result.result.message.content[0].text
            try:
                parsed = parse_response_text(raw_text)
                parsed["video_id"] = video_id
                new_results.append(parsed)
                print(f"  ✓ {video_id} | total={parsed.get('total_score', '?')}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  ✗ {video_id} | parse error: {e}")
        elif result.result.type == "errored":
            print(f"  ✗ {video_id} | API error: {result.result.error}")
        elif result.result.type == "expired":
            print(f"  ✗ {video_id} | request expired")

    return existing_results + new_results


async def main():
    parser = argparse.ArgumentParser(description="Batch-analyze transcripts with Anthropic API")
    parser.add_argument("--data-dir", default="data", help="Directory with transcript .txt files")
    parser.add_argument("--output-dir", default="results", help="Directory for output files")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for model response")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between batch status polls (default: 60)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    json_path = output_dir / "scores.json"

    model_id = os.getenv("ANTHROPIC_MODEL_ID", "claude-opus-4-6")

    # Discover transcript files
    transcript_files = sorted(data_dir.glob("*.txt"))
    if not transcript_files:
        print(f"No transcript files found in {data_dir}/")
        return
    print(f"Found {len(transcript_files)} transcript files.")

    # Resume support: skip already-scored transcripts
    existing_results, done_ids = load_existing_results(json_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} already scored, skipping.")

    to_process = [f for f in transcript_files if f.stem not in done_ids]
    if not to_process:
        print("All transcripts already scored.")
        return

    print(f"Submitting {len(to_process)} transcripts as a batch using {model_id}...\n")

    client = anthropic.Anthropic()

    # Build and submit batch
    requests = build_batch_requests(to_process, model_id, args.max_tokens)
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted: {batch.id}\n")

    # Wait asynchronously for the batch to finish
    await wait_for_batch(client, batch.id, args.poll_interval)

    # Collect and parse results
    print("\nProcessing results...")
    all_results = collect_results(client, batch.id, existing_results)

    # Save JSON + CSV
    save_results(all_results, output_dir)

    print(f"\nComplete. {len(all_results)}/{len(transcript_files)} transcripts scored.")


if __name__ == "__main__":
    asyncio.run(main())
