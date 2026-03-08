# uv run test_pipeline.py --input urls.xlsx --limit 10

"""
Test the full pipeline with a limited number of URLs.

Reads up to --limit URLs from the Excel file, fetches their transcripts,
then scores them with Bedrock. Outputs go to test_data/ and test_results/
so they don't interfere with the main pipeline.

Usage:
    uv run test_pipeline.py [--input urls.xlsx] [--limit 10]
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

from fetch_transcripts import fetch_single_transcript, read_urls_from_excel
from analyze_transcripts import analyze_transcript, get_bedrock_client, save_results, SCORE_FIELDS
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

SEPARATOR = "-" * 60


def run_fetch(entries: list[dict], output_dir: Path) -> tuple[int, int, list[str]]:
    """Fetch transcripts sequentially (no concurrency needed for small tests)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    api = YouTubeTranscriptApi()
    ok, skipped, errors = 0, 0, []

    for entry in entries:
        video_id = entry["video_id"]
        output_path = output_dir / f"{video_id}.txt"

        if output_path.exists():
            print(f"  SKIP {video_id} (already fetched)")
            skipped += 1
            continue

        try:
            transcript = api.fetch(video_id)
            full_text = " ".join(snippet.text for snippet in transcript)
            output_path.write_text(full_text, encoding="utf-8")
            word_count = len(full_text.split())
            print(f"  OK   {video_id}  ({word_count:,} words)")
            ok += 1
        except Exception as e:
            msg = f"  FAIL {video_id}: {type(e).__name__}: {e}"
            print(msg)
            errors.append(msg)

        time.sleep(random.uniform(0.5, 1.5))

    return ok, skipped, errors


def run_analysis(transcript_files: list[Path], output_dir: Path, model_id: str) -> list[dict]:
    """Score each transcript file and return results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    client = get_bedrock_client()
    results = []

    for i, filepath in enumerate(transcript_files, 1):
        video_id = filepath.stem
        transcript_text = filepath.read_text(encoding="utf-8")

        if len(transcript_text) > 100_000:
            transcript_text = transcript_text[:100_000] + "\n\n[Transcript truncated due to length]"

        try:
            print(f"  [{i}/{len(transcript_files)}] Scoring {video_id}...", end=" ", flush=True)
            result = analyze_transcript(client, model_id, transcript_text, max_tokens=2048)
            result["video_id"] = video_id
            results.append(result)

            scores = result.get("scores", {})
            score_line = "  ".join(
                f"{k[:4]}={scores.get(k, {}).get('score', '?')}" for k in SCORE_FIELDS
            )
            print(f"total={result.get('total_score', '?')}  [{score_line}]")
            print(f"         {result.get('title_or_summary', '')[:80]}")

        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

        time.sleep(0.5)

    return results


def print_summary(results: list[dict]):
    """Print a readable summary table of scores."""
    if not results:
        return

    print(f"\n{'VIDEO ID':<15} {'TITLE':<40} {'VI':>3} {'MP':>3} {'CV':>3} {'TF':>3} {'PQ':>3} {'TOT':>4}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x.get("total_score", 0), reverse=True):
        scores = r.get("scores", {})
        vi = scores.get("value_innovation", {}).get("score", "-")
        mp = scores.get("market_potential", {}).get("score", "-")
        cv = scores.get("commercial_viability", {}).get("score", "-")
        tf = scores.get("technical_feasibility", {}).get("score", "-")
        pq = scores.get("presentation_quality", {}).get("score", "-")
        tot = r.get("total_score", "-")
        title = (r.get("title_or_summary") or "")[:39]
        print(f"{r['video_id']:<15} {title:<40} {vi:>3} {mp:>3} {cv:>3} {tf:>3} {pq:>3} {tot:>4}")


def main():
    parser = argparse.ArgumentParser(description="Test the full pipeline with a limited number of URLs")
    parser.add_argument("--input", default="urls.xlsx", help="Excel file with URLs")
    parser.add_argument("--limit", type=int, default=10, help="Max number of URLs to process")
    args = parser.parse_args()

    model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    data_dir = Path("test_data")
    results_dir = Path("test_results")

    print(SEPARATOR)
    print(f"TEST PIPELINE  limit={args.limit}  model={model_id}")
    print(SEPARATOR)

    # --- Step 1: Read URLs ---
    print(f"\n[1/3] Reading URLs from {args.input}...")
    all_entries = read_urls_from_excel(args.input)
    entries = all_entries[: args.limit]
    print(f"      Using {len(entries)} of {len(all_entries)} URLs.")
    for e in entries:
        print(f"      {e['video_id']}  {e['url']}")

    # --- Step 2: Fetch Transcripts ---
    print(f"\n[2/3] Fetching transcripts -> {data_dir}/")
    ok, skipped, fetch_errors = run_fetch(entries, data_dir)
    print(f"      Done: {ok} fetched, {skipped} skipped, {len(fetch_errors)} failed.")

    # --- Step 3: Analyze ---
    transcript_files = sorted(data_dir.glob("*.txt"))
    if not transcript_files:
        print("\nNo transcript files to analyze. Exiting.")
        return

    print(f"\n[3/3] Analyzing {len(transcript_files)} transcripts -> {results_dir}/")
    results = run_analysis(transcript_files, results_dir, model_id)

    if results:
        save_results(results, results_dir)

    # --- Summary ---
    print(f"\n{SEPARATOR}")
    print("RESULTS SUMMARY  (sorted by total score, highest first)")
    print(SEPARATOR)
    print_summary(results)
    print(f"\nVI=Value Innovation  MP=Market Potential  CV=Commercial Viability")
    print(f"TF=Technical Feasibility  PQ=Presentation Quality  TOT=Total (/25)")
    print(f"\nFull output: {results_dir}/scores.csv  |  {results_dir}/scores.json")


if __name__ == "__main__":
    main()
