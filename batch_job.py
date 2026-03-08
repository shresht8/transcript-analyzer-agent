"""
Asynchronous batch processing script for Claude API with HuggingFace dataset upload.

This script processes datasets in batches asynchronously, allowing multiple batch jobs
to run concurrently, and pushes the final results to HuggingFace Hub.
"""

import argparse
import asyncio
import json
import os
import re
import time
from typing import List, Dict, Any

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from datasets import load_dataset, Dataset, concatenate_datasets
from jsonschema import validate, ValidationError
from huggingface_hub import login
from dotenv import load_dotenv
# System prompt for guiding the model
from prompts.system_prompts_str_gen import SYSTEM_PROMPT, SYSTEM_PROMPT_RETRY

def create_retry_dataset(
        original_dataset: Dataset,
):
    successful_dataset = original_dataset.filter(lambda x: x['generation_success'])
    retry_dataset = original_dataset.filter(lambda x: not x['generation_success'])
    # retry_dataset = []
    # for idx in range(len(original_dataset)):
    #     if not original_dataset[idx]['generation_success']:
    #         # print(f"Index {idx} failed with error: {original_dataset[idx]['generation_error']}")
    #         # Logic to create retry dataset
    #         retry_dataset.append(original_dataset[idx])
    #     else:
    #         successful_dataset.append(original_dataset[idx])

    # return Dataset.from_list(retry_dataset),Dataset.from_list(successful_dataset)
    return retry_dataset, successful_dataset

def create_batch_requests(dataset, batch_start: int, batch_size: int, system_prompt: str, is_retry: bool = False) -> tuple:
    """Create batch requests for the Claude API."""
    requests_list = []
    batch_end = min(batch_start + batch_size, len(dataset))
    
    for idx in range(batch_start, batch_end):
        row = dataset[idx]
        if is_retry:
            doc_and_schema = row['responses_create_params']['input'][0]['content']
            generated_output = row['generated_output']
            generation_error = row['generation_error']
            num_fields = row['schema_fields_count']
            user_prompt = f"""\n\n Document and Schema:\n{doc_and_schema}\n\n Generated Output:\n{generated_output}\n\n Generation Error:\n{generation_error}\n\n Number of Fields:\n{num_fields}"""
        else:
            doc_and_schema = row['responses_create_params']['input'][0]['content']
            user_prompt = f"""Here is the document and JSON schema you need to work with:\n\n{doc_and_schema}\n\nPlease generate the structured JSON output as per the schema."""
        request = Request(
            custom_id=f"id-{idx}",
            params=MessageCreateParamsNonStreaming(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": user_prompt,
                }]
            )
        )
        requests_list.append(request)
    
    return requests_list, batch_start, batch_end


def submit_batch(client: anthropic.Anthropic, dataset, batch_start: int, batch_size: int, system_prompt: str, is_retry: bool = False) -> tuple:
    """Submit a batch job to Claude API."""
    requests_list, batch_start, batch_end = create_batch_requests(dataset, batch_start, batch_size, system_prompt, is_retry)
    print(f"Submitting batch from {batch_start} to {batch_end}")
    response = client.messages.batches.create(requests=requests_list)
    print(f"Batch ID: {response.id}")
    return response.id, batch_start, batch_end


async def wait_for_batch_completion(client: anthropic.Anthropic, batch_id: str, poll_interval: int = 300):
    """Wait asynchronously for a batch to complete."""
    print(f"Waiting for batch {batch_id} to complete...")
    while True:
        # Run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        message_batch = await loop.run_in_executor(
            None, client.messages.batches.retrieve, batch_id
        )
        
        if message_batch.processing_status == "ended":
            print(f"Batch {batch_id} completed!")
            return batch_id
        
        print(f"Batch {batch_id} still processing... (status: {message_batch.processing_status})")
        await asyncio.sleep(poll_interval)


def process_batch_results(client: anthropic.Anthropic, dataset, batch_id: str, batch_start: int, batch_size: int) -> Dataset:
    """Process results from a completed batch."""
    print(f"Processing results for batch {batch_id}")
    batch_data = []
    batch_results = []
    batch_end = min(batch_start + batch_size, len(dataset))
    
    for idx in range(batch_start, batch_end):
        batch_data.append(dataset[idx])
    
    # Stream results
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            print(f"Success! {result.custom_id}")
            dataset_idx = int(result.custom_id.split('-')[1])
            schema_str = dataset[dataset_idx]['schema_str']
            schema = json.loads(schema_str)
            response = result.result
            response_text = response.message.content[0].text.strip()
            
            # Extract JSON if wrapped in markdown code blocks or embedded in text
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()
            else:
                # If no code blocks, try to find the outermost JSON object
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx+1]
            
            try:
                validate(instance=json.loads(response_text), schema=schema)
                batch_results.append({"success": True, "output": json.loads(response_text), "error": None})
            except ValidationError as e:
                print(f"Validation error for {result.custom_id}")
                batch_results.append({"success": False, "output": response_text, "error": str(e)})
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {result.custom_id}: {e}")
                print(f"Failed response text: {response_text[:100]}...")
                batch_results.append({"success": False, "output": response_text, "error": str(e)})
            except Exception as e:
                print(f"Unexpected error for {result.custom_id}: {e}")
                batch_results.append({"success": False, "output": response_text, "error": str(e)})
        
        elif result.result.type == "errored":
            error_msg = str(result.result.error)
            print(f"Error in {result.custom_id}: {error_msg}")
            batch_results.append({"success": False, "output": None, "error": error_msg})
        
        elif result.result.type == "expired":
            print(f"Request expired {result.custom_id}")
            batch_results.append({"success": False, "output": None, "error": "Request expired"})
    
    # Create dataset for this batch
    batch_dict = {}
    for key in batch_data[0].keys():
        batch_dict[key] = [row[key] for row in batch_data]
    # Convert outputs to JSON strings to avoid PyArrow struct type issues
    batch_dict['generated_output'] = [json.dumps(r.get('output')) if r.get('output') is not None else None for r in batch_results]
    batch_dict['generation_success'] = [r['success'] for r in batch_results]
    batch_dict['generation_error'] = [json.dumps(r.get('error')) if r.get('error') is not None else None for r in batch_results]
    
    batch_dataset = Dataset.from_dict(batch_dict)
    print(f"Processed {len(batch_dataset)} records from batch {batch_id}")
    return batch_dataset


async def process_single_batch_async(client: anthropic.Anthropic, dataset, batch_id: str, batch_start: int, batch_size: int) -> Dataset:
    """Process a single batch asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, process_batch_results, client, dataset, batch_id, batch_start, batch_size)


async def process_all_batches_async(client: anthropic.Anthropic, dataset, batch_size: int, system_prompt: str, is_retry: bool = False) -> Dataset:
    """
    Process all batches asynchronously.
    
    Submits all batches first, then waits for them to complete concurrently,
    and processes results as they finish.
    """
    total_size = len(dataset)
    batch_info = []
    
    # Submit all batches
    print(f"Submitting {(total_size + batch_size - 1) // batch_size} batches...")
    for batch_start in range(0, total_size, batch_size):
        batch_id, start, end = submit_batch(client, dataset, batch_start, batch_size, system_prompt, is_retry)
        batch_info.append((batch_id, start, batch_size))
    
    print(f"\nAll batches submitted. Waiting for completion...")
    
    # Wait for all batches to complete concurrently
    tasks = [wait_for_batch_completion(client, batch_id) for batch_id, _, _ in batch_info]
    completed_batch_ids = await asyncio.gather(*tasks)
    
    print(f"\nAll batches completed. Processing results asynchronously...")
    
    # Process results for all completed batches concurrently
    tasks = [
        process_single_batch_async(client, dataset, batch_id, batch_start, batch_size)
        for batch_id, batch_start, batch_size in batch_info
    ]
    all_batches = await asyncio.gather(*tasks)
    
    # Concatenate all batches
    final_dataset = concatenate_datasets(all_batches)
    print(f"\nFinal dataset size: {len(final_dataset)} records")
    return final_dataset


def push_to_huggingface(dataset: Dataset, repo_name: str):
    """Push dataset to HuggingFace Hub."""
    print(f"\nPushing dataset to HuggingFace Hub: {repo_name}")
    
    # Load environment variables
    load_dotenv()
    
    # Login to HuggingFace
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: HUGGING_FACE_HUB_TOKEN not found in environment")
        print("Please set the token or run: huggingface-cli login")
        return False
    
    login(token=token)
    
    # Push dataset
    dataset.push_to_hub(repo_name, private=False)
    print(f"✅ Successfully pushed to: https://huggingface.co/datasets/{repo_name}")
    return True


async def main():
    parser = argparse.ArgumentParser(
        description="Asynchronously process datasets with Claude API and push to HuggingFace"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (train, validation, or test)"
    )
    parser.add_argument(
        "--is-retry",
        action="store_true",
        help="Flag to indicate if this is a retry run"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of records per batch (default: 100)"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        required=True,
        help="HuggingFace repository name (e.g., username/dataset-name)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (for testing)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Polling interval in seconds (default: 60)"
    )
    
    args = parser.parse_args()
    # Load dataset
    print(f"Loading dataset split: {args.split}")
    if args.is_retry:
        system_prompt = SYSTEM_PROMPT_RETRY
        generated_dataset = load_dataset("shresht8/structured-ouput-dataset", split=args.split)
        dataset, successful_dataset = create_retry_dataset(generated_dataset)
        print(f"Created retry dataset with {len(dataset)} records and successful dataset with {len(successful_dataset)} records")
    else:
        system_prompt = SYSTEM_PROMPT
        dataset = load_dataset(
            "nvidia/Nemotron-RL-instruction_following-structured_outputs",
            split=args.split
        )
    
    # Limit dataset if specified
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited dataset to {len(dataset)} records")
    
    print(f"Dataset size: {len(dataset)} records")

    if not args.batch_size:
        args.batch_size = len(dataset)
    print(f"Batch size: {args.batch_size}")
    
    # Load environment variables and initialize Anthropic client
    load_dotenv()
    client = anthropic.Anthropic()
    
    # Process all batches asynchronously
    start_time = time.time()
    final_dataset = await process_all_batches_async(
        client, dataset, args.batch_size, system_prompt, args.is_retry
    )
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Total records: {len(final_dataset)}")
    print(f"Successful: {sum(final_dataset['generation_success'])}")
    print(f"Failed: {sum(1 for x in final_dataset['generation_success'] if not x)}")
    print(f"{'='*60}\n")

    if args.is_retry:
        # Combine with successful dataset
        final_dataset = concatenate_datasets([final_dataset, successful_dataset])
        print(f"Combined final dataset size after adding successful records: {len(final_dataset)}")
    
    # Push to HuggingFace
    push_to_huggingface(final_dataset, args.hf_repo)


if __name__ == "__main__":
    asyncio.run(main())
