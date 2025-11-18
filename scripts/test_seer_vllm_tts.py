import json
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from tqdm import tqdm
from typing import List, Dict, Any

os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# vLLM service configuration
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://0.0.0.0:5001/v1"

# Input and output file paths
INPUT_FILE = "xxx"
OUTPUT_FILE = "xxx"

# Number of parallel workers
MAX_WORKERS = 200

NUM_RESPONSES = 5

# Model inference parameters
MODEL_PARAMS = {
    "model": "xxx",
    "max_tokens": 4096,
    "temperature": 0.8,
    "top_p": 0.95,
}

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Read JSONL file
def load_data(json_path: str = INPUT_FILE) -> List[Dict]:
    """Load data from JSONL file, extracting prompt content and raw data"""
    try:
        data = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        record = json.loads(line.strip())
                        # Extract user message content from prompt list
                        prompt_content = ""
                        for msg in record.get("prompt", []):
                            if msg.get("role") == "user":
                                prompt_content = msg.get("content", "")
                                break
                        data.append({
                            "prompt": prompt_content,  # Store as string for hashing
                            "raw_data": record  # Keep full record for output
                        })
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSONL line: {line[:50]}...")
                        continue
        logger.info(f"Loaded {len(data)} records from {json_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File {json_path} not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

# Generate response for a single prompt
def generate_single_response(prompt: str, client: OpenAI, retries: int = 3) -> Dict[str, Any]:
    """Generate response for a single prompt using vLLM"""
    for attempt in range(retries):
        try:
            # Apply chat template to mimic tokenizer.apply_chat_template
            chat_template = f"{prompt}<|im_end|>\n<|im_start|>assistant"
            response = client.chat.completions.create(
                model=MODEL_PARAMS["model"],
                messages=[
                    {"role": "user", "content": chat_template},
                ],
                max_tokens=MODEL_PARAMS["max_tokens"],
                temperature=MODEL_PARAMS["temperature"],
                top_p=MODEL_PARAMS["top_p"],
            )
            response_text = response.choices[0].message.content
            return {"result": response_text}
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for prompt: {prompt[:50]}... Error: {str(e)}")
            if attempt == retries - 1:
                logger.error(f"Max retries reached for prompt: {prompt[:50]}...")
                return {"result": f"Error: {str(e)}"}
            time.sleep(2 ** attempt)  # Exponential backoff

# Process a single data item
def process_single_item(item: Dict, client: OpenAI) -> Dict[str, Any]:
    """Process a single data item and generate multiple results"""
    try:
        results = []
        for _ in range(NUM_RESPONSES):
            result = generate_single_response(item["prompt"], client)
            results.append(result["result"])
        res_sample = item["raw_data"]
        res_sample["result_ours"] = results  # Store all responses as a list
        return res_sample
    except Exception as e:
        logger.error(f"Error processing prompt: {item['prompt'][:50]}... {str(e)}")
        res_sample = item["raw_data"]
        res_sample["result_ours"] = [f"Error: {str(e)}"] * NUM_RESPONSES
        return res_sample

# Main function
def generate_responses(json_path: str = INPUT_FILE, output_path: str = OUTPUT_FILE, max_workers: int = MAX_WORKERS):
    """Process dataset in parallel and generate multiple responses"""
    try:
        # Load data
        data = load_data(json_path)

        # Check for already processed prompts
        processed_prompts = set()
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        # Extract user message content from prompt list
                        prompt_content = ""
                        for msg in record.get("prompt", []):
                            if msg.get("role") == "user":
                                prompt_content = msg.get("content", "")
                                break
                        processed_prompts.add(prompt_content)
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Found {len(processed_prompts)} existing responses in {output_path}")

        # Filter unprocessed data
        remaining_data = [item for item in data if item["prompt"] not in processed_prompts]
        logger.info(f"Processing {len(remaining_data)} remaining prompts")

        if not remaining_data:
            logger.info("All prompts already processed.")
            return

        # Create a shared OpenAI client
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE
        )

        # Process prompts in parallel
        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_item = {executor.submit(process_single_item, item, client): item for item in remaining_data}
            
            # Show progress with tqdm
            with tqdm(total=len(remaining_data), desc="Writing results", unit="item") as pbar:
                # Process results as they complete
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        # Append to JSONL file
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        logger.info(f"Saved response for prompt: {result['prompt'][0]['content'][:50]}...")
                        # Check if any response is successful (not an error)
                        if not all(r.startswith("Error:") for r in result["result_ours"]):
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Error processing prompt: {item['prompt'][:50]}... {str(e)}")
                        res_sample = item["raw_data"]
                        res_sample["result_ours"] = [f"Error: {str(e)}"] * NUM_RESPONSES
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(res_sample, ensure_ascii=False) + '\n')
                    pbar.update(1)  # Update progress bar

        logger.info(f"Completed: {success_count} prompts with at least one successful response, {len(remaining_data) - success_count} with all errors")

    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        raise

if __name__ == "__main__":
    generate_responses()
