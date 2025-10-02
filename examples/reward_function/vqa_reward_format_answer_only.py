import re
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
from itertools import cycle

# ==============================================================================
# --- 全局调试开关 ---
# ==============================================================================
IS_DUMMY_MODE = os.environ.get("DEBUG_REWARD", "0") == "1"


# ==============================================================================
# --- 依赖安装 ---
# ==============================================================================
if not IS_DUMMY_MODE:
    try:
        import openai
    except ImportError:
        print("openai library not found. Installing it now...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
        import openai

# ==============================================================================
# --- 日志和模型初始化 ---
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(f"RewardFunc_PID_{os.getpid()}")

log.info(f"--- Reward Function Initializing (Multi-Key Polling) --- | IS_DUMMY_MODE: {IS_DUMMY_MODE}")

# ==============================================================================
# --- ★★★ LLM Judge 多 Key 轮询配置 ★★★ ---
# ==============================================================================
clients = []
_client_cycler = None
LLM_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"

if not IS_DUMMY_MODE:
    # 从环境变量读取以逗号分隔的 API Keys
    api_keys_str = os.environ.get("NEBIUS_API_KEYS", "")
    api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
    
    API_BASE_URL = "https://api.studio.nebius.com/v1/"

    if not api_keys:
        log.warning("CRITICAL: NEBIUS_API_KEYS environment variable not set or empty. LLM Judge will fail.")
    else:
        log.info(f"Found {len(api_keys)} API keys. Initializing clients...")
        for i, key in enumerate(api_keys):
            try:
                # 为每个 Key 创建一个独立的客户端实例
                client = openai.OpenAI(api_key=key, base_url=API_BASE_URL)
                clients.append(client)
                log.info(f"✅ Client {i+1}/{len(api_keys)} initialized successfully.")
            except Exception as e:
                log.error(f"CRITICAL: Failed to initialize OpenAI client for key ending in '...{key[-4:]}': {e}", exc_info=True)
        
        if clients:
            # 创建一个可以无限循环的客户端迭代器
            _client_cycler = cycle(clients)
            log.info(f"Successfully created a cycler for {len(clients)} clients.")
else:
    log.info("Running in Dummy Mode. LLM Judge clients will not be initialized.")

LLM_JUDGE_SYSTEM_PROMPT = """You are an automated evaluation system. Your ONLY task is to compare a 'Generated Answer' to a 'Reference Answer' and determine if they are equivalent.

Follow these rules STRICTLY:
1.  For EACH item you evaluate, you MUST output EXACTLY ONE `<judge>` tag.
2.  Inside the tag, write 'True' if the generated answer is correct or equivalent to the reference.
3.  Inside the tag, write 'False' if the generated answer is incorrect, incomplete, or different from the reference.
4.  Your entire response MUST consist ONLY of a series of `<judge>` tags, one for each item.
5.  ABSOLUTELY DO NOT provide any text, explanation, reasoning, markdown, or any characters outside of the `<judge>True</judge>` or `<judge>False</judge>` tags.

Example for 3 items:
<judge>True</judge>
<judge>False</judge>
<judge>True</judge>
"""

# ==============================================================================
# --- 辅助函数 ---
# ==============================================================================
def get_llm_answer_scores(prompts_for_eval: List[Dict]) -> List[float]:
    """
    对一个 batch 的 item 统一打分，内部实现了分批处理、自动重试和API Key轮询。
    """
    if not prompts_for_eval:
        return []
    if _client_cycler is None:
        log.warning("LLM Judge clients not initialized. Returning 0.0 for all scores.")
        return [0.0] * len(prompts_for_eval)

    LLM_JUDGE_BATCH_SIZE = 8
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1
    all_scores = []

    log.info(f"Total items to evaluate: {len(prompts_for_eval)}. Processing in chunks of {LLM_JUDGE_BATCH_SIZE}...")

    for i in range(0, len(prompts_for_eval), LLM_JUDGE_BATCH_SIZE):
        chunk = prompts_for_eval[i:i + LLM_JUDGE_BATCH_SIZE]
        
        current_client = next(_client_cycler)
        client_index = clients.index(current_client) + 1
        log.info(f"--- Processing chunk {i//LLM_JUDGE_BATCH_SIZE + 1} with client {client_index}/{len(clients)} ---")
        
        user_message_parts = ["Please evaluate the following items..."]
        for j, prompt in enumerate(chunk):
            item_text = (f"\n--- ITEM {j+1} ---\n## Query\n{prompt.get('query', 'N/A')}\n\n## Reference Answer\n{prompt.get('reference_answer', 'N/A')}\n\n## Generated Answer\n{prompt.get('generated_answer', 'N/A')}")
            user_message_parts.append(item_text)
        user_message = "".join(user_message_parts)

        chunk_scores = []
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                log.info(f"Attempt {attempt + 1}/{MAX_RETRIES} for chunk...")
                response = current_client.chat.completions.create(
                    model=LLM_JUDGE_MODEL,
                    messages=[{"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
                    max_tokens=len(chunk) * 100,
                    temperature=0.0
                )
                full_response_text = response.choices[0].message.content
                duration = time.time() - start_time
                log.info(f"LLM Judge API call for chunk (Attempt {attempt + 1}) took {duration:.2f} seconds. Response: '{full_response_text}'")

                judges = re.findall(r'<judge>(True|False|true|false)</judge>', full_response_text)
                
                if len(judges) == len(chunk):
                    chunk_scores = [1.0 if j.lower() == 'true' else 0.0 for j in judges]
                    log.info(f"✅ Success on attempt {attempt + 1}. Parsed scores for chunk: {chunk_scores}")
                    break
                else:
                    log.warning(f"❗ Mismatch on attempt {attempt + 1}: Expected {len(chunk)} scores, but parsed {len(judges)}. Retrying...")

            except Exception as e:
                log.error(f"CRITICAL ERROR during API call on attempt {attempt + 1}: {e}", exc_info=True)

            if attempt < MAX_RETRIES - 1:
                log.info(f"Waiting for {RETRY_DELAY_SECONDS} seconds before next attempt...")
                time.sleep(RETRY_DELAY_SECONDS)
        
        if not chunk_scores:
            log.error(f"❌ All {MAX_RETRIES} attempts failed for the chunk. Assigning 0.0 scores.")
            chunk_scores = [0.0] * len(chunk)
            
        all_scores.extend(chunk_scores)
            
    return all_scores

def parse_generation(response: str) -> Dict[str, Optional[str]]:
    """解析包含<think>, <description>, <answer>标签的生成文本。"""
    think = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    description = re.search(r'<description>(.*?)</description>', response, re.DOTALL)
    answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    return {
        "think": think.group(1).strip() if think else None,
        "description": description.group(1).strip() if description else None,
        "answer": answer.group(1).strip() if answer else None,
    }

def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    为一批样本计算奖励分数 (batch 模式)。
    遵循 EasyR1 框架，从 'ground_truth' 字段提取参考答案。
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("This reward function expects a list of dicts for `reward_type=batch`.")

    num_items = len(reward_inputs)
    log.info(f"--- compute_score (Multi-Key, Direct Answer) started. Received {num_items} items. ---")
    if not num_items:
        return []

    W_ANSWER = 0.8
    W_FORMAT = 0.2

    # --- 步骤 1: 预处理 ---
    valid_inputs = [item for item in reward_inputs if item is not None]
    if not valid_inputs:
        log.warning("No valid items found in reward_inputs. Returning all zeros.")
        return [{"overall": 0.0, "format_score": 0.0, "answer_score": 0.0}] * num_items
        
    parsed_items = [parse_generation(item.get("response", "")) for item in valid_inputs]
    format_scores = [1.0 if all(p.get(k) is not None for k in ["think", "description", "answer"]) else 0.0 for p in parsed_items]

    # --- 步骤 2: 批量计算 Answer 分数 ---
    answer_scores = [0.0] * len(valid_inputs)
    if IS_DUMMY_MODE:
        for i, parsed in enumerate(parsed_items):
            if format_scores[i] > 0 and parsed.get("answer"):
                answer_scores[i] = min(1.0, len(parsed["answer"]) / 10.0)
    else:
        prompts_for_llm_eval = []
        indices_to_update_in_answers = []
        
        for i, (item, parsed) in enumerate(zip(valid_inputs, parsed_items)):
            if parsed.get("answer"):
                reference_answer = item.get("ground_truth", "")
                if not reference_answer:
                    log.warning(f"Sample index {i} is missing the 'ground_truth' field for reference. Skipping LLM evaluation for this item.")
                    continue
                
                prompts_for_llm_eval.append({
                    "query": item.get("prompt", ""),
                    "reference_answer": reference_answer,
                    "generated_answer": parsed["answer"]
                })
                indices_to_update_in_answers.append(i)
        
        if prompts_for_llm_eval:
            llm_scores_list = get_llm_answer_scores(prompts_for_llm_eval)
            for score, idx in zip(llm_scores_list, indices_to_update_in_answers):
                answer_scores[idx] = score

    # --- 步骤 3: 计算总奖励并构建返回列表 ---
    final_scores = []
    log.info("\n--- Reward Calculation Details (Multi-Key, Direct Answer) ---")
    
    valid_item_idx = 0
    for i in range(num_items):
        if reward_inputs[i] is None:
            score_dict = {"overall": 0.0, "format_score": 0.0, "answer_score": 0.0}
            parsed_answer_for_log = "N/A (Invalid Item)"
        else:
            total_reward = (W_ANSWER * answer_scores[valid_item_idx]) + (W_FORMAT * format_scores[valid_item_idx])
            score_dict = {
                "overall": total_reward,
                "format_score": format_scores[valid_item_idx],
                "answer_score": answer_scores[valid_item_idx]
            }
            parsed_answer_for_log = (parsed_items[valid_item_idx].get("answer") or "N/A")[:100]
            valid_item_idx += 1
        
        final_scores.append(score_dict)

        log_entry = {
            "sample_index": i, "mode": "Dummy" if IS_DUMMY_MODE else "Real (Multi-Key)", "scores": score_dict,
            "parsed_answer": parsed_answer_for_log
        }
        log.info(json.dumps(log_entry, indent=2))
        
    log.info(f"--- compute_score (Multi-Key, Direct Answer) finished. ---")
    return final_scores