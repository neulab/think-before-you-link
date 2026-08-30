import json
import os
import re
import time
import traceback
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from merlin_eval.config import EvalConfig
from merlin_eval.prompts import (
    create_agentic_prompt,
    create_reasoning_prompt,
    create_search_tool_definition,
)
from merlin_eval.retriever import BaseRetriever
from merlin_eval.scoring import get_score_detailed
from merlin_eval.utils import encode_image_base64, get_image_mime_type


# ============================================================================
# Server health helpers
# ============================================================================

def _is_server_error(e: Exception) -> bool:
    """Return True if *e* looks like the inference server crashed / is unreachable."""
    try:
        import openai as _openai
        if isinstance(e, _openai.APIConnectionError):
            return True
        if isinstance(e, _openai.APIStatusError) and e.status_code in (502, 503, 504):
            return True
    except Exception:
        pass
    if isinstance(e, (ConnectionError, ConnectionRefusedError, ConnectionResetError)):
        return True
    msg = str(e).lower()
    return any(s in msg for s in (
        "connection refused", "connection reset", "broken pipe",
        "server disconnected", "remote end closed",
    ))


def _wait_for_server_health(server_url: str, timeout: float = 300.0) -> bool:
    """Block until the inference server's /health endpoint responds 200.

    *server_url* is the OpenAI-compatible base, e.g.
    ``http://localhost:30000/v1``.  We strip ``/v1`` and probe ``/health``.

    Returns True if the server recovered within *timeout* seconds.
    """
    base = server_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    health_url = f"{base}/health"

    print(f"  [Recovery] Waiting for server at {health_url} "
          f"(timeout={timeout:.0f}s)...", flush=True)
    start = time.time()
    interval = 5  # seconds between probes

    while time.time() - start < timeout:
        try:
            req = urllib.request.urlopen(health_url, timeout=5)
            if req.status == 200:
                elapsed = time.time() - start
                print(f"  [Recovery] Server is back after {elapsed:.0f}s",
                      flush=True)
                return True
        except Exception:
            pass
        time.sleep(interval)

    print(f"  [Recovery] Server did not recover within {timeout:.0f}s",
          flush=True)
    return False


# ============================================================================
# Helpers
# ============================================================================

def _serialize_usage(response) -> Optional[Dict]:
    """Extract token usage from an API response, if available."""
    if not hasattr(response, "usage") or response.usage is None:
        return None
    u = response.usage
    result = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = getattr(u, field, None)
        if val is not None:
            result[field] = val
    return result if result else None


def _extract_content(message) -> Tuple[str, str]:
    """Extract usable text from a response message.

    Returns (extracted_text, source_label).
    source_label is one of: "content", "reasoning_after_think",
    "reasoning_last_line", "reasoning_full", "empty".
    """
    content = message.content
    reasoning = getattr(message, "reasoning_content", None)

    if content and content.strip():
        return content.strip(), "content"

    # content is None/empty — try reasoning_content (thinking models may
    # put everything there when the response is truncated or the server
    # parser is active)
    if reasoning and reasoning.strip():
        if "</think>" in reasoning:
            after = reasoning.split("</think>")[-1].strip()
            if after:
                return after, "reasoning_after_think"
        # Last line may be the answer
        last_line = reasoning.rstrip().rsplit("\n", 1)[-1].strip()
        if last_line:
            return last_line, "reasoning_last_line"
        # Fall back to full reasoning so Layer 2 can try to extract
        return reasoning.strip(), "reasoning_full"

    return "", "empty"


# ============================================================================
# Layer 1 — Agentic RAG loop (tool-calling)
# ============================================================================

def _layer1_agentic_once(
    client: OpenAI,
    entity_name: str,
    article_title: str,
    image_path: str,
    language: str,
    retriever: BaseRetriever,
    config: EvalConfig,
    max_iterations: int = 20,
) -> Tuple[str, List[Dict], List[str], Dict]:
    """Single attempt at agentic reasoning with Wikipedia tool calls.

    Returns (explanation, conversation_log, seen_titles, meta).
    """
    start_time = time.time()

    system_prompt, user_prompt = create_agentic_prompt(
        entity_name, article_title, language, config.use_image,
        max_searches=config.max_searches)
    if config.system_prompt_prefix:
        system_prompt = config.system_prompt_prefix.rstrip() + "\n\n" + system_prompt
    tools = create_search_tool_definition()

    # Build user message content
    if config.use_image:
        image_base64 = encode_image_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
            },
            {"type": "text", "text": user_prompt}
        ]
    else:
        user_content = user_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    conversation_log: List[Dict] = []
    seen_titles: List[str] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_finish_reason = None
    iterations_completed = 0
    searches_done = 0

    for iteration in range(max_iterations):
        # Check if search budget is exhausted
        search_budget_exhausted = (
            config.max_searches > 0 and searches_done >= config.max_searches
        )

        # Force tool call on first iteration if configured (and budget allows)
        if iteration == 0 and config.layer1_force_first_tool and not search_budget_exhausted:
            current_tool_choice = {"type": "function", "function": {"name": "search_wikipedia"}}
        elif search_budget_exhausted:
            current_tool_choice = "none"
        else:
            current_tool_choice = "auto"

        # Build API call kwargs — omit tools entirely when budget exhausted
        api_kwargs = dict(
            model=config.model,
            messages=messages,
            max_tokens=config.layer1_max_tokens,
            timeout=config.api_timeout,
        )
        # Sampling — temperature>0 used for self-consistency; default greedy
        if config.temperature > 0:
            api_kwargs["temperature"] = config.temperature
        if config.top_p < 1.0:
            api_kwargs["top_p"] = config.top_p
        # Family-specific extra_body (e.g., Gemma 4 enable_thinking)
        eb = config.get_extra_body()
        if eb:
            api_kwargs["extra_body"] = eb
        if not search_budget_exhausted:
            api_kwargs["tools"] = tools
            api_kwargs["tool_choice"] = current_tool_choice
        else:
            if iteration > 0:
                print(f"  [Iteration {iteration+1}] Search budget exhausted "
                      f"({searches_done}/{config.max_searches}), "
                      f"requesting final reasoning", flush=True)

        try:
            response = client.chat.completions.create(**api_kwargs)
            choice = response.choices[0]
            assistant_message = choice.message
            finish_reason = choice.finish_reason
            last_finish_reason = finish_reason
            iterations_completed = iteration + 1

            # Token usage
            usage = _serialize_usage(response)
            if usage:
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)

            if finish_reason == "length":
                print(f"  [Iteration {iteration+1}] Hit max_tokens ({config.layer1_max_tokens}), "
                      f"extracting partial content", flush=True)

            if assistant_message.tool_calls:
                print(f"  [Iteration {iteration+1}] Tool calls: "
                      f"{len(assistant_message.tool_calls)}", flush=True)

                # Minimal entry for API messages (no extra fields)
                msg_entry = {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                }
                messages.append(msg_entry)

                # Enriched entry for logging
                log_entry = dict(msg_entry)
                log_entry["reasoning_content"] = getattr(
                    assistant_message, "reasoning_content", None)
                log_entry["finish_reason"] = finish_reason
                log_entry["usage"] = usage
                conversation_log.append(log_entry)

                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        print(f"    -> Bad tool args (skipping): {tool_call.function.arguments[:100]}", flush=True)
                        tool_msg = {
                            "role": "tool",
                            "content": "Error: could not parse arguments.",
                            "tool_call_id": tool_call.id,
                            "function": func_name,
                        }
                        messages.append(tool_msg)
                        # Log with raw bad arguments
                        tool_log = dict(tool_msg)
                        tool_log["error"] = (
                            f"JSONDecodeError: {tool_call.function.arguments[:500]}")
                        conversation_log.append(tool_log)
                        continue

                    if func_name == "search_wikipedia":
                        query = func_args.get("query", "")
                        limit = func_args.get("limit", 10)
                        print(f"    -> Searching: '{query}'", flush=True)

                        try:
                            search_results = retriever.search(
                                query=query, top_k=limit)
                        except (ValueError, Exception) as search_err:
                            # BM25 English stemmer fails on non-English queries.
                            print(f"    -> Search error: {search_err}", flush=True)
                            search_results = []
                            result_text = (
                                "ERROR: Search failed because the query "
                                "contains non-English text. The Wikipedia "
                                "index only supports English queries. "
                                "Please rephrase your search using English "
                                "terms (e.g. translate the entity name or "
                                "use its English equivalent)."
                            )
                            tool_msg = {
                                "role": "tool",
                                "content": result_text,
                                "tool_call_id": tool_call.id,
                                "function": func_name,
                            }
                            messages.append(tool_msg)
                            conversation_log.append(tool_msg)
                            continue

                        for r in search_results:
                            if r['title'] not in seen_titles:
                                seen_titles.append(r['title'])

                        if search_results:
                            parts = []
                            for idx, r in enumerate(search_results, 1):
                                parts.append(
                                    f"{idx}. **{r['title']}**\n"
                                    f"   Description: {r['description']}\n"
                                    f"   Relevance Score: {r['score']:.3f}\n"
                                )
                            result_text = "\n".join(parts)
                        else:
                            result_text = "No results found for this query."

                        tool_msg = {
                            "role": "tool",
                            "content": result_text,
                            "tool_call_id": tool_call.id,
                            "function": func_name,
                        }
                        messages.append(tool_msg)
                        conversation_log.append(tool_msg)
                        searches_done += 1

                continue  # next iteration

            # No tool calls → final answer (or truncated)
            extracted, source = _extract_content(assistant_message)
            if extracted:
                label = "Final reasoning" if finish_reason != "length" else "Truncated reasoning"
                print(f"  [Iteration {iteration+1}] {label} received "
                      f"({len(extracted)} chars)", flush=True)
                msg_entry = {"role": "assistant", "content": extracted}
                messages.append(msg_entry)

                log_entry = {
                    "role": "assistant",
                    "content": extracted,
                    "reasoning_content": getattr(
                        assistant_message, "reasoning_content", None),
                    "finish_reason": finish_reason,
                    "content_source": source,
                    "usage": usage,
                }
                conversation_log.append(log_entry)

                meta = {
                    "iterations": iterations_completed,
                    "searches": searches_done,
                    "finish_reason": finish_reason,
                    "content_source": source,
                    "token_usage": total_usage,
                    "duration_s": round(time.time() - start_time, 2),
                }
                return extracted, conversation_log, seen_titles, meta

            # Truly empty — log and break
            print(f"  [Iteration {iteration+1}] Empty response "
                  f"(finish_reason={finish_reason})", flush=True)
            break

        except Exception as e:
            print(f"  [Iteration {iteration+1}] Exception in Layer 1 agentic: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            if _is_server_error(e):
                print(f"  [Iteration {iteration+1}] Server appears down, "
                      f"waiting for recovery...", flush=True)
                if _wait_for_server_health(config.server_url):
                    print(f"  [Iteration {iteration+1}] Server recovered, "
                          f"retrying iteration...", flush=True)
                    continue  # retry; iteration counter advances by 1
            iterations_completed = iteration + 1
            break

    meta = {
        "iterations": iterations_completed,
        "searches": searches_done,
        "finish_reason": last_finish_reason,
        "content_source": "empty",
        "token_usage": total_usage,
        "duration_s": round(time.time() - start_time, 2),
    }
    return "", conversation_log, seen_titles, meta


def layer1_agentic(
    client: OpenAI,
    entity_name: str,
    article_title: str,
    image_path: str,
    language: str,
    retriever: BaseRetriever,
    config: EvalConfig,
    max_iterations: int = 20,
) -> Tuple[str, List[Dict], List[str], Dict]:
    """Agentic reasoning with retries.

    Returns (explanation, conversation_log, seen_titles, meta).
    """
    start_time = time.time()
    total_attempts = 1 + config.layer1_retries
    last_log: List[Dict] = []
    last_titles: List[str] = []
    last_meta: Dict = {}

    for attempt in range(total_attempts):
        if attempt > 0:
            print(f"  [Layer 1] Retry {attempt}/{config.layer1_retries}...", flush=True)

        explanation, conv_log, seen, meta = _layer1_agentic_once(
            client=client,
            entity_name=entity_name,
            article_title=article_title,
            image_path=image_path,
            language=language,
            retriever=retriever,
            config=config,
            max_iterations=max_iterations,
        )
        last_log = conv_log
        last_titles = seen
        last_meta = meta

        if explanation:
            meta["attempts"] = attempt + 1
            meta["total_duration_s"] = round(time.time() - start_time, 2)
            return explanation, conv_log, seen, meta

        print(f"  [Layer 1] Attempt {attempt+1}/{total_attempts} returned empty", flush=True)

    last_meta["attempts"] = total_attempts
    last_meta["total_duration_s"] = round(time.time() - start_time, 2)
    return "", last_log, last_titles, last_meta


# ============================================================================
# Layer 1 — Single-shot reasoning (no RAG)
# ============================================================================

def _layer1_reasoning_once(
    client: OpenAI,
    entity_name: str,
    article_title: str,
    image_path: str,
    language: str,
    config: EvalConfig,
) -> Tuple[str, List[Dict], Dict]:
    """Single attempt at reasoning without RAG.

    Returns (explanation, conversation_log, meta).
    """
    start_time = time.time()

    system_prompt, user_prompt = create_reasoning_prompt(
        entity_name, article_title, language, config.use_image)
    if config.system_prompt_prefix:
        system_prompt = config.system_prompt_prefix.rstrip() + "\n\n" + system_prompt

    if config.use_image:
        image_base64 = encode_image_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
            },
            {"type": "text", "text": user_prompt}
        ]
    else:
        user_content = user_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    conversation_log: List[Dict] = []

    try:
        print(f"  [Layer 1] Generating reasoning (no RAG)...", flush=True)
        api_kwargs = dict(
            model=config.model,
            messages=messages,
            max_tokens=config.layer1_max_tokens,
            timeout=config.api_timeout,
        )
        if config.temperature > 0:
            api_kwargs["temperature"] = config.temperature
        if config.top_p < 1.0:
            api_kwargs["top_p"] = config.top_p
        eb = config.get_extra_body()
        if eb:
            api_kwargs["extra_body"] = eb
        response = client.chat.completions.create(**api_kwargs)
        choice = response.choices[0]
        assistant_message = choice.message
        finish_reason = choice.finish_reason
        usage = _serialize_usage(response)

        if finish_reason == "length":
            print(f"  [Layer 1] Hit max_tokens ({config.layer1_max_tokens}), "
                  f"extracting partial content", flush=True)

        extracted, source = _extract_content(assistant_message)
        if extracted:
            label = "Reasoning" if finish_reason != "length" else "Truncated reasoning"
            print(f"  [Layer 1] {label} received ({len(extracted)} chars)", flush=True)
            entry = {
                "role": "assistant",
                "content": extracted,
                "reasoning_content": getattr(
                    assistant_message, "reasoning_content", None),
                "finish_reason": finish_reason,
                "content_source": source,
                "usage": usage,
            }
            conversation_log.append(entry)
            meta = {
                "iterations": 1,
                "finish_reason": finish_reason,
                "content_source": source,
                "token_usage": usage,
                "duration_s": round(time.time() - start_time, 2),
            }
            return extracted, conversation_log, meta

        print(f"  [Layer 1] Empty response (finish_reason={finish_reason})", flush=True)
        meta = {
            "iterations": 1,
            "finish_reason": finish_reason,
            "content_source": "empty",
            "token_usage": usage,
            "duration_s": round(time.time() - start_time, 2),
        }
        return "", conversation_log, meta

    except Exception as e:
        print(f"  [Layer 1] Exception: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        if _is_server_error(e):
            _wait_for_server_health(config.server_url)
            # Server may be back; layer1_reasoning() retry loop will re-call us.
        meta = {
            "iterations": 1,
            "finish_reason": None,
            "content_source": "empty",
            "token_usage": None,
            "duration_s": round(time.time() - start_time, 2),
            "error": f"{type(e).__name__}: {e}",
        }
        return "", conversation_log, meta


def layer1_reasoning(
    client: OpenAI,
    entity_name: str,
    article_title: str,
    image_path: str,
    language: str,
    config: EvalConfig,
) -> Tuple[str, List[Dict], Dict]:
    """Single-shot reasoning with retries.

    Returns (explanation, conversation_log, meta).
    """
    start_time = time.time()
    total_attempts = 1 + config.layer1_retries
    last_log: List[Dict] = []
    last_meta: Dict = {}

    for attempt in range(total_attempts):
        if attempt > 0:
            print(f"  [Layer 1] Retry {attempt}/{config.layer1_retries}...", flush=True)

        explanation, conv_log, meta = _layer1_reasoning_once(
            client=client,
            entity_name=entity_name,
            article_title=article_title,
            image_path=image_path,
            language=language,
            config=config,
        )
        last_log = conv_log
        last_meta = meta

        if explanation:
            meta["attempts"] = attempt + 1
            meta["total_duration_s"] = round(time.time() - start_time, 2)
            return explanation, conv_log, meta

        print(f"  [Layer 1] Attempt {attempt+1}/{total_attempts} returned empty", flush=True)

    last_meta["attempts"] = total_attempts
    last_meta["total_duration_s"] = round(time.time() - start_time, 2)
    return "", last_log, last_meta


# ============================================================================
# Layer 2 — Extract English Wikipedia title
# ============================================================================

def _escape_for_regex(title: str) -> str:
    """Escape regex metacharacters (compatible with SGLang/Rust regex engine)."""
    metacharacters = r'\.^$*+?{}[]|()'
    result = []
    for char in title:
        if char in metacharacters:
            result.append('\\')
        result.append(char)
    return ''.join(result)


def _build_title_regex(seen_titles: List[str]) -> str:
    """Build alternation regex from seen titles for constrained decoding."""
    if not seen_titles:
        return ".*"
    escaped = [_escape_for_regex(t) for t in seen_titles]
    return "(" + "|".join(escaped) + ")"


def layer2_extract(
    client: OpenAI,
    entity_explanation: str,
    entity_name: str,
    language: str,
    seen_titles: List[str],
    config: EvalConfig,
) -> Tuple[str, Dict]:
    """Extract the English Wikipedia title from Layer 1 explanation.

    Returns (prediction, layer2_meta).
    """
    start_time = time.time()
    lang = language.capitalize()

    # Trie mode doesn't need seen_titles; regex mode does
    if config.constrained_mode == "trie":
        use_constrained = config.constrained and config.trie_processor_str
    else:
        use_constrained = config.constrained and seen_titles

    if use_constrained:
        system_prompt = (
            "You are a precise entity extraction system. Your task is to extract "
            "the EXACT English Wikipedia page title from an entity analysis.\n\n"
            "You will receive a detailed analysis of an entity. Your task is to "
            "extract the EXACT ENGLISH WIKIPEDIA page title that was identified.\n\n"
            "You MUST output ONLY the exact English Wikipedia page title, nothing else. "
            "No explanation, no punctuation, no quotes - just the title."
        )
        user_prompt = (
            f'**ENTITY EXTRACTION TASK**\n\n'
            f'Original Entity Name (in {lang}): "{entity_name}"\n\n'
            f'**Analysis to extract from:**\n{entity_explanation}\n\n'
            f'---\n\n'
            f'**YOUR TASK:**\n'
            f'Based on the analysis above, output the EXACT English Wikipedia page '
            f'title that this entity corresponds to.\n\n'
            f'Output ONLY the exact title. Nothing else.'
        )
    else:
        system_prompt = (
            "You are a precise entity extraction system. Your task is to extract "
            "the EXACT English Wikipedia page title from an entity analysis.\n\n"
            "You will receive a detailed analysis of an entity. Your task is to "
            "extract the EXACT ENGLISH WIKIPEDIA page title that was identified.\n\n"
            "Think and reason as much as you need, but your final answer should "
            "ONLY be the exact English Wikipedia page title, nothing else."
        )
        user_prompt = (
            f'**ENTITY EXTRACTION TASK**\n\n'
            f'Original Entity Name (in {lang}): "{entity_name}"\n\n'
            f'**Proposed English Wikipedia Title (extract from here):**\n'
            f'{entity_explanation}\n\n'
            f'---\n\n'
            f'**YOUR TASK:**\n'
            f'Based on the analysis above, extract the EXACT English Wikipedia page '
            f'title that this entity corresponds to.\n\n'
            f'Reason as long as you wish, but after that, output ONLY the exact '
            f'English Wikipedia page title, nothing else. DO NOT OUTPUT EXTRA '
            f'EXPLANATION TEXT OR ANYTHING ELSE. ONLY OUTPUT THE WIKIPEDIA PAGE TITLE.'
        )

    use_trie = use_constrained and config.constrained_mode == "trie"
    use_regex = use_constrained and config.constrained_mode == "regex"

    # Determine mode label for logging
    if use_trie:
        mode = "trie"
    elif use_regex:
        mode = "regex"
    else:
        mode = "free"

    try:
        n_titles = len(seen_titles) if seen_titles else 0
        if use_trie:
            print(f"  [Layer 2] Extracting (trie-constrained, all Wikipedia)...", flush=True)
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={
                    "custom_logit_processor": config.trie_processor_str,
                    "custom_params": {
                        "trie_url": config.trie_server_url,
                    },
                },
                # Greedy decoding: with 7.4M entities in the trie, sampling
                # (temperature > 0) picks wrong continuations (e.g., "BHar"
                # instead of "Bihar").  Greedy ensures the highest-probability
                # valid token is always selected.
                max_tokens=8000,
                temperature=0,
            )
        elif use_regex:
            print(f"  [Layer 2] Extracting (regex-constrained to {n_titles} seen titles)...", flush=True)
            title_regex = _build_title_regex(seen_titles)
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={
                    "regex": title_regex,
                    "top_k": 20,
                    "repitition_penalty": 1.0,
                    "presence_penalty": 1.5,
                },
                max_tokens=40000,
                temperature=0.6,
                top_p=0.95,
            )
        else:
            print(f"  [Layer 2] Extracting English Wikipedia title...", flush=True)
            l2_kwargs = dict(
                model=config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=8000,
            )
            eb = config.get_extra_body()
            if eb:
                l2_kwargs["extra_body"] = eb
            response = client.chat.completions.create(**l2_kwargs)

        msg = response.choices[0].message
        api_content = msg.content
        reasoning = getattr(msg, "reasoning_content", None)
        finish_reason = response.choices[0].finish_reason
        usage = _serialize_usage(response)

        raw_content = api_content

        # When SGLang's --reasoning-parser is active WITH a custom logit
        # processor, the parser may put everything (including post-think
        # entity tokens) into reasoning_content, leaving content = None.
        # Also treat empty-string content the same as None.
        if not raw_content or not raw_content.strip():
            if reasoning:
                if "</think>" in reasoning:
                    # Entity title is after the last </think> in reasoning
                    raw_content = reasoning.split("</think>")[-1]
                    print(f"  [Layer 2] content=None; extracted from reasoning_content", flush=True)
                else:
                    # Parser already stripped </think> — entity may be at the
                    # very end of reasoning_content (last line).
                    last_line = reasoning.rstrip().rsplit("\n", 1)[-1].strip()
                    if last_line:
                        raw_content = last_line
                        print(f"  [Layer 2] content=None; using last line of reasoning", flush=True)
                    else:
                        raise ValueError(
                            "Layer 2 content is None and reasoning_content "
                            f"has no extractable title (last 100 chars: "
                            f"...{reasoning[-100:]})"
                        )
            else:
                raise ValueError("Layer 2 returned no content and no reasoning_content")

        prediction = raw_content.strip()

        # Strip <think>...</think> if still present in the text
        if "</think>" in prediction:
            prediction = prediction.split("</think>")[-1].strip()

        print(f"  [Layer 2] Extracted: {prediction}", flush=True)

        layer2_meta = {
            "api_content": api_content,
            "api_reasoning_content": reasoning,
            "mode": mode,
            "finish_reason": finish_reason,
            "token_usage": usage,
            "duration_s": round(time.time() - start_time, 2),
        }
        return prediction, layer2_meta

    except Exception as e:
        print(f"Error in Layer 2 extraction: {e}", flush=True)
        if _is_server_error(e):
            _wait_for_server_health(config.server_url)
            # Server may be back; this sample uses regex fallback, but
            # the next sample's API calls will succeed.
        fallback = _fallback_title_extraction(
            entity_explanation, entity_name, seen_titles)
        print(f"  [Layer 2] Falling back to regex extraction: {fallback}", flush=True)

        layer2_meta = {
            "api_content": None,
            "api_reasoning_content": None,
            "mode": "fallback_regex",
            "finish_reason": None,
            "token_usage": None,
            "duration_s": round(time.time() - start_time, 2),
            "error": f"{type(e).__name__}: {e}",
        }
        return fallback, layer2_meta


def _fallback_title_extraction(explanation: str, entity_name: str,
                               seen_titles: Optional[List[str]] = None) -> str:
    """Regex-based fallback when Layer 2 LLM call fails."""
    patterns = [
        r'Wikipedia page[:\s]+["\']?([^"\'\.]+)["\']?',
        r'refers to[:\s]+["\']?([^"\'\.]+)["\']?',
        r'entity is[:\s]+["\']?([^"\'\.]+)["\']?',
        r'identified as[:\s]+["\']?([^"\'\.]+)["\']?',
        r'corresponds to[:\s]+["\']?([^"\'\.]+)["\']?',
    ]

    for pattern in patterns:
        match = re.search(pattern, explanation, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if seen_titles:
                extracted_lower = extracted.lower()
                for title in seen_titles:
                    if (title.lower() == extracted_lower or
                            title.lower() in extracted_lower):
                        return title
            return extracted

    if seen_titles:
        for title in seen_titles:
            if title.lower() in explanation.lower():
                return title

    return entity_name


# ============================================================================
# Process a single sample (dispatches to correct Layer 1 + Layer 2)
# ============================================================================

def process_single_sample(
    sample: Dict,
    client: OpenAI,
    language: str,
    images_dir: str,
    aliases_cache: Dict,
    entity_links: Dict,
    retriever: Optional[BaseRetriever],
    config: EvalConfig,
) -> Dict:
    """Run the full 2-layer pipeline on one sample."""
    sample_start = time.time()

    entity_name = sample["Entity Name"]
    article_title = sample["Article Title"]
    image_name = sample["Image Name"]
    target = sample["English Wikipedia Title"]
    wikidata_id = sample["Wikidata ID"]

    image_path = os.path.join(images_dir, language.lower(), image_name)

    print(f"\n--- [{language}] Entity: \"{entity_name}\" | Article: \"{article_title}\" | Target: {target}", flush=True)

    # Check image exists (needed even in no-image mode for the result record)
    if config.use_image and not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return _error_result(
            sample, entity_links, aliases_cache, "Image not found",
            duration_s=round(time.time() - sample_start, 2))

    # Track state for error handler
    conversation_log: List[Dict] = []
    seen_titles: List[str] = []
    layer1_meta: Dict = {}

    try:
        # --- Layer 1 ---
        if config.rag and retriever is not None:
            entity_explanation, conversation_log, seen_titles, layer1_meta = \
                layer1_agentic(
                    client=client,
                    entity_name=entity_name,
                    article_title=article_title,
                    image_path=image_path,
                    language=language,
                    retriever=retriever,
                    config=config,
                )
        else:
            entity_explanation, conversation_log, layer1_meta = \
                layer1_reasoning(
                    client=client,
                    entity_name=entity_name,
                    article_title=article_title,
                    image_path=image_path,
                    language=language,
                    config=config,
                )
            seen_titles = []

        if not entity_explanation:
            # Layer 1 returned nothing after all retries — log and bail
            print(f"  [WARN] Layer 1 returned empty after all retries", flush=True)
            aliases = aliases_cache.get(wikidata_id, [])
            return {
                "title": article_title,
                "entity": entity_name,
                "layer1_explanation": "",
                "layer2_prediction": "",
                "image": image_name,
                "target": target,
                "is_correct": False,
                "wikidata_id": wikidata_id,
                "wikidata_incoming_links": entity_links.get(
                    wikidata_id, {}).get("wikidata_incoming_links", 0),
                "aliases": aliases,
                "conversation_log": conversation_log,
                "seen_titles": seen_titles,
                "error": "Layer 1 failed to produce explanation after all retries",
                "layer1_meta": layer1_meta,
                "layer2_meta": None,
                "score_detail": None,
                "total_duration_s": round(time.time() - sample_start, 2),
            }

        # --- Layer 2 ---
        prediction, layer2_meta = layer2_extract(
            client=client,
            entity_explanation=entity_explanation,
            entity_name=entity_name,
            language=language,
            seen_titles=seen_titles,
            config=config,
        )

        # --- Score ---
        aliases = aliases_cache.get(wikidata_id, [])
        is_correct, match_type, pred_normalized, target_normalized = \
            get_score_detailed(
                pred=prediction,
                target=target,
                aliases=aliases,
                use_target_in_pred=config.target_in_pred,
                use_aliases=config.use_alias,
            )

        symbol = "CORRECT" if is_correct else "WRONG"
        print(f"  [{symbol}] Pred: {prediction} | Target: {target}", flush=True)

        result = {
            # Core fields (backward compatible)
            "title": article_title,
            "entity": entity_name,
            "layer1_explanation": entity_explanation,
            "layer2_prediction": prediction,
            "image": image_name,
            "target": target,
            "is_correct": is_correct,
            "wikidata_id": wikidata_id,
            "wikidata_incoming_links": entity_links.get(
                wikidata_id, {}).get("wikidata_incoming_links", 0),
            "aliases": aliases,
            "conversation_log": conversation_log,
            "seen_titles": seen_titles,
            # New: metadata
            "layer1_meta": layer1_meta,
            "layer2_meta": layer2_meta,
            "score_detail": {
                "match_type": match_type,
                "pred_normalized": pred_normalized,
                "target_normalized": target_normalized,
            },
            "total_duration_s": round(time.time() - sample_start, 2),
        }
        return result

    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        tb_str = traceback.format_exc()
        return _error_result(
            sample, entity_links, aliases_cache, str(e),
            conversation_log=conversation_log,
            seen_titles=seen_titles,
            layer1_meta=layer1_meta,
            duration_s=round(time.time() - sample_start, 2),
            traceback_str=tb_str,
        )


def _error_result(
    sample: Dict,
    entity_links: Dict,
    aliases_cache: Dict,
    error_msg: str,
    conversation_log: Optional[List[Dict]] = None,
    seen_titles: Optional[List[str]] = None,
    layer1_meta: Optional[Dict] = None,
    duration_s: Optional[float] = None,
    traceback_str: Optional[str] = None,
) -> Dict:
    wikidata_id = sample["Wikidata ID"]
    return {
        "title": sample["Article Title"],
        "entity": sample["Entity Name"],
        "layer1_explanation": "",
        "layer2_prediction": "",
        "image": sample["Image Name"],
        "target": sample["English Wikipedia Title"],
        "is_correct": False,
        "wikidata_id": wikidata_id,
        "wikidata_incoming_links": entity_links.get(
            wikidata_id, {}).get("wikidata_incoming_links", 0),
        "aliases": aliases_cache.get(wikidata_id, []),
        "error": error_msg,
        "error_traceback": traceback_str,
        "conversation_log": conversation_log or [],
        "seen_titles": seen_titles or [],
        "layer1_meta": layer1_meta,
        "layer2_meta": None,
        "score_detail": None,
        "total_duration_s": duration_s,
    }
