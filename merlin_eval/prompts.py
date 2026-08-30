from typing import Tuple


def create_search_tool_definition() -> list:
    """OpenAI-compatible tool definition for Wikipedia search."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_wikipedia",
                "description": (
                    "Search English Wikipedia for pages matching a query. "
                    "Returns Wikipedia page titles with their descriptions. "
                    "Use this to find the correct English Wikipedia page for an entity. "
                    "Since this searches ENGLISH Wikipedia, translate or transliterate "
                    "entity names to English for best results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The search query to find Wikipedia pages. Should be in ENGLISH. "
                                "Translate or transliterate non-English entity names. "
                                "Include relevant context like profession, location, or field "
                                "to disambiguate."
                            )
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]


# ---------------------------------------------------------------------------
# Shared prompt fragments
# ---------------------------------------------------------------------------

_MISTAKE_EXAMPLE = """\
**Example of this mistake:**
- Context: "Taylor Swift performed at Madison Square Garden"
- Entity mention: "Madison Square Garden"
- Image: Shows Taylor Swift
- WRONG: Linking to "Taylor Swift" because she's prominent
- CORRECT: Linking to "Madison Square Garden" because that's the entity mention"""

_MISTAKE_EXAMPLE_NO_IMAGE = """\
**Example of this mistake:**
- Context: "Taylor Swift performed at Madison Square Garden"
- Entity mention: "Madison Square Garden"
- WRONG: Linking to "Taylor Swift" because she's prominent
- CORRECT: Linking to "Madison Square Garden" because that's the entity mention"""


def _verify_block() -> str:
    return """\
### Step {n}: VERIFY YOUR ANSWER
Before concluding, perform this sanity check:
- [ ] Does my answer match the TYPE of entity in the mention? (If mention is a place, answer should be a place)
- [ ] Am I linking the entity MENTION, not some other entity from the context?
- [ ] If the context says "[Person] visited [Place]" and the entity mention is [Place], am I linking [Place] and NOT [Person]?"""


# ---------------------------------------------------------------------------
# Agentic prompt (RAG mode — includes search instructions)
# ---------------------------------------------------------------------------

def create_agentic_prompt(entity_name: str, article_title: str,
                          language: str, use_image: bool,
                          max_searches: int = 0) -> Tuple[str, str]:
    """System + user prompt for the agentic RAG loop."""
    lang = language.capitalize()

    # --- system prompt ---
    if use_image:
        input_list = f"""\
1. An **ENTITY MENTION** in {lang} — THIS IS WHAT YOU MUST LINK. Nothing else.
2. A **CONTEXT** in {lang} — This provides surrounding information to help disambiguate.
3. An **IMAGE** — This MAY or MAY NOT be relevant. Treat it with skepticism."""
        mistake_block = _MISTAKE_EXAMPLE
        image_step = f"""\
### Step 3: CRITICALLY EVALUATE THE IMAGE
The image may be:
- DIRECTLY relevant (shows the entity mention itself)
- INDIRECTLY relevant (shows related context)
- MISLEADING (shows a different entity from the context)
- USELESS (unrelated or generic)

Ask yourself: "Does this image show the ENTITY MENTION, or does it show something ELSE from the context?"

If the image shows a person but your entity mention is not that person, the image is probably not helpful for your task.

"""
        translate_step_n = 4
        search_step_n = 5
        verify_step_n = 6
    else:
        input_list = f"""\
1. An **ENTITY MENTION** in {lang} — THIS IS WHAT YOU MUST LINK. Nothing else.
2. A **CONTEXT** in {lang} — This provides surrounding information to help disambiguate."""
        mistake_block = _MISTAKE_EXAMPLE_NO_IMAGE
        image_step = ""
        translate_step_n = 3
        search_step_n = 4
        verify_step_n = 5

    if max_searches == 1:
        search_instructions = """\
You have exactly ONE search opportunity. Make it count:
- Combine the translated/transliterated entity name with key disambiguating context
- Include the entity type (person, place, organization, etc.) if helpful
- Use English terms for best results

After receiving search results, carefully analyze all candidates and select the best match based on your reasoning."""
    else:
        search_instructions = """\
ALWAYS make multiple targeted searches:
1. First search: Direct translation/transliteration of entity mention
2. Second search: Add disambiguating context (type, location, field, era)
3. Third search: Try alternative spellings or related terms
4. Additional searches as needed based on results

Remember, you must make at least 3 search attempts before concluding. Ideally you should make more to be thorough."""

    system_prompt = f"""\
You are an expert cross-lingual entity linking specialist. Your task is to find the correct ENGLISH Wikipedia page for a SPECIFIC entity mention.

## CRITICAL: UNDERSTANDING YOUR TASK

You will receive:
{input_list}

## THE #1 MISTAKE TO AVOID

The entity mention is often NOT the main subject of the context. DO NOT link the wrong entity.

**WRONG approach:** "The context mentions [famous person]{', and the image shows [famous person],' if use_image else ''} so I should link [famous person]."

**CORRECT approach:** "The entity mention is '[specific term]'. Even though [famous person] appears in the context{' and image' if use_image else ''}, I must find the Wikipedia page for '[specific term]' specifically."

{mistake_block}

## YOUR METHODOLOGY

### Step 1: LOCK IN THE ENTITY MENTION
Before doing ANYTHING else, answer these questions:
- What EXACT phrase is the entity mention?
- What TYPE of entity is this? (person / place / organization / event / object / concept / product / other)
- Is this entity the SUBJECT of the context, or something else mentioned in it?

Write this down explicitly. Return to it frequently.

### Step 2: ANALYZE CONTEXT FOR DISAMBIGUATION
The context tells you ABOUT the entity mention:
- What role does the entity mention play in the context?
- What clues help identify WHICH specific entity this refers to?
- Are there related entities mentioned that might help narrow down the search?

DO NOT get distracted by other entities in the context. They are there to help you identify the entity mention, not to replace it.

{image_step}### Step {translate_step_n}: TRANSLATE/TRANSLITERATE FOR ENGLISH WIKIPEDIA SEARCH
Since you're searching ENGLISH Wikipedia:
- Transliterate names phonetically to English
- Use common English equivalents for places
- Translate concepts and common nouns
- For products/brands, use the English market name

### Step {search_step_n}: SEARCH STRATEGICALLY
{search_instructions}

### Step {verify_step_n}: VERIFY YOUR ANSWER
Before concluding, perform this sanity check:
- [ ] Does my answer match the TYPE of entity in the mention? (If mention is a place, answer should be a place)
- [ ] Am I linking the entity MENTION, not some other entity from the context?
- [ ] If the context says "[Person] visited [Place]" and the entity mention is [Place], am I linking [Place] and NOT [Person]?

## OUTPUT FORMAT

After your investigation, state your conclusion clearly:
"The entity mention '[ENTITY]' refers to: [ENGLISH WIKIPEDIA PAGE TITLE] because [verbose reasoning that confirms you're linking the correct entity]"

Begin your analysis now. Remember: LINK THE ENTITY MENTION, NOT OTHER ENTITIES IN THE CONTEXT."""

    # --- user prompt ---
    if use_image:
        image_section = """\

### IMAGE:
An image is attached. Evaluate whether it shows the ENTITY MENTION specifically, or whether it shows something else from the context. Be skeptical -- the image may be misleading.
"""
        task_second = "2. **SECOND:** Analyze if the image is relevant to the entity mention (not just to the context)"
        task_third = f'3. **THIRD:** Translate/transliterate "{entity_name}" to English for searching'
        if max_searches == 1:
            task_fourth = "4. **FOURTH:** Search English Wikipedia once with your best query"
        else:
            task_fourth = "4. **FOURTH:** Search English Wikipedia multiple times with different strategies"
        task_fifth = "5. **FIFTH:** Verify your final answer is for the entity mention, not a different entity from the context"
        pitfall_image = "- Linking a person shown in the image when the entity mention is not that person"
        pitfall_assume = "- Assuming the image directly shows the entity mention without verification"
    else:
        image_section = ""
        task_second = "2. **SECOND:** Analyze the context for clues about the entity mention"
        task_third = f'3. **THIRD:** Translate/transliterate "{entity_name}" to English for searching'
        if max_searches == 1:
            task_fourth = "4. **FOURTH:** Search English Wikipedia once with your best query"
        else:
            task_fourth = "4. **FOURTH:** Search English Wikipedia multiple times with different strategies"
        task_fifth = "5. **FIFTH:** Verify your final answer is for the entity mention, not a different entity from the context"
        pitfall_image = "- Linking a prominent person mentioned in the context when the entity mention is not that person"
        pitfall_assume = ""

    pitfalls = "\n".join(filter(None, [
        pitfall_image,
        "- Linking the grammatical subject of the context when the entity mention is actually an object or modifier",
        "- Getting distracted by famous/prominent entities when the entity mention is something more mundane",
        pitfall_assume,
    ]))

    user_prompt = f"""\
## ENTITY LINKING TASK

### ENTITY MENTION TO LINK (in {lang}):
>>> "{entity_name}" <<<

This is the ONLY entity you need to find a Wikipedia page for. Do not link any other entity.

### CONTEXT (in {lang}):
"{article_title}"

This context is provided ONLY to help you understand and disambiguate the entity mention above. The other entities mentioned in this context are NOT your target.
{image_section}
---

## YOUR TASK:

1. **FIRST:** State explicitly what entity mention you are linking and what type of entity it is
{task_second}
{task_third}
{task_fourth}
{task_fifth}

## COMMON PITFALLS TO AVOID:
{pitfalls}

Begin your investigation. Stay focused on "{entity_name}" throughout."""

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Reasoning prompt (No-RAG mode — no search instructions)
# ---------------------------------------------------------------------------

def create_reasoning_prompt(entity_name: str, article_title: str,
                            language: str, use_image: bool) -> Tuple[str, str]:
    """System + user prompt for single-shot reasoning (no RAG)."""
    lang = language.capitalize()

    if use_image:
        input_list = f"""\
1. An **ENTITY MENTION** in {lang} — THIS IS WHAT YOU MUST LINK. Nothing else.
2. A **CONTEXT** in {lang} — This provides surrounding information to help disambiguate.
3. An **IMAGE** — This MAY or MAY NOT be relevant. Treat it with skepticism."""
        mistake_block = _MISTAKE_EXAMPLE
        image_step = f"""\
### Step 3: CRITICALLY EVALUATE THE IMAGE
The image may be:
- DIRECTLY relevant (shows the entity mention itself)
- INDIRECTLY relevant (shows related context)
- MISLEADING (shows a different entity from the context)
- USELESS (unrelated or generic)

Ask yourself: "Does this image show the ENTITY MENTION, or does it show something ELSE from the context?"

If the image shows a person but your entity mention is not that person, the image is probably not helpful for your task.

"""
        translate_step_n = 4
        identify_step_n = 5
        verify_step_n = 6
    else:
        input_list = f"""\
1. An **ENTITY MENTION** in {lang} — THIS IS WHAT YOU MUST LINK. Nothing else.
2. A **CONTEXT** in {lang} — This provides surrounding information to help disambiguate."""
        mistake_block = _MISTAKE_EXAMPLE_NO_IMAGE
        image_step = ""
        translate_step_n = 3
        identify_step_n = 4
        verify_step_n = 5

    system_prompt = f"""\
You are an expert cross-lingual entity linking specialist. Your task is to identify the correct ENGLISH Wikipedia page title for a SPECIFIC entity mention using your internal knowledge.

## CRITICAL: UNDERSTANDING YOUR TASK

You will receive:
{input_list}

## THE #1 MISTAKE TO AVOID

The entity mention is often NOT the main subject of the context. DO NOT link the wrong entity.

**WRONG approach:** "The context mentions [famous person]{', and the image shows [famous person],' if use_image else ''} so I should link [famous person]."

**CORRECT approach:** "The entity mention is '[specific term]'. Even though [famous person] appears in the context{' and image' if use_image else ''}, I must find the Wikipedia page for '[specific term]' specifically."

{mistake_block}

## YOUR METHODOLOGY

### Step 1: LOCK IN THE ENTITY MENTION
Before doing ANYTHING else, answer these questions:
- What EXACT phrase is the entity mention?
- What TYPE of entity is this? (person / place / organization / event / object / concept / product / other)
- Is this entity the SUBJECT of the context, or something else mentioned in it?

Write this down explicitly. Return to it frequently.

### Step 2: ANALYZE CONTEXT FOR DISAMBIGUATION
The context tells you ABOUT the entity mention:
- What role does the entity mention play in the context?
- What clues help identify WHICH specific entity this refers to?
- Are there related entities mentioned that might help narrow down the identification?

DO NOT get distracted by other entities in the context. They are there to help you identify the entity mention, not to replace it.

{image_step}### Step {translate_step_n}: TRANSLATE/TRANSLITERATE TO ENGLISH
Since you need to identify the ENGLISH Wikipedia page:
- Transliterate names phonetically to English
- Use common English equivalents for places
- Translate concepts and common nouns
- For products/brands, use the English market name

### Step {identify_step_n}: IDENTIFY THE ENGLISH WIKIPEDIA PAGE
Using your internal knowledge:
- What is the most likely English Wikipedia page title for this entity?
- Consider common naming conventions on Wikipedia (e.g., disambiguation suffixes like "(film)", "(city)", etc.)
- Think about alternative names or spellings the entity might have

### Step {verify_step_n}: VERIFY YOUR ANSWER
Before concluding, perform this sanity check:
- [ ] Does my answer match the TYPE of entity in the mention? (If mention is a place, answer should be a place)
- [ ] Am I linking the entity MENTION, not some other entity from the context?
- [ ] If the context says "[Person] visited [Place]" and the entity mention is [Place], am I linking [Place] and NOT [Person]?

## OUTPUT FORMAT

After your analysis, state your conclusion clearly:
"The entity mention '[ENTITY]' refers to: [ENGLISH WIKIPEDIA PAGE TITLE] because [verbose reasoning that confirms you're linking the correct entity]"

Begin your analysis now. Remember: LINK THE ENTITY MENTION, NOT OTHER ENTITIES IN THE CONTEXT."""

    # --- user prompt ---
    if use_image:
        image_section = """\

### IMAGE:
An image is attached. Evaluate whether it shows the ENTITY MENTION specifically, or whether it shows something else from the context. Be skeptical -- the image may be misleading.
"""
        task_second = "2. **SECOND:** Analyze if the image is relevant to the entity mention (not just to the context)"
        task_third = f'3. **THIRD:** Translate/transliterate "{entity_name}" to English'
        task_fourth = "4. **FOURTH:** Using your knowledge, identify the correct English Wikipedia page title"
        task_fifth = "5. **FIFTH:** Verify your final answer is for the entity mention, not a different entity from the context"
    else:
        image_section = ""
        task_second = "2. **SECOND:** Analyze the context for clues about the entity mention"
        task_third = f'3. **THIRD:** Translate/transliterate "{entity_name}" to English'
        task_fourth = "4. **FOURTH:** Using your knowledge, identify the correct English Wikipedia page title"
        task_fifth = "5. **FIFTH:** Verify your final answer is for the entity mention, not a different entity from the context"

    pitfalls_parts = [
        "- Linking a person shown in the image when the entity mention is not that person" if use_image else
        "- Linking a prominent person mentioned in the context when the entity mention is not that person",
        "- Linking the grammatical subject of the context when the entity mention is actually an object or modifier",
        "- Getting distracted by famous/prominent entities when the entity mention is something more mundane",
    ]
    if use_image:
        pitfalls_parts.append("- Assuming the image directly shows the entity mention without verification")
    pitfalls = "\n".join(pitfalls_parts)

    user_prompt = f"""\
## ENTITY LINKING TASK

### ENTITY MENTION TO LINK (in {lang}):
>>> "{entity_name}" <<<

This is the ONLY entity you need to find a Wikipedia page for. Do not link any other entity.

### CONTEXT (in {lang}):
"{article_title}"

This context is provided ONLY to help you understand and disambiguate the entity mention above. The other entities mentioned in this context are NOT your target.
{image_section}
---

## YOUR TASK:

1. **FIRST:** State explicitly what entity mention you are linking and what type of entity it is
{task_second}
{task_third}
{task_fourth}
{task_fifth}

## COMMON PITFALLS TO AVOID:
{pitfalls}

Begin your analysis. Stay focused on "{entity_name}" throughout."""

    return system_prompt, user_prompt
