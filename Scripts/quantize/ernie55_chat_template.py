"""ERNIE chatv2p3 template synthesis for ErnieChatTokenizer checkpoints."""
from __future__ import annotations

import json
from pathlib import Path


def is_ernie_chat_source(source: Path) -> bool:
    if (source / "encoding_ernie.py").is_file():
        return True
    tok_cfg = source / "tokenizer_config.json"
    if tok_cfg.is_file():
        try:
            obj = json.loads(tok_cfg.read_text(encoding="utf-8"))
            if obj.get("tokenizer_class") == "ErnieChatTokenizer":
                return True
        except Exception:
            pass
    return False


def synthesize_ernie_chat_template() -> str:
    return "\n".join([
        "{%- set thinking_off = (enable_thinking is defined and (enable_thinking == false or enable_thinking == 0 or enable_thinking == 'false')) -%}",
        "{%- set effort_raw = reasoning_effort | default('high') | string | lower -%}",
        "{%- set effort = 'low' if effort_raw in ['low', 'minimal', 'min', 'light', 'fast', 'quick'] else ('medium' if effort_raw in ['medium', 'mid', 'med', 'normal', 'auto', 'default'] else 'high') -%}",
        "{{- '<|ERNIE_MASK|><|ernie:im_start|>system\\n' -}}",
        "{%- if messages[0]['role'] == 'system' -%}",
        "{{- '<ernie:system_setting>\\n' -}}",
        "{%- if messages[0]['content'] is string -%}{{- messages[0]['content'] -}}{%- elif messages[0]['content'] is iterable -%}{%- for it in messages[0]['content'] -%}{%- if it['type'] == 'text' -%}{{- it['text'] -}}{%- endif -%}{%- endfor -%}{%- endif -%}",
        "{{- '\\n</ernie:system_setting>\\n' -}}",
        "{%- endif -%}",
        "{{- '<ernie:global_setting>' -}}",
        "{%- if not thinking_off -%}{{- '\\nreasoning_effort=' + effort -}}{%- endif -%}",
        "{%- if tools is defined and tools -%}{{- '\\nparallel_tool_calls=True' -}}{%- endif -%}",
        "{{- '\\n</ernie:global_setting>' -}}",
        "{%- if tools is defined and tools -%}",
        "{{- '\\n<ernie:tool_list>\\n[' -}}{%- for t in tools -%}{{- t | tojson -}}{%- if not loop.last -%}{{- ', ' -}}{%- endif -%}{%- endfor -%}{{- ']\\n</ernie:tool_list>' -}}",
        "{%- endif -%}",
        "{{- '<|ernie:im_end|>\\n' -}}",
        "{%- for m in messages -%}",
        "{%- if m['role'] == 'user' or (m['role'] == 'system' and not loop.first) -%}",
        "{{- '<|ernie:im_start|>' + m['role'] + '\\n' -}}",
        "{%- if m['content'] is string -%}{{- m['content'] -}}{%- elif m['content'] is iterable -%}{%- for it in m['content'] -%}{%- if it['type'] == 'text' -%}{{- it['text'] -}}{%- endif -%}{%- endfor -%}{%- endif -%}",
        "{{- '<|ernie:im_end|>\\n' -}}",
        "{%- elif m['role'] == 'assistant' -%}",
        "{{- '<|ernie:im_start|>assistant\\n<ernie:think>' -}}",
        "{%- if not thinking_off and m.get('reasoning_content') -%}{{- m['reasoning_content'] -}}{%- endif -%}",
        "{{- '</ernie:think>' -}}",
        "{%- if m['content'] is string -%}{{- m['content'] -}}{%- elif m['content'] is iterable -%}{%- for it in m['content'] -%}{%- if it['type'] == 'text' -%}{{- it['text'] -}}{%- endif -%}{%- endfor -%}{%- endif -%}",
        "{%- if m.get('tool_calls') -%}",
        "{{- '<ernie:tool_call>\\n' -}}",
        "{%- for tc in m['tool_calls'] -%}{%- set fn = tc.get('function', tc) -%}{{- '<invoke name=\"' + fn['name'] + '\">' -}}{%- for k, v in (fn.get('arguments') or {}).items() | sort -%}{{- '<parameter name=\"' + k + '\">' + (v if v is string else (v | tojson)) + '</parameter>' -}}{%- endfor -%}{{- '</invoke>\\n' -}}{%- endfor -%}",
        "{{- '</ernie:tool_call>' -}}",
        "{%- endif -%}",
        "{{- '<|ernie:im_end|>\\n' -}}",
        "{%- elif m['role'] == 'tool' -%}",
        "{%- if loop.first or messages[loop.index0 - 1]['role'] != 'tool' -%}{{- '<|ernie:im_start|>tool' -}}{%- endif -%}",
        "{{- '\\n<ernie:tool_output>' -}}{%- if m['content'] is string -%}{{- m['content'] -}}{%- elif m['content'] is iterable -%}{%- for it in m['content'] -%}{%- if it['type'] == 'text' -%}{{- it['text'] -}}{%- endif -%}{%- endfor -%}{%- endif -%}{{- '</ernie:tool_output>' -}}",
        "{%- if loop.last or messages[loop.index0 + 1]['role'] != 'tool' -%}{{- '<|ernie:im_end|>\\n' -}}{%- endif -%}",
        "{%- endif -%}",
        "{%- endfor -%}",
        "{%- if add_generation_prompt -%}{{- '<|ernie:im_start|>assistant\\n' + ('<ernie:think></ernie:think>' if thinking_off else '<ernie:think>') -}}{%- endif -%}",
    ]) + "\n"
