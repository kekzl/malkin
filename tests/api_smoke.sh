#!/usr/bin/env bash
# API smoke test against a running malkin: tests/api_smoke.sh [base_url]
set -euo pipefail
B=${1:-http://127.0.0.1:8080}
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

check health "curl -sf $B/health | jq -e '.status == \"ok\"' >/dev/null"
M=$(curl -s "$B"/v1/models | jq -r '.data[0].id')
check models "[ -n '$M' ]"

msg='{"model":"'"$M"'","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}'
r1=$(curl -s "$B"/v1/chat/completions -H 'Content-Type: application/json' -d "$msg")
check "chat content" "echo '$r1' | jq -e '.choices[0].message.content | test(\"Paris\")' >/dev/null"
check "chat usage" "echo '$r1' | jq -e '.usage.prompt_tokens > 0 and .usage.completion_tokens > 0' >/dev/null"
r2=$(curl -s "$B"/v1/chat/completions -H 'Content-Type: application/json' -d "$msg")
check "prefix cache" "echo '$r2' | jq -e '.usage.prompt_tokens_details.cached_tokens == .usage.prompt_tokens - 1' >/dev/null"
check "greedy deterministic" "[ \"\$(echo '$r1' | jq -r .choices[0].message.content)\" = \"\$(echo '$r2' | jq -r .choices[0].message.content)\" ]"

# shellcheck disable=SC2034  # used inside check's eval string
s=$(curl -sN "$B"/v1/chat/completions -H 'Content-Type: application/json' \
  -d "${msg%\}},\"stream\":true,\"stream_options\":{\"include_usage\":true}}")
check "chat stream text" "echo \"\$s\" | grep '^data: {' | sed 's/^data: //' | jq -rj '.choices[0].delta.content // empty' | grep -q Paris"
check "chat stream done" "echo \"\$s\" | tail -2 | grep -q 'data: \[DONE\]'"
check "chat stream usage" "echo \"\$s\" | grep '^data: {' | sed 's/^data: //' | jq -e 'select(.usage) | .usage.completion_tokens > 0' >/dev/null"

c=$(curl -s "$B"/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"'"$M"'","prompt":"1, 2, 3, 4,","max_tokens":6,"temperature":0,"stop":["8"]}')
check "completion + stop" "echo '$c' | jq -e '.choices[0].text | test(\"5\") and (test(\"8\") | not)' >/dev/null"

a=$(curl -s "$B"/v1/messages -H 'Content-Type: application/json' \
  -d '{"model":"'"$M"'","max_tokens":32,"system":"Answer briefly.","messages":[{"role":"user","content":"What is the capital of Italy?"}],"temperature":0}')
check "messages" "echo '$a' | jq -e '.content[-1].text | test(\"Rome\")' >/dev/null"
# shellcheck disable=SC2034  # used inside check's eval string
as=$(curl -sN "$B"/v1/messages -H 'Content-Type: application/json' \
  -d '{"model":"'"$M"'","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"What is the capital of Italy?"}],"temperature":0}')
check "messages stream" "echo \"\$as\" | grep '^data: ' | sed 's/^data: //' | jq -rj 'select(.type==\"content_block_delta\") | .delta.text // empty' | grep -q Rome"
check "messages stream stop" "echo \"\$as\" | grep -q 'event: message_stop'"

e=$(curl -s -o /dev/null -w '%{http_code}' "$B"/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":"x"}')
check "bad request 400" "[ $e = 400 ]"
exit $fail
