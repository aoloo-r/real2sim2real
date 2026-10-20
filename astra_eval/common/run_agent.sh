#!/usr/bin/env bash
# Prepare one agent run on one packet and print the command to start the agent. Usage: run_agent.sh codex|claude PACKET N
# The agent is started by hand in the printed directory, so every permission prompt and every answer to a
# question is a visible, countable intervention. Blind packets get only the packet; p3_tamp gets a repo worktree.
set -e
AGENT=$1; PACKET=$2; N=${3:-1}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=$ROOT/astra_eval/runs/$PACKET/${AGENT}_$N
PK=$ROOT/astra_eval/$PACKET
[ -d "$PK" ] || { echo "no packet $PK"; exit 1; }
[ -e "$RUN" ] && { echo "$RUN exists; pick another N"; exit 1; }
mkdir -p "$RUN/out"
WT=$RUN/work
if [ "$PACKET" = "p3_tamp" ]; then
  git -C "$ROOT" worktree add -q "$WT" HEAD                          # repo access: the agent drives our executor
  cp -r "$PK" "$WT/packet"; rm -rf "$WT/astra_eval"                  # but not the other packets or their scores
else
  mkdir -p "$WT"; cp -r "$PK"/. "$WT"/                               # blind: only the packet
fi
MODEL=${MODEL:-$([ "$AGENT" = codex ] && echo gpt-6-astra || echo claude-fable-5-1)}
PY=${NEWTON_PY:-$HOME/miniforge3/envs/newton-1.7/bin/python}
$PY "$ROOT/astra_eval/common/runlog.py" start "$RUN" --agent "$AGENT" --model "$MODEL" --packet "$PACKET"
{ cat "$PK/PROMPT.md"; echo; echo "Write every output file into: $RUN/out/"; } > "$RUN/prompt.txt"
cat <<EOF
[run] prepared $RUN
  cd $WT
  $( [ "$AGENT" = codex ] && echo "codex --model $MODEL \"\$(cat $RUN/prompt.txt)\"" || echo "claude --model $MODEL \"\$(cat $RUN/prompt.txt)\"" )
  (keep the transcript: run it under  script -q $RUN/agent.log  or copy the session log there)
when it ends:
  $PY $ROOT/astra_eval/common/runlog.py end $RUN --interventions N [--tokens T --cost USD --note "..."]
EOF
