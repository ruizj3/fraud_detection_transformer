#!/usr/bin/env bash

set -euo pipefail

output_dir="${1:-artifacts/service-gifs}"
mkdir -p "$output_dir"

capture_text_gif() {
  local name="$1"
  local title="$2"
  local source="$3"
  local frames="$output_dir/.${name}-frames"
  local clean_source="$frames/source.txt"
  rm -rf "$frames"
  mkdir -p "$frames"
  LC_ALL=C sed 's/[^ -~]//g' "$source" > "$clean_source"

  awk -v title="$title" -v output_frames="$frames" '
    { lines[NR] = $0 }
    END {
      total = NR
      if (total == 0) lines[++total] = "No output captured"
      for (frame = 1; frame <= 8; frame++) {
        if (frame == 1) {
          start = total - 10
          if (start < 1) start = 1
          finish = total
        } else {
          start = int((frame - 1) * total / 8) + 1
          finish = int(frame * total / 8)
        }
        if (finish < start) finish = start
        file = sprintf("%s/%02d.srt", output_frames, frame)
        printf "1\n00:00:00,000 --> 00:00:01,000\nfraud_detection_transformer\n%s\n\n", title > file
        for (i = start; i <= finish && i <= total; i++) print lines[i] >> file
        close(file)
      }
    }
  ' "$clean_source"

  for subtitle in "$frames"/*.srt; do
    frame="${subtitle%.srt}.png"
    ffmpeg -hide_banner -loglevel error -f lavfi -i "color=c=0x111827:s=1280x720" \
      -vf "subtitles='$subtitle':force_style='FontName=Menlo,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00111827,BorderStyle=1,Outline=2,Alignment=7,MarginL=64,MarginV=48'" \
      -frames:v 1 -y "$frame"
  done
  ffmpeg -hide_banner -loglevel error -framerate 2 -i "$frames/%02d.png" \
    -vf "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse" \
    -loop 0 -y "$output_dir/$name.gif"
  rm -rf "$frames"
}

docker compose logs --no-color --tail=30 ingester > "$output_dir/ingester-output.log"
docker compose logs --no-color --tail=30 trainer > "$output_dir/trainer-output.log"
docker compose logs --no-color --tail=30 server > "$output_dir/server-output.log"

docker compose exec -T postgres psql -U fraud -d fraud_training -P pager=off -c \
  "\\echo 'fraud_events (latest rows)' \
SELECT event_id, event_timestamp, amount, merchant_category, is_fraud FROM fraud_events ORDER BY event_timestamp DESC LIMIT 8; \
\\echo 'fraud_predictions (latest rows)' \
SELECT event_id, fraud_probability, predicted_is_fraud, scored_at FROM fraud_predictions ORDER BY scored_at DESC LIMIT 8;" \
  > "$output_dir/postgres-output.log"

{
  printf 'GET / HTTP/1.1\n'
  curl -fsS -D - -o /dev/null http://localhost:8082/
} > "$output_dir/adminer-output.log"

capture_text_gif "ingester-output" "Ingester service" "$output_dir/ingester-output.log"
capture_text_gif "trainer-output" "Trainer service" "$output_dir/trainer-output.log"
capture_text_gif "server-output" "Scoring server" "$output_dir/server-output.log"
capture_text_gif "postgres-output" "Postgres query result" "$output_dir/postgres-output.log"
capture_text_gif "adminer-http-output" "Adminer HTTP response" "$output_dir/adminer-output.log"

rm -f "$output_dir"/*.log
printf 'Created service output GIFs in %s\n' "$output_dir"