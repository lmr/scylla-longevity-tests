# $1 - file to archive
# $2 - size splitter in Bytes
# $3 - archive name
wc -lc $1 | awk '{print $1,$2}' |  while read lines filesize
do
  parts=$(( ( filesize + $2 - 1 ) / $2 ))
  lines_part=$(( ( lines + parts - 1 ) / parts ))
  echo "file will be split into $parts parts, per $lines_part lines"
  for (( i=1; i<=lines; i+=lines_part))
  do
    start_time=$(sed -n "${i}p" $1 | awk '{print $2,$3}' | sed -e 's/t://g;s/ /__/g;s/-/_/g;s/:/_/g;s/,/_/g;')
    tail -n "+$i" "$1" | head -n "$lines_part" | zstd -o "${start_time}.$3.zst"
  done
done
