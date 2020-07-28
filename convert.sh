format="wav"
for file in /home/darshanakg/speech_commands/new_describe/*; do
    echo "$(basename "$file")"
    ffmpeg -i ${file} -ac 1 -ar 16000 -acodec pcm_s16le "${file/mp3/$format}"
    rm ${file}
done