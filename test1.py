from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import time
import creds

start_time = time.time()

pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=creds.api_key)

count = 0
time_ranges = []
audio = pipeline('audio.wav')

# делаем разметку файла
for turn, _, speaker in audio.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    time_ranges.append((round(turn.start) * 1000, round(turn.end) * 1000, speaker))
   


# режем вав

wav_for_slice = AudioSegment.from_wav('audio.wav') 

if not os.path.exists('output'):
    os.makedirs('output')

for i in time_ranges:

    count += 1
    start = i[0]
    end = i[1]
    name = i[2]
    extract = wav_for_slice[start:end]
    extract.export(f'output/Part_{count}_{name}.wav', format='wav')


print('Splitting done', len(time_ranges) ,'files created')
print('Given .wav file was', len(wav_for_slice) // 1000, 'seconds long')
print("Execution time: --- %s seconds ---" % (time.time() - start_time))

