import streamlit as st
from audio_recorder_streamlit import audio_recorder
import whisper
from io import BytesIO
import soundfile as sf

audio_bytes = audio_recorder()

if audio_bytes:
    data, samplerate = sf.read(BytesIO(audio_bytes))
    sf.write("audio.wav",data,samplerate)
    st.audio(audio_bytes, format = "audio/wav")

    model = whisper.load_model("tiny")
    #result = model.transcribe(((data[:,0]+data[:,1])/2.0).astype(np.float32))
    result = model.transcribe("audio.wav")
    st.write(result)