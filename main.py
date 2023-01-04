import streamlit as st
from audio_recorder_streamlit import audio_recorder
import whisper
from io import BytesIO
import soundfile as sf

st.title("Whisper App :loudspeaker:")
st.write(
    """
    This app is a mix of [OpenAI's Whisper Model](https://github.com/openai/whisper), and a custom
    Streamlit component that [records audio](https://github.com/Joooohan/audio-recorder-streamlit) Online.

    Currently the app shows great results on Spanish and English Languages.

    We used the Tiny version of the Whisper Model.
    """)
audio_bytes = audio_recorder(
    pause_threshold= 2
)
try:
    data, samplerate = sf.read(BytesIO(audio_bytes))
    sf.write("audio.wav",data,samplerate)
    st.audio(audio_bytes, format = "audio/wav")
    if data.shape[0]>=10000:
        model = whisper.load_model("tiny")
        result = model.transcribe("audio.wav")
        st.info(result["text"])

    else: st.warning("Recorded Audio is too short, try again :relieved:")#wink
except:
    st.warning("Recorded Audio is too short, try again :relieved:")#wink
