import os
import shutil
import tempfile

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from TTS.api import TTS


# Require the user to review and accept XTTS v2's terms before launching.
if os.environ.get("COQUI_TOS_AGREED") != "1":
    raise RuntimeError(
        "Review and accept the XTTS v2 model terms first. "
        "Then set COQUI_TOS_AGREED=1 in your Colab notebook before running app.py."
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected. In Colab, select Runtime > Change runtime type > T4 GPU, "
        "restart the runtime, and run again."
    )

device = "cuda"
print("GPU:", torch.cuda.get_device_name(0))

print("Loading speech recognition model...")
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=0,
)

print("Loading chat model...")
llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
llm.eval()

print("Loading voice synthesis model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print("Models loaded.")

# Temporary profile files are kept in the Colab runtime, not in the repository.
PROFILE_DIR = os.path.join(tempfile.gettempdir(), "mak_ai_profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)


def enroll_voice(audio_path, consent, previous_profile):
    if not consent:
        return None, "Confirm that you own this voice or have permission to use it.", None
    if not audio_path or not os.path.exists(audio_path):
        return None, "Record or upload a voice sample first.", None

    try:
        # Convert the input recording to mono WAV at 24 kHz for XTTS.
        audio, sample_rate = librosa.load(audio_path, sr=24000, mono=True)

        if audio.size == 0:
            return None, "The recording is empty. Please try another sample.", None

        duration = len(audio) / sample_rate
        peak = float(np.max(np.abs(audio)))

        if duration < 6:
            return None, "Use a clear sample of at least 6 seconds; 10–20 seconds is recommended.", None
        if peak < 0.01:
            return None, "The sample is very quiet. Check the microphone or choose a clearer recording.", None
        if peak > 0.99:
            return None, "The sample may be distorted. Record again with the microphone farther away.", None

        profile_path = tempfile.NamedTemporaryFile(
            dir=PROFILE_DIR,
            suffix=".wav",
            delete=False,
        ).name
        sf.write(profile_path, audio, sample_rate)

        if previous_profile and os.path.exists(previous_profile):
            try:
                os.remove(previous_profile)
            except OSError:
                pass

        status = f"Saved voice reference ({duration:.1f} seconds). Play the preview to verify it."
        return profile_path, status, profile_path

    except Exception as exc:
        return None, f"Could not process that recording: {exc}", None


def delete_voice(profile_path):
    if profile_path and os.path.exists(profile_path):
        try:
            os.remove(profile_path)
        except OSError:
            pass

    return None, "Voice reference deleted from this Colab session.", None


def transcribe_question(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        return ""
    result = asr(audio_path)
    return result.get("text", "").strip()


def generate_answer(text):
    messages = [
        {
            "role": "system",
            "content": "You are Mak AI, a helpful assistant. Answer clearly and naturally.",
        },
        {"role": "user", "content": text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

    with torch.inference_mode():
        output = llm.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return answer or "Sorry, I couldn't generate a reply."


def speak_with_profile(text, profile_path):
    if not profile_path or not os.path.exists(profile_path):
        raise gr.Error("Record or upload a voice reference and save it first.")

    audio = tts.tts(
        text=text.strip(),
        speaker_wav=profile_path,
        language="en",
        split_sentences=True,
    )
    audio = np.asarray(audio, dtype=np.float32).squeeze()

    if audio.size == 0 or not np.isfinite(audio).all():
        raise gr.Error("Speech generation failed. Try a clearer voice sample.")

    # Raise output level safely without exceeding the audio range.
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio *= min(0.85 / peak, 1.0)

    output_path = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ).name
    sf.write(output_path, audio, int(tts.synthesizer.output_sample_rate))
    return output_path


def render_avatar(speaking=False):
    mode = "speaking" if speaking else "idle"
    status = "MAK AI IS SPEAKING" if speaking else "MAK AI READY"

    return f"""
    <div class="avatar-stage {mode}">
      <div class="orbit orbit-one"></div>
      <div class="orbit orbit-two"></div>
      <div class="avatar">
        <div class="face">
          <div class="brow brow-left"></div>
          <div class="brow brow-right"></div>
          <div class="eye eye-left"></div>
          <div class="eye eye-right"></div>
          <div class="nose"></div>
          <div class="mouth"></div>
        </div>
      </div>
      <div class="avatar-caption">
        <span class="status-dot"></span>{status}
      </div>
    </div>
    """


def chat(message, question_audio, profile_path, consent):
    if not consent:
        raise gr.Error("Confirm that you own this voice or have permission to use it.")
    if not profile_path or not os.path.exists(profile_path):
        raise gr.Error("Record or upload a voice reference and save it first.")

    text = (message or "").strip()
    if not text and question_audio:
        text = transcribe_question(question_audio)
    if not text:
        raise gr.Error("Type a message or record/upload a question.")

    answer = generate_answer(text)
    reply_audio = speak_with_profile(answer, profile_path)
    return text, answer, reply_audio, render_avatar(speaking=True)


css = """
.gradio-container {
    background:
      radial-gradient(ellipse at 50% 0%, rgba(0,105,190,.18), transparent 45%),
      #040914 !important;
    color: #d7f5ff !important;
}
h1, h2, h3, label { color: #bceeff !important; }
button.primary {
    background: linear-gradient(90deg, #0879d1, #00bde8) !important;
    border: 1px solid #55dcff !important;
    box-shadow: 0 0 18px rgba(0,187,255,.35) !important;
}
.avatar-stage {
    min-height: 320px; display:flex; flex-direction:column;
    align-items:center; justify-content:center; position:relative;
    overflow:hidden; border:1px solid rgba(45,193,255,.35);
    border-radius:22px;
    background:radial-gradient(ellipse, rgba(0,139,255,.18), transparent 58%),
               linear-gradient(145deg,#071426,#030812);
    color:#bceeff; font-family:Arial,sans-serif;
}
.avatar {
    width:178px; height:178px; display:grid; place-items:center;
    border-radius:50%; border:1px solid #4bd3ff;
    background:radial-gradient(circle,#10476a,#030c19 72%);
    box-shadow:0 0 20px #00bfff,0 0 55px rgba(0,153,255,.5),
               inset 0 0 30px rgba(0,191,255,.3);
    animation:glow 2s ease-in-out infinite alternate; z-index:2;
}
.face {
    position:relative; width:112px; height:135px;
    border:2px solid #68e0ff; border-radius:48% 48% 44% 44%;
    background:linear-gradient(160deg,rgba(20,84,121,.75),rgba(4,20,39,.95));
    box-shadow:0 0 18px rgba(49,207,255,.55),inset 0 0 22px rgba(37,183,255,.15);
}
.brow { position:absolute; top:39px; width:25px; height:2px; background:#79eaff; }
.brow-left { left:18px; transform:rotate(-7deg); }
.brow-right { right:18px; transform:rotate(7deg); }
.eye {
    position:absolute; top:48px; width:24px; height:10px;
    border-radius:50%; background:#37d9ff; box-shadow:0 0 12px #00c8ff;
}
.eye-left { left:17px; } .eye-right { right:17px; }
.nose {
    position:absolute; left:54px; top:61px; width:2px; height:28px;
    background:linear-gradient(transparent,rgba(104,224,255,.7),transparent);
}
.mouth {
    position:absolute; left:37px; top:103px; width:38px; height:4px;
    border-radius:50%; background:#70e7ff; box-shadow:0 0 8px #00c8ff;
}
.speaking .mouth { animation:talk .25s ease-in-out infinite alternate; }
.orbit {
    position:absolute; width:225px; height:225px;
    border:1px solid rgba(31,186,255,.28); border-radius:50%;
}
.orbit-one { animation:pulse 2s ease-out infinite; }
.orbit-two { width:270px; height:270px; animation:pulse 2s ease-out 1s infinite; }
.avatar-caption {
    z-index:2; margin-top:20px; font-size:11px; letter-spacing:3px;
    color:#83dfff; text-shadow:0 0 10px #00aaff;
}
.status-dot {
    display:inline-block; width:7px; height:7px; margin-right:8px;
    border-radius:50%; background:#20ddff; box-shadow:0 0 10px #00c8ff;
}
@keyframes talk { from { height:4px; } to { height:15px; } }
@keyframes glow {
    from { filter:drop-shadow(0 0 8px #00bfff); }
    to { filter:drop-shadow(0 0 20px #00d9ff); }
}
@keyframes pulse {
    0% { transform:scale(.82); opacity:.7; }
    100% { transform:scale(1.2); opacity:0; }
}
"""

with gr.Blocks(
    title="Mak AI",
    theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate"),
    css=css,
) as demo:
    gr.Markdown("# ✦ MAK AI\n### Your personal voice-powered assistant")

    consent = gr.Checkbox(
        label="I own this voice or have explicit permission to use it.",
        value=False,
    )
    profile_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            avatar = gr.HTML(render_avatar())
        with gr.Column():
            gr.Markdown("#### 1 · Set up your voice")
            voice_sample = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or upload a clear voice sample (10–20 seconds recommended)",
            )
            with gr.Row():
                save_button = gr.Button("Save voice", variant="primary")
                delete_button = gr.Button("Delete voice")

            profile_status = gr.Textbox(label="Voice profile status", interactive=False)
            profile_preview = gr.Audio(label="Saved reference preview", type="filepath")

    save_button.click(
        fn=enroll_voice,
        inputs=[voice_sample, consent, profile_state],
        outputs=[profile_state, profile_status, profile_preview],
    )
    delete_button.click(
        fn=delete_voice,
        inputs=[profile_state],
        outputs=[profile_state, profile_status, profile_preview],
    )

    gr.Markdown("#### 2 · Chat with Mak AI")
    message = gr.Textbox(label="Message", placeholder="Type your question…", lines=2)
    question_audio = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Or record/upload a question",
    )
    send_button = gr.Button("✦ Ask Mak AI", variant="primary")

    transcript_output = gr.Textbox(label="You said")
    answer_output = gr.Textbox(label="Mak AI's answer", lines=4)
    spoken_output = gr.Audio(label="Mak AI speaking", autoplay=True)

    send_button.click(
        fn=chat,
        inputs=[message, question_audio, profile_state, consent],
        outputs=[transcript_output, answer_output, spoken_output, avatar],
    )

demo.queue().launch(share=True)