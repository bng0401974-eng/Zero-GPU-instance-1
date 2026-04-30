import gradio as gr
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile
import os
import spaces # Задолжително за Zero-GPU

# 1. Model Setup - ПРЕФРЛАЊЕ НА LARGE[cite: 1]
model_id = "facebook/musicgen-large"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicgenForConditionalGeneration.from_pretrained(model_id)

# Зголемено времетраење на GPU сесијата за да издржи 3 минути генерирање[cite: 1]
@spaces.GPU(duration=250)
def generate_music(prompt, duration, mode):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        if mode == "Loop Ready (Seamless)":
            final_prompt = f"{prompt}, seamless loop, perfectly repetitive, continuous audio cycle"
        else:
            final_prompt = prompt

        # Пресметка: 50 токени по секунда. За 180 сек = 9000 токени[cite: 1]
        max_new_tokens = int(duration * 50)
        inputs = processor(text=[final_prompt], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                guidance_scale=3.5 # Малку зголемена скала за подобра дефиниција на Large моделот
            )

        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        file_path = "generated_audio.wav"
        scipy.io.wavfile.write(file_path, sampling_rate, audio_data)

        return (sampling_rate, audio_data), file_path
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# 2. Респонзивен CSS[cite: 1]
css = """
.gradio-container { 
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    width: 100% !important;
}
@media (min-width: 1024px) {
    .main-column {
        width: 66.66% !important; /* 4/6 центар на 1600px[cite: 1] */
        margin: 0 auto !important;
    }
}
#title-text { text-align: center !important; margin-bottom: 30px; }
.input-group, .output-group { 
    border-radius: 15px !important; 
    border: 1px solid #e5e7eb !important; 
    padding: 25px !important; 
    background-color: #fcfcfc !important;
    margin-bottom: 20px !important;
    width: 100% !important;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}
.gen-button { 
    background-color: #6366f1 !important; 
    color: white !important; 
    font-weight: bold !important; 
    width: 100% !important; 
    height: 75px !important; 
    font-size: 22px !important;
    border-radius: 12px !important;
}
#download-btn { background-color: #c305f7 !important; color: white !important; font-weight: bold !important; width: 100% !important; }
"""

# 3. Layout[cite: 1]
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎵 LATIVM AI Music Studio (Large Model)", elem_id="title-text")

    with gr.Column(elem_classes="main-column"):
        with gr.Group(elem_classes="input-group"):
            mode_radio = gr.Radio(
                choices=["Normal", "Loop Ready (Seamless)"],
                value="Loop Ready (Seamless)",
                label="Output Mode"
            )

            t_in = gr.Textbox(
                label="Sound Description",
                value="cinematic orchestral hybrid, deep sub bass, 90 bpm",
                lines=8,
                elem_id="description-box"
            )

            # ПОСТАВЕНО НА 180 СЕКУНДИ (3 МИНУТИ)[cite: 1]
            d_in = gr.Slider(minimum=1, maximum=180, value=10, step=1, label="Duration (seconds)")

        btn = gr.Button("🚀 GENERATE MASTERPIECE", variant="primary", elem_classes="gen-button")

        with gr.Group(elem_classes="output-group"):
            audio_out = gr.Audio(label="Audio Monitor", type="numpy")
            download_btn = gr.DownloadButton("📥 Studio Export (WAV)", visible=False, elem_id="download-btn")

    btn.click(
        fn=generate_music,
        inputs=[t_in, d_in, mode_radio],
        outputs=[audio_out, download_btn],
        show_progress="full"
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[download_btn]
    )

if __name__ == "__main__":
    demo.launch()