import gradio as gr
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import spaces  # Ова е клучно за Zero-GPU!

# Вчитување на моделот (Large верзија)
model_id = "facebook/musicgen-large"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicgenForConditionalGeneration.from_pretrained(model_id)


# Функција која ќе ја користи графичката картичка
@spaces.GPU(duration=60)  # Му даваме 60 секунди GPU време по генерирање
def generate_music(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    audio_values = model.generate(**inputs, max_new_tokens=512)  # Генерира околу 10-15 сек

    # Конвертирање во аудио формат
    sampling_rate = model.config.audio_encoder.sampling_rate
    return (sampling_rate, audio_values[0, 0].cpu().numpy())


# Gradio Интерфејс
with gr.Blocks() as demo:
    gr.Markdown("# LATIVM MusicGen Large (Zero-GPU)")
    input_text = gr.Textbox(label="Опиши ја музиката")
    output_audio = gr.Audio(label="Генерирано аудио")
    btn = gr.Button("Генерирај")

    btn.click(fn=generate_music, inputs=input_text, outputs=output_audio)

demo.launch()