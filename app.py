"""MedGemma 27B Gradio Web UI - Medical AI Chat with Image Support."""

import base64
import os
from io import BytesIO
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv(Path(__file__).parent / ".env")

VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
MODEL_NAME = os.getenv("MODEL_NAME", "google/medgemma-27b-it")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")

client = OpenAI(base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1", api_key="unused")

SYSTEM_PROMPTS = {
    "General Medical Assistant": (
        "You are a knowledgeable medical AI assistant powered by MedGemma. "
        "Provide helpful, accurate medical information. Always recommend "
        "consulting a healthcare professional for diagnosis and treatment. "
        "Be thorough but clear in your explanations."
    ),
    "Radiologist": (
        "You are an expert radiology AI assistant. When presented with medical "
        "images such as X-rays, CT scans, or MRIs, provide detailed observations "
        "about findings, potential abnormalities, and relevant anatomical structures. "
        "Always note that official interpretation should be done by a licensed radiologist."
    ),
    "Dermatology": (
        "You are a dermatology AI assistant. When presented with skin images, "
        "describe visible features including lesion morphology, color, borders, "
        "and distribution. Suggest possible differential diagnoses and recommend "
        "appropriate next steps. Always advise consulting a dermatologist."
    ),
    "Pathology": (
        "You are a pathology AI assistant. When presented with histopathology "
        "or cytology images, describe tissue architecture, cellular features, "
        "and staining patterns. Provide differential diagnoses based on "
        "morphological findings. Always note that definitive diagnosis requires "
        "a board-certified pathologist."
    ),
}


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image to a base64 data URI."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def build_messages(history, user_text, image, system_prompt_name):
    """Build OpenAI-format messages from chat history, new input, and optional image."""
    system_prompt = SYSTEM_PROMPTS.get(system_prompt_name, SYSTEM_PROMPTS["General Medical Assistant"])
    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history
    for msg in history:
        if msg["role"] == "user":
            # History entries are text-only (images aren't preserved in history)
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})

    # Build the new user message
    if image is not None:
        pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        data_uri = encode_image_to_base64(pil_image)
        content = [
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
        if user_text.strip():
            content.append({"type": "text", "text": user_text})
        else:
            content.append({"type": "text", "text": "Describe this medical image in detail."})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_text})

    return messages


RESPONSE_LENGTHS = {
    "Short (~256 tokens)": 256,
    "Medium (~1024 tokens)": 1024,
    "Long (~2048 tokens)": 2048,
    "Extra Long (~4096 tokens)": 4096,
}


def chat(user_text, image, history, system_prompt_name, response_length):
    """Send a message (with optional image) and stream the response."""
    if not user_text.strip() and image is None:
        yield history, gr.update()
        return

    # Add user message to history display
    if image is not None:
        # Show text + note about image in history
        display_text = user_text.strip() if user_text.strip() else "(image uploaded)"
        display_text += "\n\n*[Image attached]*"
    else:
        display_text = user_text

    history = history + [{"role": "user", "content": display_text}]

    messages = build_messages(history[:-1], user_text, image, system_prompt_name)

    # Stream the response
    history = history + [{"role": "assistant", "content": ""}]
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            max_tokens=RESPONSE_LENGTHS.get(response_length, 2048),
            temperature=0.3,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                history[-1]["content"] += delta.content
                yield history, gr.update(value=None)
    except Exception as e:
        error_msg = f"**Error:** {e}\n\nMake sure the vLLM server is running on {VLLM_HOST}:{VLLM_PORT}."
        history[-1]["content"] = error_msg

    # Final yield to ensure history is committed after stream ends
    yield history, gr.update(value=None)


# Force-dark theme: set both light and dark variants to dark colors
# so the UI is dark regardless of browser/OS theme preference.
dark_theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    # Body / page background
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    body_text_color="*neutral_50",
    body_text_color_dark="*neutral_50",
    body_text_color_subdued="*neutral_300",
    body_text_color_subdued_dark="*neutral_300",
    # Primary fill areas
    background_fill_primary="*neutral_950",
    background_fill_primary_dark="*neutral_950",
    background_fill_secondary="*neutral_800",
    background_fill_secondary_dark="*neutral_800",
    # Chat bubble colors
    color_accent_soft="*neutral_700",
    color_accent_soft_dark="*neutral_700",
    # Blocks / panels
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_label_background_fill="*neutral_800",
    block_label_background_fill_dark="*neutral_800",
    block_label_text_color="*neutral_200",
    block_label_text_color_dark="*neutral_200",
    block_title_text_color="*neutral_100",
    block_title_text_color_dark="*neutral_100",
    block_border_color="*neutral_700",
    block_border_color_dark="*neutral_700",
    # Input fields
    input_background_fill="*neutral_800",
    input_background_fill_dark="*neutral_800",
    input_border_color="*neutral_600",
    input_border_color_dark="*neutral_600",
    input_placeholder_color="*neutral_400",
    input_placeholder_color_dark="*neutral_400",
    # Panels
    panel_background_fill="*neutral_900",
    panel_background_fill_dark="*neutral_900",
    panel_border_color="*neutral_700",
    panel_border_color_dark="*neutral_700",
    # Buttons
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_dark="*primary_600",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_secondary_background_fill="*neutral_700",
    button_secondary_background_fill_dark="*neutral_700",
    button_secondary_text_color="*neutral_100",
    button_secondary_text_color_dark="*neutral_100",
    # Borders
    border_color_primary="*neutral_600",
    border_color_primary_dark="*neutral_600",
    shadow_drop="none",
    shadow_drop_lg="none",
)

# Inject <script> in <head> so dark class is set before Svelte renders
DARK_HEAD = "<script>document.documentElement.classList.add('dark');</script>"

# Build the Gradio UI
with gr.Blocks(
    title="MedGemma 27B - Medical AI Assistant",
) as demo:
    gr.Markdown(
        "# MedGemma 27B Medical AI Assistant\n"
        "Chat with Google's MedGemma 27B model. Upload medical images (X-rays, pathology, "
        "dermatology) for analysis, or ask text-based medical questions.\n\n"
        "**Disclaimer:** This is an AI research tool. Do not use for clinical decision-making. "
        "Always consult qualified healthcare professionals."
    )

    with gr.Row():
        with gr.Column(scale=1):
            system_prompt = gr.Dropdown(
                choices=list(SYSTEM_PROMPTS.keys()),
                value="General Medical Assistant",
                label="Assistant Mode",
            )
            response_length = gr.Dropdown(
                choices=list(RESPONSE_LENGTHS.keys()),
                value="Long (~2048 tokens)",
                label="Response Length",
            )
            image_input = gr.Image(label="Upload Medical Image (optional)", type="pil")
            gr.Markdown(
                "### Example Queries\n"
                "- What are the symptoms of pneumonia?\n"
                "- Describe the findings in this chest X-ray\n"
                "- What is the differential diagnosis for a lung nodule?\n"
                "- Explain the pathophysiology of heart failure"
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=550,
            )
            text_input = gr.Textbox(
                placeholder="Ask a medical question or describe the uploaded image...",
                label="Message",
                lines=4,
                max_lines=10,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Conversation", scale=1)

    # Wire up events
    send_btn.click(
        fn=chat,
        inputs=[text_input, image_input, chatbot, system_prompt, response_length],
        outputs=[chatbot, image_input],
    ).then(fn=lambda: "", outputs=text_input)

    text_input.submit(
        fn=chat,
        inputs=[text_input, image_input, chatbot, system_prompt, response_length],
        outputs=[chatbot, image_input],
    ).then(fn=lambda: "", outputs=text_input)

    clear_btn.click(fn=lambda: ([], None), outputs=[chatbot, image_input])

if __name__ == "__main__":
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        theme=dark_theme,
        head=DARK_HEAD,
        css="""
        .gradio-container { max-width: 1400px !important; }
        footer { display: none !important; }
        """,
    )
