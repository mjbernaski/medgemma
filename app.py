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


def chat(user_text, image, history, system_prompt_name):
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
            max_tokens=2048,
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
        yield history, gr.update(value=None)


# Build the Gradio UI
with gr.Blocks(
    title="MedGemma 27B - Medical AI Assistant",
    theme=gr.themes.Soft(),
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
                type="messages",
            )
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="Ask a medical question or describe the uploaded image...",
                    label="Message",
                    scale=4,
                    lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear Conversation")

    # Wire up events
    send_btn.click(
        fn=chat,
        inputs=[text_input, image_input, chatbot, system_prompt],
        outputs=[chatbot, image_input],
    ).then(fn=lambda: "", outputs=text_input)

    text_input.submit(
        fn=chat,
        inputs=[text_input, image_input, chatbot, system_prompt],
        outputs=[chatbot, image_input],
    ).then(fn=lambda: "", outputs=text_input)

    clear_btn.click(fn=lambda: ([], None), outputs=[chatbot, image_input])

if __name__ == "__main__":
    demo.launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT)
