"""Ứng dụng Gradio nhận dạng chữ số viết tay bằng mô hình MNIST CNN."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

from model import MNISTCNN, MNIST_MEAN, MNIST_STD


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "models" / "mnist_cnn.pth"
CANVAS_SIZE = 280
MODEL: MNISTCNN | None = None
MODEL_LOAD_ERROR: str | None = None

APP_CSS = """
:root {
    --app-ink: #172033;
    --app-muted: #64748b;
    --app-primary: #4f46e5;
    --app-secondary: #7c3aed;
}

.gradio-container {
    max-width: 1080px !important;
    min-height: 100vh;
    margin: 0 auto !important;
    padding: 34px 22px 48px !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif !important;
}

body {
    background:
        radial-gradient(circle at 12% 8%, rgba(99, 102, 241, 0.16), transparent 32%),
        radial-gradient(circle at 88% 92%, rgba(168, 85, 247, 0.14), transparent 30%),
        #f8fafc;
}

#hero {
    margin-bottom: 24px;
    text-align: center;
}

#hero .hero-badge {
    display: inline-flex;
    align-items: center;
    padding: 7px 13px;
    border: 1px solid rgba(79, 70, 229, 0.2);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.72);
    color: var(--app-primary);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.12em;
}

#hero h1 {
    margin: 14px 0 8px;
    color: var(--app-ink);
    font-size: clamp(32px, 5vw, 52px);
    line-height: 1.08;
    letter-spacing: -0.04em;
}

#hero h1 span {
    background: linear-gradient(110deg, var(--app-primary), var(--app-secondary));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

#hero p {
    max-width: 620px;
    margin: 0 auto;
    color: var(--app-muted);
    font-size: 16px;
    line-height: 1.7;
}

.glass-card {
    padding: 22px !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 24px !important;
    background: rgba(255, 255, 255, 0.9) !important;
    box-shadow: 0 20px 55px rgba(51, 65, 85, 0.1) !important;
    backdrop-filter: blur(14px);
}

.section-heading h3 {
    margin: 0 0 4px !important;
    color: var(--app-ink);
    font-size: 18px !important;
}

.section-heading p {
    margin: 0 0 14px !important;
    color: var(--app-muted);
    font-size: 13px;
}

#drawing-board {
    overflow: hidden;
    border: 2px solid #e2e8f0 !important;
    border-radius: 18px !important;
    background: #050505 !important;
}

#predict-button,
#clear-button {
    min-height: 46px;
    border-radius: 13px !important;
    font-weight: 750 !important;
}

#predict-button {
    border: 0 !important;
    background: linear-gradient(110deg, var(--app-primary), var(--app-secondary)) !important;
    box-shadow: 0 10px 24px rgba(79, 70, 229, 0.24);
}

#prediction-box textarea,
#prediction-box input {
    min-height: 86px !important;
    color: var(--app-primary) !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    text-align: center;
}

#probability-box {
    margin-top: 4px;
}

.privacy-note {
    margin-top: 22px;
    color: var(--app-muted);
    text-align: center;
    font-size: 13px;
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 22px 12px 32px !important;
    }

    .glass-card {
        padding: 16px !important;
    }
}
"""

APP_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
)


def load_trained_model(checkpoint_path: Path = CHECKPOINT_PATH) -> MNISTCNN:
    """Nạp checkpoint trên CPU và đưa mô hình về chế độ suy luận."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy weights tại '{checkpoint_path}'. "
            "Hãy chạy notebook mnist_cnn_training.ipynb trước."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint không đúng định dạng: thiếu 'model_state_dict'.")

    model = MNISTCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def initialize_model() -> None:
    """Nạp mô hình nhưng vẫn cho phép giao diện khởi động nếu weights bị thiếu/hỏng."""

    global MODEL, MODEL_LOAD_ERROR
    try:
        MODEL = load_trained_model()
        MODEL_LOAD_ERROR = None
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        MODEL = None
        MODEL_LOAD_ERROR = str(error)


def blank_canvas() -> dict[str, Any]:
    """Tạo nền đen sạch theo định dạng EditorValue của Gradio."""

    background = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), color="black")
    return {"background": background, "layers": [], "composite": background.copy()}


def _extract_image(editor_value: Any) -> Image.Image:
    if editor_value is None:
        raise ValueError("Bảng vẽ đang trống. Hãy vẽ một chữ số từ 0 đến 9.")

    value = editor_value.get("composite") if isinstance(editor_value, dict) else editor_value
    if value is None:
        raise ValueError("Không đọc được nét vẽ. Hãy thử vẽ lại chữ số.")

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, np.ndarray):
        return Image.fromarray(value.astype(np.uint8))
    if isinstance(value, (str, Path)):
        return Image.open(value)
    raise ValueError("Định dạng ảnh từ bảng vẽ không được hỗ trợ.")


def _to_grayscale_array(image: Image.Image) -> np.ndarray:
    """Chuyển ảnh về grayscale và đặt vùng trong suốt lên nền đen."""

    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    rgb_on_black = rgba[..., :3] * alpha
    grayscale = (
        0.299 * rgb_on_black[..., 0]
        + 0.587 * rgb_on_black[..., 1]
        + 0.114 * rgb_on_black[..., 2]
    )

    # Hỗ trợ cả kiểu vẽ đen trên nền trắng lẫn trắng trên nền đen.
    if float(grayscale.mean()) > 127.0:
        grayscale = 255.0 - grayscale
    return np.clip(grayscale, 0, 255).astype(np.uint8)


def preprocess_drawing(editor_value: Any) -> tuple[torch.Tensor, np.ndarray]:
    """Cắt, resize và căn giữa nét vẽ theo phong cách dữ liệu MNIST."""

    grayscale = _to_grayscale_array(_extract_image(editor_value))
    mask = grayscale > 20
    if not np.any(mask):
        raise ValueError("Chưa phát hiện nét vẽ. Hãy vẽ chữ số rõ hơn rồi thử lại.")

    rows, columns = np.where(mask)
    cropped = grayscale[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]

    height, width = cropped.shape
    scale = 20.0 / max(height, width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = np.asarray(
        Image.fromarray(cropped).resize((resized_width, resized_height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )

    total_intensity = float(resized.sum())
    if total_intensity <= 0:
        raise ValueError("Nét vẽ quá mờ để mô hình nhận dạng.")
    y_grid, x_grid = np.indices(resized.shape)
    center_x = float((x_grid * resized).sum() / total_intensity)
    center_y = float((y_grid * resized).sum() / total_intensity)

    left = int(round(13.5 - center_x))
    top = int(round(13.5 - center_y))
    left = min(max(left, 0), 28 - resized_width)
    top = min(max(top, 0), 28 - resized_height)

    processed = np.zeros((28, 28), dtype=np.uint8)
    processed[top : top + resized_height, left : left + resized_width] = resized

    normalized = processed.astype(np.float32) / 255.0
    normalized = (normalized - MNIST_MEAN[0]) / MNIST_STD[0]
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor, processed


def predict_digit(editor_value: Any) -> tuple[str, dict[str, float]]:
    """Dự đoán chữ số và trả về kết quả cùng xác suất của 10 lớp."""

    if MODEL is None:
        detail = MODEL_LOAD_ERROR or "Mô hình chưa được khởi tạo."
        raise gr.Error(f"Không thể sử dụng mô hình: {detail}")

    try:
        tensor, _ = preprocess_drawing(editor_value)
    except ValueError as error:
        raise gr.Error(str(error)) from error

    with torch.inference_mode():
        probabilities_tensor = torch.softmax(MODEL(tensor), dim=1)[0]

    predicted_digit = int(probabilities_tensor.argmax().item())
    confidence = float(probabilities_tensor[predicted_digit].item())
    probabilities = {
        str(digit): float(probabilities_tensor[digit].item()) for digit in range(10)
    }
    result = f"{predicted_digit} — độ tin cậy {confidence:.2%}"
    return result, probabilities


def reset_interface() -> tuple[dict[str, Any], str, None]:
    """Reset ngay bảng vẽ và kết quả mà không đi qua hàng đợi Gradio."""

    return blank_canvas(), "", None


def create_demo() -> gr.Blocks:
    """Tạo giao diện Gradio cho ứng dụng localhost."""

    with gr.Blocks(title="Nhận dạng chữ số MNIST") as demo:
        gr.HTML(
            """
            <div class="hero-badge">CNN · PYTORCH · MNIST</div>
            <h1>Nhận dạng <span>chữ số viết tay</span></h1>
            <p>Vẽ một chữ số từ 0 đến 9. Mô hình CNN sẽ phân tích nét vẽ
            và hiển thị dự đoán cùng mức độ tin cậy.</p>
            """,
            elem_id="hero",
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["glass-card"]):
                gr.Markdown(
                    "### 1. Vẽ chữ số\nDùng chuột hoặc trackpad để vẽ nét trắng trên bảng đen.",
                    elem_classes=["section-heading"],
                )
                drawing = gr.ImageEditor(
                    value=blank_canvas,
                    type="pil",
                    image_mode="RGB",
                    sources=(),
                    brush=gr.Brush(
                        colors=["#FFFFFF"],
                        default_color="#FFFFFF",
                        color_mode="fixed",
                        default_size=18,
                    ),
                    eraser=gr.Eraser(default_size=28),
                    layers=False,
                    transforms=(),
                    canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
                    fixed_canvas=True,
                    height=340,
                    label="Bảng vẽ",
                    show_label=False,
                    buttons=[],
                    elem_id="drawing-board",
                )
                with gr.Row():
                    predict_button = gr.Button(
                        "Nhận dạng chữ số", variant="primary", elem_id="predict-button"
                    )
                    clear_button = gr.Button(
                        "Xóa bảng vẽ", variant="secondary", elem_id="clear-button"
                    )
            with gr.Column(scale=1, elem_classes=["glass-card"]):
                gr.Markdown(
                    "### 2. Xem kết quả\nCác xác suất bên dưới cho biết mức tin cậy của từng chữ số.",
                    elem_classes=["section-heading"],
                )
                prediction = gr.Textbox(
                    label="Kết quả dự đoán",
                    value="Hãy vẽ một chữ số",
                    interactive=False,
                    elem_id="prediction-box",
                )
                class_probabilities = gr.Label(
                    label="Xác suất theo từng chữ số",
                    num_top_classes=10,
                    elem_id="probability-box",
                )

        predict_button.click(
            fn=predict_digit,
            inputs=drawing,
            outputs=[prediction, class_probabilities],
            queue=False,
            show_progress="minimal",
        )
        clear_button.click(
            fn=reset_interface,
            inputs=None,
            outputs=[drawing, prediction, class_probabilities],
            queue=False,
            show_progress="hidden",
        )

        gr.HTML(
            "🔒 Ứng dụng chạy cục bộ — dữ liệu nét vẽ không được gửi lên dịch vụ ngoài.",
            elem_classes=["privacy-note"],
        )
    return demo


initialize_model()
demo = create_demo()


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=APP_THEME,
        css=APP_CSS,
    )
