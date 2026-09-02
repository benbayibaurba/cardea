"""Render one demo panel's running log with Chain-of-Box overlays."""

import base64
import re
from io import BytesIO

from PIL import ImageDraw

from cardea.boxes import BBOX_PATTERN, RATIO_BASE

_MAX_EDGE = 1344
_BOX = re.compile(BBOX_PATTERN)


def snapshot_markdown(snapshot):
    """Format one structured task snapshot for display."""
    sections = []
    if snapshot.reasoning is not None:
        sections.append(f"**Thinking Process:**\n\n{snapshot.reasoning}")
    if snapshot.answer_text or snapshot.result is not None:
        sections.append(f"**Final Answer:**\n\n{snapshot.answer_text}")
    text = "\n\n---\n\n".join(sections)
    if snapshot.notice:
        text += f"\n\n_{snapshot.notice}_"
    return text


def _b64(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > _MAX_EDGE:
        s = _MAX_EDGE / max(image.size)
        image = image.resize((int(image.width * s), int(image.height * s)))
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _draw_boxes(text, images):
    def one(m):
        g = m.groupdict()
        idx = int(g["image_n"])
        rx1, ry1, rx2, ry2 = int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"])
        # skip a box the model got wrong: bad image index, coords off the 0-1000
        # scale, or an inverted rectangle. Leave the raw text in place.
        if not (0 <= idx < len(images)
                and all(0 <= v <= RATIO_BASE for v in (rx1, ry1, rx2, ry2))
                and rx1 <= rx2 and ry1 <= ry2):
            return m.group(0)
        canvas = images[idx].copy().convert("RGB")
        w, h = canvas.size
        x1, y1 = rx1 / RATIO_BASE * w, ry1 / RATIO_BASE * h
        x2, y2 = rx2 / RATIO_BASE * w, ry2 / RATIO_BASE * h
        # box only; the label is often noise after RLVR, so we don't draw it
        ImageDraw.Draw(canvas).rectangle((x1, y1, x2, y2), outline="red", width=3)
        return f'<br><img src="data:image/jpeg;base64,{_b64(canvas)}" style="max-width: 500px; margin: 10px 0;" /><br>'

    return _BOX.sub(one, text)


class ChatLog:
    """Holds the (role, text, images) entries for one panel and renders them."""

    def __init__(self):
        self.entries = []

    def reset(self):
        self.entries = []

    def add(self, role, text, images=None):
        imgs = [im.convert("RGB") if im.mode in ("RGBA", "P") else im for im in (images or [])]
        self.entries.append((role, text, imgs))

    def update_last(self, text):
        role, _, imgs = self.entries[-1]
        self.entries[-1] = (role, text, imgs)

    def render(self, boxes=True):
        """Render to gr.Chatbot 'messages' format: a list of {role, content}."""
        out = []
        for i, (role, msg, imgs) in enumerate(self.entries):
            text = msg
            if role == "user" or i % 2 == 0:
                for im in imgs:
                    tag = (f'<img src="data:image/jpeg;base64,{_b64(im)}" '
                           'style="max-width: 200px; display: inline-block; margin: 5px; border: 1px solid #ddd;" />')
                    text = text.replace("<image>", f"\n<br>{tag}", 1) if "<image>" in text else text + f"\n<br>{tag}"
                out.append({"role": "user", "content": text})
            else:
                prev = self.entries[i - 1][2] if i > 0 else []
                if boxes and text and prev and _BOX.search(text):
                    text = _draw_boxes(text, prev)
                out.append({"role": "assistant", "content": text})
        return out
