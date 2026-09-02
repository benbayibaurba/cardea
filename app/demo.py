"""Gradio demo: upload DICOM clips, run the full pipeline, watch it stream.

Start the model first (docker compose up, or your own vLLM), then run this.
"""

import asyncio
import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # loads .env for local runs; in Docker the values come from compose

from chat_log import ChatLog, snapshot_markdown
from cardea.imaging import frame_to_square_pil, read_dcm_3d_array, subsample_frames
from cardea.pipeline import STUDY_EDGE, pick_study_frames
from cardea.streaming import stream_first_with_boxes, stream_snapshots
from cardea.tasks import complexity, dominance, keyframe, report, view

MAX_SLOTS = 15
FLUSH = 30  # push a UI update after this many new chars

STUDY_TASKS = {
    "dominance": ("Classify coronary dominance.", dominance),
    "report": ("Generate clinical report.", report),
    "complexity": ("Assess SYNTAX complexity.", complexity),
}
STUDY_TITLES = {"dominance": "Dominance", "report": "Report", "complexity": "SYNTAX Complexity"}


async def run_into(task, input_value, on_update):
    result, last = None, 0
    async for snapshot in stream_snapshots(task, input_value):
        text = snapshot_markdown(snapshot)
        if snapshot.result is not None or len(text) - last >= FLUSH:
            await on_update(text)
            last = len(text)
        result = snapshot.result or result
    if result is None:
        raise RuntimeError("task stream ended without a result")
    return result


async def run_study(task, keyframes, log, lock, cob_on, cob_k):
    """Fill log's last entry with a study-task result.

    Single stream, or, when cob_on, k parallel rollouts shown live: keep the
    leading rollout while it reasons; once its answer starts without a box,
    switch to the next; keep the first whose reasoning gets a box.
    """
    async def set_last(t):
        async with lock:
            log.update_last(t)

    snapshots = (
        stream_first_with_boxes(task, keyframes, attempts=cob_k)
        if cob_on
        else stream_snapshots(task, keyframes)
    )
    async for snapshot in snapshots:
        await set_last(snapshot_markdown(snapshot))


class SlotUI:
    def __init__(self, index):
        with gr.Column(visible=False) as self.card:
            with gr.Row():
                gr.Markdown(f"### Clip {index + 1}")
                self.regen_btn = gr.Button("Regenerate", size="sm", variant="secondary")
            self.chatbot = gr.Chatbot(label="Analysis Log", height=560, sanitize_html=False,
                                      render_markdown=True, type="messages", autoscroll=True)
            self.status = gr.Markdown("*Awaiting analysis...*")


async def analyze_clip(path, log, lock):
    try:
        video = read_dcm_3d_array(path)
        if video is None:
            raise ValueError("Unreadable DICOM")
        frames = subsample_frames(video)
        pil = [frame_to_square_pil(f, STUDY_EDGE) for f in frames]

        async with lock:
            log.reset()
            log.add("user", f"Analyzing {os.path.basename(path)}\n<image>", pil)
            log.add("assistant", "")

        async def set_last(t):
            async with lock:
                log.update_last(t)

        keyframe_result = await run_into(keyframe, frames, set_last)
        selection = keyframe_result.answer
        idx = selection.best_frame_idx if 0 <= selection.best_frame_idx < len(pil) else 0
        best = pil[idx]

        async with lock:
            log.add("user", "View classification.", [best])
            log.add("assistant", "Analyzing view...")

        async def set_view(t):
            async with lock:
                log.update_last(f"**View:**\n{t}")

        view_result = await run_into(view, best, set_view)
        v = view_result.answer
        return {"ok": True, "view": v, "best": best}
    except Exception as e:  # noqa: BLE001 - isolate failures to the affected clip panel
        async with lock:
            log.add("assistant", f"Inference failed: {e}")
        return {"ok": False, "error": str(e)}


async def run_pipeline_ui(files, cob_on, cob_k):
    if not files:
        yield {status_msg: "Please upload DICOM files to proceed."}
        return

    paths = [f.name for f in files][:MAX_SLOTS]
    n = len(paths)
    logs = [ChatLog() for _ in range(n)]
    locks = [asyncio.Lock() for _ in range(n)]
    results = [None] * n

    updates = {b: gr.update(interactive=False) for b in all_buttons}
    updates.update({status_msg: f"Processing {n} clip(s)...",
                    main_tabs: gr.update(selected="analysis_tab"), stored_paths: paths})
    for i in range(MAX_SLOTS):
        updates[slots[i].card] = gr.update(visible=(i < n))
        updates[slots[i].chatbot] = gr.update(value=[], autoscroll=True)
        updates[slots[i].status] = "*Queued...*" if i < n else ""
    yield updates

    async def work(i):
        results[i] = await analyze_clip(paths[i], logs[i], locks[i])

    tasks = [asyncio.create_task(work(i)) for i in range(n)]
    while not all(t.done() for t in tasks):
        for i in range(n):
            async with locks[i]:
                updates[slots[i].chatbot] = gr.update(value=logs[i].render(), autoscroll=True)
                updates[slots[i].status] = "Completed" if results[i] else "Processing..."
        yield updates
        await asyncio.sleep(0.1)

    for i in range(n):
        async with locks[i]:
            updates[slots[i].chatbot] = gr.update(value=logs[i].render(), autoscroll=True)
            updates[slots[i].status] = "Completed" if results[i] and results[i]["ok"] else "Failed"

    lca = [r["best"] for r in results if r and r.get("ok") and r.get("view") == "LCA"]
    rca = [r["best"] for r in results if r and r.get("ok") and r.get("view") == "RCA"]
    keyframes, note = pick_study_frames(lca, rca)
    updates[stored_keyframes] = keyframes
    if not keyframes:
        updates[status_msg] = f"### {note}"
        for b in all_buttons:
            updates[b] = gr.update(interactive=True)
        yield updates
        return

    updates[status_msg] = "Frames selected. Running study-level tasks..."
    updates[main_tabs] = gr.update(selected="report_tab")
    yield updates

    lock = asyncio.Lock()
    study = {}
    for key, (label, _) in STUDY_TASKS.items():
        log = ChatLog()
        log.add("user", label, keyframes)
        log.add("assistant", "")
        study[key] = log

    async def study_work(key):
        await run_study(STUDY_TASKS[key][1], keyframes, study[key], lock, cob_on, cob_k)

    tasks = [asyncio.create_task(study_work(k)) for k in STUDY_TASKS]
    while not all(t.done() for t in tasks):
        async with lock:
            for key, cb in study_chatbots.items():
                updates[cb] = gr.update(value=study[key].render(), autoscroll=True)
        yield updates
        await asyncio.sleep(0.1)
    await asyncio.gather(*tasks)

    async with lock:
        for key, cb in study_chatbots.items():
            updates[cb] = gr.update(value=study[key].render(), autoscroll=True)
    updates[status_msg] = "Pipeline completed."
    for b in all_buttons:
        updates[b] = gr.update(interactive=True)
    yield updates


async def regen_clip(idx, paths):
    if not paths or idx >= len(paths):
        yield [], "No file to regenerate."
        return
    log, lock = ChatLog(), asyncio.Lock()
    task = asyncio.create_task(analyze_clip(paths[idx], log, lock))
    while not task.done():
        async with lock:
            yield log.render(), "Re-analyzing..."
        await asyncio.sleep(0.1)
    async with lock:
        yield log.render(), "Re-analysis complete."


async def regen_study(keyframes, key, cob_on, cob_k):
    if not keyframes:
        yield []
        return
    label, task = STUDY_TASKS[key]
    log, lock = ChatLog(), asyncio.Lock()
    log.add("user", label, keyframes)
    log.add("assistant", "")

    worker = asyncio.create_task(run_study(task, keyframes, log, lock, cob_on, cob_k))
    while not worker.done():
        async with lock:
            yield log.render()
        await asyncio.sleep(0.1)
    await worker
    async with lock:
        yield log.render()


with gr.Blocks(title="CARDEA demo", theme=gr.themes.Default()) as demo:
    gr.Markdown("# CARDEA End-to-End CAG Pipeline")

    stored_paths = gr.State([])
    stored_keyframes = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload DICOM Files", file_count="multiple", file_types=[".dcm"])
            start_btn = gr.Button("Run Pipeline", variant="primary", interactive=False)
            cob_resample = gr.Checkbox(label="Chain-of-Box resampling — keep first rollout with a box (study level)", value=True)
            cob_k = gr.Number(label="samples (k)", value=7, precision=0, minimum=1, maximum=20)
        with gr.Column(scale=2):
            status_msg = gr.Markdown("### Awaiting data upload...")

    study_chatbots = {}
    study_regen_btns = {}
    with gr.Tabs() as main_tabs:
        with gr.Tab("1. Single-View Analysis", id="analysis_tab"):
            slots = []
            for r in range(5):
                with gr.Row():
                    for c in range(3):
                        slots.append(SlotUI(r * 3 + c))
        with gr.Tab("2. Study-Level Analysis", id="report_tab"), gr.Row():
            for key in STUDY_TASKS:
                with gr.Column():
                    with gr.Row():
                        gr.Markdown(f"### {STUDY_TITLES[key]}")
                        study_regen_btns[key] = gr.Button("Regenerate", size="sm")
                    study_chatbots[key] = gr.Chatbot(label=STUDY_TITLES[key], height=700,
                                                     sanitize_html=False, type="messages", autoscroll=True)

    all_buttons = [start_btn] + list(study_regen_btns.values()) + [s.regen_btn for s in slots]

    def toggle_start(files):
        if files and len(files) > MAX_SLOTS:
            raise gr.Error(f"At most {MAX_SLOTS} files can be uploaded.")
        return gr.update(interactive=bool(files))

    file_input.change(toggle_start, [file_input], [start_btn])
    file_input.clear(lambda: gr.update(interactive=False), None, [start_btn])

    outputs = [status_msg, main_tabs, stored_paths, stored_keyframes] + list(study_chatbots.values())
    for s in slots:
        outputs += [s.card, s.chatbot, s.status]
    outputs += all_buttons
    start_btn.click(run_pipeline_ui, [file_input, cob_resample, cob_k], outputs)

    def slot_regen(idx):
        async def fn(paths):
            u = {b: gr.update(interactive=False) for b in all_buttons}
            yield u
            async for chat, status in regen_clip(idx, paths):
                u[slots[idx].chatbot] = gr.update(value=chat, autoscroll=True)
                u[slots[idx].status] = status
                yield u
            for b in all_buttons:
                u[b] = gr.update(interactive=True)
            yield u
        return fn

    for i in range(MAX_SLOTS):
        slots[i].regen_btn.click(slot_regen(i), [stored_paths],
                                 [slots[i].chatbot, slots[i].status] + all_buttons)

    def study_regen(key):
        async def fn(keyframes, cob_on, cob_k):
            u = {b: gr.update(interactive=False) for b in all_buttons}
            yield u
            async for chat in regen_study(keyframes, key, cob_on, cob_k):
                u[study_chatbots[key]] = gr.update(value=chat, autoscroll=True)
                yield u
            for b in all_buttons:
                u[b] = gr.update(interactive=True)
            yield u
        return fn

    for key, btn in study_regen_btns.items():
        btn.click(study_regen(key), [stored_keyframes, cob_resample, cob_k], [study_chatbots[key]] + all_buttons)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("APP_PORT", "7860")))
