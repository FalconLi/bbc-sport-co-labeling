"""Export the shared animated replay and a chart of the existing evaluation.

Run with python render_presentation.py after installing requirements-presentation.txt.
No learning code is changed and the ten-run experiment is not repeated.
"""

from __future__ import annotations

import os

for variable in ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")

import argparse
import math
import shutil
import tempfile
from pathlib import Path

from colabeling.data import ROOT
from presentation_data import build_story

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900
BG, INK, MUTED, GREEN = "#F7F9F7", "#203832", "#65776F", "#087F67"
BLUE, ORANGE, LINE = "#377DA1", "#B5733E", "#DCE5DF"


def clamp(value):
    return max(0.0, min(1.0, value))


def ease(value):
    p = clamp(value)
    return p * p * (3 - 2 * p)


def blend(first, second, amount):
    a = tuple(int(first[i:i + 2], 16) for i in [1, 3, 5])
    b = tuple(int(second[i:i + 2], 16) for i in [1, 3, 5])
    return tuple(round(x + (y - x) * clamp(amount)) for x, y in zip(a, b))


class StoryRenderer:
    def __init__(self, data):
        self.data = data
        self.fonts = {}
        self.font_paths = {
            False: font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans")),
            True: font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")),
        }
        self.old_indices = [i for i, s in enumerate(data["sheets"]) if s["origin_round"] == 0]
        self.new_indices = [i for i, s in enumerate(data["sheets"]) if s["origin_round"] == 1]
        self.sampled = [data["optimizer"]["snapshots"][i] for i in data["optimizer"]["display_indices"]]

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(self.font_paths[bold], size)
        return self.fonts[key]

    def text(self, x, y, value, size=24, color=INK, bold=False, anchor="lt"):
        self.draw.text((x, y), str(value), font=self.font(size, bold), fill=color, anchor=anchor)

    def paragraph(self, x, y, value, width, size=24, color=INK, bold=False, line_height=None):
        line_height = line_height or round(size * 1.45)
        line = ""
        for word in value.split():
            proposed = f"{line} {word}".strip()
            if line and self.font(size, bold).getlength(proposed) > width:
                self.text(x, y, line, size, color, bold)
                y += line_height
                line = word
            else:
                line = proposed
        if line:
            self.text(x, y, line, size, color, bold)
        return y + line_height

    def rect(self, box, fill="white", outline=None, radius=10, width=1):
        self.draw.rounded_rectangle(tuple(round(v) for v in box), radius=radius, fill=fill, outline=outline, width=width)

    def arrow(self, start, end, color=GREEN, width=3, packet=None):
        self.draw.line([start, end], fill=color, width=width)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        points = [end, (end[0] - 14 * math.cos(angle - .45), end[1] - 14 * math.sin(angle - .45)),
                  (end[0] - 14 * math.cos(angle + .45), end[1] - 14 * math.sin(angle + .45))]
        self.draw.polygon(points, fill=color)
        if packet is not None:
            p = clamp(packet)
            x, y = start[0] + (end[0] - start[0]) * p, start[1] + (end[1] - start[1]) * p
            self.rect((x - 17, y - 22, x + 17, y + 22), "white", color, 3, 2)
            for offset in [-9, 0, 9]:
                self.draw.line((x - 10, y + offset, x + 10, y + offset), fill=color, width=2)

    def header(self, title, subtitle, t):
        self.text(80, 38, "BBCSPORT  /  INDEPENDENT IMPLEMENTATION", 17, GREEN, True)
        self.text(1520, 38, "REAL TRAINING REPLAY", 16, MUTED, anchor="rt")
        self.text(80, 88, title, 43, INK, True)
        self.text(80, 153, subtitle, 24, MUTED)
        self.draw.line((80, 205, 1520, 205), fill=LINE, width=2)
        for chapter in self.data["chapters"]:
            x1 = 80 + 1440 * chapter["start"] / 90
            x2 = 80 + 1440 * chapter["end"] / 90 - 5
            self.rect((x1, 870, x2, 877), LINE, radius=3)
            if t > chapter["start"]:
                progress = clamp((t - chapter["start"]) / (chapter["end"] - chapter["start"]))
                self.rect((x1, 870, x1 + (x2 - x1) * progress, 877), GREEN, radius=3)

    def caption(self, first, second):
        self.draw.line((80, 756, 1520, 756), fill=LINE, width=2)
        self.text(80, 778, first, 26, INK, True)
        self.text(80, 822, second, 20, MUTED)

    def legend(self, y=703):
        x = 100
        for name, color in zip(self.data["categories"], self.data["colors"]):
            self.rect((x, y, x + 15, y + 15), color, radius=3)
            self.text(x + 24, y - 2, name.capitalize(), 18, MUTED)
            x += 150

    def grid(self, at_round, progress=1, *, reveal=False, t=0):
        state = self.data["rounds"][at_round]
        old = self.data["rounds"][max(0, at_round - 1)]["predictions"]
        changed = state["changed"]
        for i, category in enumerate(state["predictions"]):
            x, y = 100 + (i % 30) * 25, 360 + (i // 30) * 25
            local = ease((progress - (i / 360) * .40) / .60)
            target = self.data["colors"][category]
            if reveal:
                color = blend(LINE, target, local)
            elif changed[i]:
                color = blend(self.data["colors"][old[i]], target, local)
            else:
                color = target
            self.rect((x, y, x + 21, y + 21), color, radius=3)
            if not reveal and changed[i] and progress < 1:
                pulse = .4 + .6 * abs(math.sin(t * 5))
                outline = blend(BG, INK, pulse)
                self.draw.rounded_rectangle((x - 2, y - 2, x + 23, y + 23), radius=4, outline=outline, width=2)
        self.text(100, 674, "360 articles. Each square keeps the same article throughout.", 19, MUTED)
        self.legend()

    def readers(self, y=255):
        self.rect((975, y, 1205, y + 100), "#EAF1F7", radius=8)
        self.rect((1265, y, 1495, y + 100), "#F7EEE6", radius=8)
        self.text(1090, y + 15, "Reader A", 26, BLUE, True, "mt")
        self.text(1380, y + 15, "Reader B", 26, ORANGE, True, "mt")
        self.text(1090, y + 61, "2,306 word features", 16, BLUE, anchor="mt")
        self.text(1380, y + 61, "2,307 word features", 16, ORANGE, anchor="mt")

    def prediction_scene(self, at_round, progress, t, *, initial=False, fast=False):
        self.readers()
        if initial:
            for i, article in enumerate(self.data["known"]):
                x = 100 + 75 * i
                color = self.data["colors"][article["category"]]
                self.rect((x, 246, x + 66, 292), blend("#FFFFFF", color, .19), color, 5)
                self.text(x + 33, 261, f"#{article['id']}", 15, INK, anchor="mt")
            self.text(100, 311, "Ten supplied labels, two per sport", 22, INK, True)
            self.grid(0, progress, reveal=True, t=t)
            self.text(980, 412, "Already trained", 33, GREEN, True)
            self.paragraph(980, 468, "Both readers learned from the ten known labels before making these predictions.", 510, 25)
            if t < 8:
                self.arrow((1010, 597), (880, 597), BLUE, packet=(t % 2) / 2)
                self.text(1010, 638, "Predictions fill the grid", 21, MUTED)
        else:
            state = self.data["rounds"][at_round]
            self.text(100, 265, f"ROUND {at_round}", 23, GREEN, True)
            self.text(100, 311, "Updated sport predictions", 24, INK, True)
            self.grid(at_round, progress, t=t)
            self.text(975, 400, str(state["changed_count"]), 110, GREEN, True)
            self.text(975, 529, "of 360 predictions changed", 26, INK)
            self.text(975, 583, "Recorded changes since the previous round", 18, MUTED)
            counts = state["football_sheets"]
            self.text(975, 638, f"Football sheets kept: A {counts[0]} / B {counts[1]}", 22, MUTED)
            if progress < 1:
                self.arrow((1030, 729), (1430, 729), GREEN, packet=(t % 1.5) / 1.5)

    def sheet_creation(self, t):
        self.text(100, 242, "Reader B's football ranking", 25, ORANGE, True)
        self.text(623, 242, "11 new label lists", 25, GREEN, True)
        self.text(100, 290, "Rank", 17, MUTED)
        self.text(220, 290, "Article", 17, MUTED)
        self.text(380, 290, "Score*", 17, MUTED)
        progress = clamp((t - 18) / 7)
        active = min(10, int(progress * 11))
        for i, article in enumerate(self.data["preview"]):
            y = 353 + i * 48
            self.text(100, y, article["rank"], 24, MUTED)
            self.text(220, y, f"#{article['id']}", 24, INK, True)
            self.text(380, y + 3, f"{article['score']:+.4f}", 18, MUTED)
        for j, index in enumerate(self.new_indices):
            sheet = self.data["sheets"][index]
            x = 623 + j * 78
            visible = j <= active
            self.text(x + 27, 290, f"{sheet['fraction']:.0%}", 19, GREEN if visible else MUTED, True, "mt")
            self.text(x + 27, 320, sheet["football_count"] if visible else "", 18, MUTED, anchor="mt")
            for i, answer in enumerate(sheet["preview"]):
                y = 349 + i * 48
                fill = self.data["colors"][2] if answer == 1 else "#E7ECE8"
                self.rect((x + 7, y, x + 49, y + 34), fill if visible else LINE, radius=4)
                if visible:
                    self.text(x + 28, y + 7, "F" if answer == 1 else "-", 19, INK, True, "mt")
            if j == active and t < 25:
                self.draw.rounded_rectangle((x - 3, 279, x + 59, 636), radius=6, outline=GREEN, width=2)
        self.text(100, 671, "*Classifier score, not a probability", 18, MUTED)
        self.text(623, 671, "F = football.  - = another sport.", 20, MUTED)
        self.text(623, 708, "The number below each percentage counts football labels out of 360.", 18, MUTED)

    def exchange(self, t):
        self.text(280, 242, "Reader A", 37, BLUE, True, "mt")
        self.text(1320, 242, "Reader B", 37, ORANGE, True, "mt")
        self.text(800, 267, "11 new football sheets in each direction", 24, INK, anchor="mt")
        self.arrow((1240, 332), (360, 332), ORANGE, 4, packet=((t - 26) % 2.4) / 2.4)
        self.arrow((360, 401), (1240, 401), BLUE, 4, packet=((t - 26.8) % 2.4) / 2.4)
        self.text(800, 443, "Reader A keeps 11 earlier sheets and adds B's 11 new sheets", 26, INK, True, "mt")
        expansion = ease((t - 28) / 4)
        for j in range(11):
            x = 107 + j * 127
            self.text(x + 48, 505, f"{10 + j * 2}%", 22, MUTED, True, "mt")
            self.rect((x, 549, x + 96, 592), "#E6EFF6", BLUE, 5)
            self.text(x + 48, 561, "earlier", 17, BLUE, anchor="mt")
            start_x = 1250
            new_x = start_x + (x - start_x) * expansion
            new_y = 355 + (612 - 355) * expansion
            self.rect((new_x, new_y, new_x + 96, new_y + 43), "#E3F1E9", GREEN, 5)
            self.text(new_x + 48, new_y + 12, "new", 17, GREEN, anchor="mt")
        self.text(800, 707, "Two layers: group sheets made with the same percentage", 24, MUTED, anchor="mt")

    def weighting(self, t):
        phase = clamp((t - 35) / 20) * (len(self.sampled) - 1)
        index = int(phase)
        current = self.sampled[index]
        previous = self.sampled[max(0, index - 1)]
        # Numeric labels always show an actual observed state. Only bar geometry
        # is eased between the preceding and current observed states.
        movement = ease(min(1, (phase - index) * 3)) if index else 1
        shares = np.asarray(previous["shares"]) + (np.asarray(current["shares"]) - previous["shares"]) * movement
        self.text(100, 235, "Fit Reader A to its word patterns", 23, BLUE, True)
        self.arrow((565, 249), (800, 249), GREEN, packet=((t - 35) % 2) / 2)
        self.text(840, 235, "Update the sheets' influence, then fit again", 23, GREEN, True)
        self.text(100, 290, "Football in sheet", 18, MUTED, True)
        self.text(550, 290, "Earlier sheets (11)", 21, BLUE, True, "mt")
        self.text(1180, 290, "New sheets (11)", 21, GREEN, True, "mt")
        maximum = max(10, math.ceil(max(max(s["shares"]) for s in self.sampled) / 2) * 2)
        lefts, length = [320, 950], 410
        for left in lefts:
            for value in [0, maximum / 2, maximum]:
                x = left + length * value / maximum
                self.draw.line((x, 346, x, 669), fill=LINE, width=1)
                self.text(x, 324, f"{value:g}%", 15, MUTED, anchor="mt")
        for j in range(11):
            y = 351 + j * 29
            self.text(100, y + 1, f"{10 + j * 2}% of articles", 18, MUTED)
            for column, source_indices in enumerate([self.old_indices, self.new_indices]):
                k = source_indices[j]
                x, color = lefts[column], [BLUE, GREEN][column]
                self.rect((x, y, x + length, y + 18), "#E9EEEA", radius=3)
                if shares[k] > 0:
                    self.rect((x, y, x + length * shares[k] / maximum, y + 18), color, radius=3)
                self.text(x + length + 17, y - 1, f"{current['shares'][k]:.2f}%", 18, INK, True)
        self.text(100, 696, "Share of learned weight", 23, INK, True)
        self.text(1495, 696, f"{current['label']}  /  sampled real updates", 18, MUTED, anchor="rt")

    def frame(self, t):
        t = max(0.0, min(t, 90 - 1e-6))
        self.image = Image.new("RGB", (W, H), BG)
        self.draw = ImageDraw.Draw(self.image)
        if t < 12:
            self.header("Start with two trained readers", "Each classifier has already learned from the ten supplied labels.", t)
            self.prediction_scene(0, clamp((t - 3) / 7), t, initial=True)
            self.caption("The first predictions come from trained classifiers.", "Their only supplied answers are the ten labeled articles. The other 360 labels stay hidden.")
        elif t < 18:
            self.header("The first exchange changes predictions", "The same article positions make changes visible across training rounds.", t)
            self.prediction_scene(1, clamp((t - 12) / 4.5), t)
            self.caption("Round 1 changes 98 of the 360 predictions.", "These are actual label changes. A changed prediction is not automatically a correct prediction.")
        elif t < 26:
            self.header("Round 2: create several possible answer sheets", "Reader B ranks articles from most to least likely to be about football.", t)
            self.sheet_creation(t)
            self.caption("Try different numbers of football labels instead of trusting one guess.", "Each column is a real label list. Six articles near the cutoffs make the differences visible.")
        elif t < 35:
            self.header("Exchange new sheets and keep earlier ones", "We follow the football sheets sent from Reader B to Reader A.", t)
            self.exchange(t)
            self.caption("11 earlier sheets + 11 new sheets = 22 alternatives for Reader A.", "Sheets made with the same percentage belong to the same group. Reader B also receives sheets from A.")
        elif t < 60:
            self.header("Learn how much influence each sheet gets", "Reader A combines B's proposed labels using Reader A's own word features.", t)
            self.weighting(t)
            if t < 43:
                first, second = "Start equally, then adjust the classifier and the weights together.", "The program looks for a separation of football and other articles that fits Reader A's word patterns."
            elif t < 54:
                first, second = "Fit the classifier, update influence, and fit again.", "Numbers are real optimizer snapshots. Smooth bar movement connects those recorded states."
            else:
                first, second = "Some sheets finish with little or zero influence.", "Influence is a share of learned weight. It is not confidence or the percentage of correct answers."
            self.caption(first, second)
        elif t < 70:
            self.header("Round 2: the learned readers update their predictions", "The same process is repeated for the other four sports.", t)
            self.prediction_scene(2, clamp((t - 60) / 6.5), t)
            self.caption("35 of the 360 predictions change in this recorded round.", "The new predictions will generate the next round's sheets. Earlier useful alternatives remain available.")
        elif t < 85:
            round_index = min(8, 3 + int((t - 70) / 2.5))
            local = ((t - 70) % 2.5) / 1.6
            self.header(f"Repeat the process: round {round_index} of 8", "Create sheets, exchange them, learn their influence, and update the readers.", t)
            self.prediction_scene(round_index, clamp(local), t, fast=True)
            self.caption("Later rounds are replayed faster, using their saved predictions.", "No label changes occur in the last two rounds of this recording; the display preserves that result.")
        else:
            self.header("Training finished. Now compare on new articles.", "A training animation explains the procedure; test results measure its performance.", t)
            for x, number, label in [(310, "8", "rounds replayed"), (800, "360", "unlabeled articles"), (1290, "367", "separate test articles")]:
                self.text(x, 315, number, 110, GREEN, True, "mt")
                self.text(x, 464, label, 28, INK, anchor="mt")
            self.text(800, 594, "Return to slide 7 for the ten-run comparison", 35, INK, True, "mt")
            self.caption("Compare against conventional co-training and the ten-label baseline.", "The results chart reports our implementation, averaged across ten runs, with variation between runs.")
        return self.image


def render_chart(data, png, pdf=None):
    results = data["results"]
    means = [r["map_mean"] for r in results]
    stds = [r["map_std"] for r in results]
    colors = ["#A2AEA8", "#5286B2", "#69A693", GREEN]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 13, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(4.5, 2.25), dpi=300)
        fig.subplots_adjust(left=.43, right=.94, top=.98, bottom=.31)
        y = np.arange(4)
        ax.barh(y, means, height=.56, color=colors, xerr=stds,
                error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "#354D44"}, zorder=3)
        ax.set_yticks(y, ["Ten labels only", "Co-training", "Co-labeling, 1 layer", "Co-labeling, 2 layers"])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Mean average precision (%)", labelpad=7, fontsize=11)
        ax.grid(axis="x", color="#E5EBE7", linewidth=.7, zorder=0)
        ax.tick_params(axis="both", length=0, labelsize=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, value in enumerate(means):
            ax.text(4, i, f"{value:.2f}", ha="left", va="center", fontsize=12,
                    color="white", weight="bold", zorder=5)
        fig.text(.04, .025, "Mean ± SD; 10 runs; 367 test articles per run", fontsize=9, color=MUTED)
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, facecolor="white")
        if pdf is not None:
            pdf.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pdf, facecolor="white", metadata={"Title": "Our BBCSport implementation: measured test results"})
        plt.close(fig)


def render_video(renderer, destination, fps):
    # imageio-ffmpeg supplies FFmpeg; no separate system install is required.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="colabeling-video-") as temporary:
        movie = Path(temporary) / "animation.mp4"
        writer = imageio_ffmpeg.write_frames(
            str(movie), (W, H), fps=fps, codec="libx264", quality=None, pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            macro_block_size=1, ffmpeg_log_level="error",
            output_params=["-crf", "19", "-preset", "fast", "-movflags", "+faststart"],
        )
        writer.send(None)
        try:
            for i in range(90 * fps):
                writer.send(renderer.frame(i / fps).tobytes())
                if i % (15 * fps) == 0:
                    print(f"Rendering animation: {i // fps}/90 seconds", flush=True)
        finally:
            writer.close()
        shutil.copy2(movie, destination)
    print(f"Saved 90-second animation: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Export the real training animation and measured results chart.")
    default_slides = ROOT.parent / "Co_Labeling___HKU_Paper_Discussion"
    parser.add_argument("--slides-dir", type=Path, default=default_slides if default_slides.exists() else None)
    parser.add_argument("--refresh-trace", action="store_true", help="Observe the focused solver replay again.")
    parser.add_argument("--skip-video", action="store_true", help="Only regenerate replay data, poster, and chart.")
    parser.add_argument("--preview-dir", type=Path, help="Optional location for scene preview images.")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error("--fps must be between 1 and 60.")
    data = build_story(refresh=args.refresh_trace)
    renderer = StoryRenderer(data)
    demo = ROOT / "demo"
    pdf = args.slides_dir / "images/implementation-results.pdf" if args.slides_dir else None
    render_chart(data, demo / "training-results.png", pdf)
    renderer.frame(9).save(demo / "training-poster.png")
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        for t in [1, 9, 15, 22, 32, 36, 48, 58, 66, 76, 88]:
            renderer.frame(t).save(args.preview_dir / f"scene-{t:02d}.png")
    if not args.skip_video:
        render_video(renderer, demo / "training-animation.mp4", args.fps)
        if args.slides_dir:
            shutil.copy2(demo / "training-animation.mp4", args.slides_dir / "bbc-sport-demo.mp4")
    print("Ready: open the demo, or use the exported MP4 and results chart.", flush=True)


if __name__ == "__main__":
    main()
