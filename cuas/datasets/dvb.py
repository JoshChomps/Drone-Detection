"""
Drone-vs-Bird challenge videos (WOSDETC; github.com/wosdetc/challenge; request by email + data usage agreement).

Annotation format, as documented on the challenge page (checked 2026-09-27/29): one text file per video,
one line per frame:
    framenum num_objs_in_frame obj1_x_left obj1_y_top obj1_w obj1_h obj1_class obj2_x_left ...

VERIFIED 2026-09-30 on the 77 public training annotation files (github.com/wosdetc/challenge/annotations; copy in
C:/cuas_data/dvb/annotations), all parsed by this module without error:
  * frame numbers start at 0, and every frame has a line (9,086 lines with 0 objects);
  * **only drones are annotated** (119,259 boxes, class "drone" only). Birds are unlabelled distractors, so a
    "bird" track is a detector track that matches no drone label; audit a sample by eye, since "no label" has
    already proved not to mean "no drone" in Halmstad (14 §5). "parrot" in a file name is the Parrot drone brand.
  * box widths 4-1,403 px, median 25; 29% under 16 px (real small drones, not simulated).
The TEST videos and their annotations are withheld by the organisers (challenge test set); split the training
videos by video into our own train/val/test.
STILL UNVERIFIED until the videos arrive: that every video received has an annotation file (the repo was last
updated 2022-12; a newer edition may send more videos) and the frame rate per video.

    ann = parse_annotations(Path("clip.txt").read_text())      # {frame: [(x, y, w, h, class_name), ...]}
    for t, boxes in label_stream(ann, fps=25): ...
"""


def parse_annotations(text: str) -> dict:
    out = {}
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        frame, count = int(parts[0]), int(parts[1])
        if len(parts) != 2 + 5 * count:
            raise ValueError(f"line {n}: {count} objects need {2 + 5 * count} fields, found {len(parts)}")
        boxes = []
        for k in range(count):
            x, y, w, h = (float(v) for v in parts[2 + 5 * k: 6 + 5 * k])
            boxes.append((x, y, w, h, parts[6 + 5 * k].lower()))
        out[frame] = boxes
    return out


def label_stream(ann: dict, fps: float, first_frame: int = None):
    """(t, boxes) for every frame from the first to the last annotated one; frames with no line give []."""
    if not ann:
        return
    f0 = min(ann) if first_frame is None else first_frame
    for f in range(f0, max(ann) + 1):
        yield (f - f0) / fps, ann.get(f, [])
