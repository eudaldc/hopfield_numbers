#!/usr/bin/env python
"""Narration for the blog video with Kokoro-82M (runs in its own venv).

    <tts-venv>/bin/python scripts/tts_kokoro.py cues.json OUT_DIR --voice af_heart

cues.json is a list of {"id": ..., "text": ...}; each cue becomes OUT_DIR/<id>.wav
(24 kHz mono). Existing files are kept, so only edited cues are re-synthesised.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import misaki.espeak  # noqa: F401  (points phonemizer at the bundled espeak-ng)
from kokoro import KPipeline
from phonemizer.backend.espeak.wrapper import EspeakWrapper

# The bundled espeak-ng library looks for its data at a path baked in at build
# time (absent here) and phonemizer ignores set_data_path; use the system one.
EspeakWrapper.set_library("/usr/lib64/libespeak-ng.so.1")
EspeakWrapper.set_data_path("/usr/share/espeak-ng-data")

SR = 24000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cues", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--lang", default="a", help="Kokoro language code (a = US English, b = UK)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in json.loads(args.cues.read_text())
            if not (args.out / f"{c['id']}.wav").exists()]
    if not todo:
        return
    pipe = KPipeline(lang_code=args.lang, repo_id="hexgrad/Kokoro-82M")
    for c in todo:
        parts = [np.asarray(a) for _, _, a in pipe(c["text"], voice=args.voice, speed=args.speed)]
        sf.write(args.out / f"{c['id']}.wav", np.concatenate(parts), SR, subtype="PCM_16")
        print(f"  {c['id']}")


if __name__ == "__main__":
    main()
