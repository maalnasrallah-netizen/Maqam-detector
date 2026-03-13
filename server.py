"""
Maqam Detector — Backend Server
Extracts audio features and classifies Arabic maqamat.
"""

import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa

app = Flask(__name__)
CORS(app)

# ─── Maqam Definitions ───────────────────────────────────────────────
# Each maqam is defined by its interval pattern (in cents from root)
# and characteristic jins (tetrachord/pentachord)

MAQAMAT = {
    "Bayati": {
        "ar": "بياتي",
        "intervals_cents": [0, 150, 300, 500, 700, 850, 1000, 1200],
        "jins_lower": "Bayati",
        "jins_upper": "Nahawand",
        "characteristic": "3/4 flat on 2nd degree",
        "common_root": "D",
    },
    "Rast": {
        "ar": "راست",
        "intervals_cents": [0, 200, 350, 500, 700, 900, 1050, 1200],
        "jins_lower": "Rast",
        "jins_upper": "Rast",
        "characteristic": "3/4 flat on 3rd and 7th degrees",
        "common_root": "C",
    },
    "Hijaz": {
        "ar": "حجاز",
        "intervals_cents": [0, 100, 400, 500, 700, 850, 1000, 1200],
        "jins_lower": "Hijaz",
        "jins_upper": "Nahawand",
        "characteristic": "augmented 2nd between 2nd-3rd degrees",
        "common_root": "D",
    },
    "Saba": {
        "ar": "صبا",
        "intervals_cents": [0, 150, 300, 400, 700, 800, 1000, 1200],
        "jins_lower": "Saba",
        "jins_upper": "Hijaz",
        "characteristic": "diminished feel, flattened 4th",
        "common_root": "D",
    },
    "Nahawand": {
        "ar": "نهاوند",
        "intervals_cents": [0, 200, 300, 500, 700, 800, 1000, 1200],
        "jins_lower": "Nahawand",
        "jins_upper": "Kurd",
        "characteristic": "similar to Western natural minor",
        "common_root": "C",
    },
    "Ajam": {
        "ar": "عجم",
        "intervals_cents": [0, 200, 400, 500, 700, 900, 1100, 1200],
        "jins_lower": "Ajam",
        "jins_upper": "Ajam",
        "characteristic": "similar to Western major scale",
        "common_root": "Bb",
    },
    "Kurd": {
        "ar": "كرد",
        "intervals_cents": [0, 100, 300, 500, 700, 800, 1000, 1200],
        "jins_lower": "Kurd",
        "jins_upper": "Nahawand",
        "characteristic": "minor 2nd from root, Phrygian feel",
        "common_root": "D",
    },
    "Sikah": {
        "ar": "سيكاه",
        "intervals_cents": [0, 150, 350, 500, 700, 850, 1050, 1200],
        "jins_lower": "Sikah",
        "jins_upper": "Rast",
        "characteristic": "starts on E half-flat",
        "common_root": "E",
    },
}

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


# ─── Audio Feature Extraction ────────────────────────────────────────

def extract_features(audio_bytes, sr=22050):
    """Load audio from bytes and extract pitch/interval features."""

    # Load audio
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) < sr * 0.5:
        raise ValueError("Audio too short — need at least 0.5 seconds")

    # ── Pitch tracking using pYIN (built into librosa) ──
    # pYIN is excellent for monophonic/vocal pitch detection
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=80, fmax=1000, sr=sr, frame_length=2048
    )

    # Filter to only confident voiced frames
    mask = voiced_flag & (voiced_prob > 0.6)
    f0_voiced = f0[mask]
    probs_voiced = voiced_prob[mask]

    if len(f0_voiced) < 10:
        raise ValueError("Not enough pitched content detected")

    # ── Convert to cents relative to estimated root ──
    root_hz = estimate_root(f0_voiced)
    cents = 1200 * np.log2(f0_voiced / root_hz)

    # Wrap to single octave (0-1200 cents)
    cents_wrapped = cents % 1200

    # ── Build pitch histogram (120 bins = 10 cents each) ──
    hist, bin_edges = np.histogram(cents_wrapped, bins=120, range=(0, 1200))
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-9  # normalize

    # ── Extract scale degrees (peaks in histogram) ──
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(hist, height=0.01, distance=5, prominence=0.005)
    peak_cents = (peaks * 10) + 5  # center of each bin

    # ── Detect root note name ──
    root_midi = librosa.hz_to_midi(root_hz)
    root_note_idx = int(round(root_midi)) % 12
    root_name = NOTE_NAMES[root_note_idx]
    root_octave = int(round(root_midi)) // 12 - 1

    # ── Compute chroma features for additional context ──
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=24)  # quarter-tone
    chroma_mean = chroma.mean(axis=1)

    return {
        "root_hz": float(root_hz),
        "root_name": root_name,
        "root_octave": root_octave,
        "pitch_histogram": hist.tolist(),
        "scale_degree_cents": peak_cents.tolist(),
        "chroma_24": chroma_mean.tolist(),
        "n_voiced_frames": int(len(f0_voiced)),
        "duration_sec": float(len(y) / sr),
    }


def estimate_root(f0_voiced):
    """
    Estimate the tonal root (qarar) from pitched frames.
    Uses a combination of:
    1. Most frequent pitch class
    2. Lowest sustained pitch
    3. Final pitch (Arabic music often cadences on root)
    """
    # Convert to MIDI for easier pitch class analysis
    midi = librosa.hz_to_midi(f0_voiced)
    pitch_classes = midi % 12

    # Histogram of pitch classes (24 bins for quarter-tone)
    hist, edges = np.histogram(pitch_classes, bins=24, range=(0, 12))

    # Weight: last 20% of frames get extra weight (cadence)
    n = len(f0_voiced)
    last_20 = f0_voiced[int(n * 0.8):]
    midi_last = librosa.hz_to_midi(last_20)
    pc_last = midi_last % 12
    hist_last, _ = np.histogram(pc_last, bins=24, range=(0, 12))
    hist_combined = hist + hist_last * 2  # boost cadence weight

    # Find most common pitch class
    best_bin = np.argmax(hist_combined)
    best_pc = (best_bin / 24) * 12  # back to MIDI pitch class

    # Find the lowest octave instance of this pitch class
    target_midi = []
    for m in midi:
        if abs((m % 12) - best_pc) < 0.75:
            target_midi.append(m)

    root_midi = np.median(target_midi) if target_midi else np.median(midi)
    return float(librosa.midi_to_hz(root_midi))


# ─── Maqam Classification ────────────────────────────────────────────

def classify_maqam(features):
    """
    Classify the maqam by comparing extracted scale degrees
    against known maqam interval templates.

    Returns sorted list of (maqam_name, confidence_score).
    """
    observed_cents = np.array(features["scale_degree_cents"])
    histogram = np.array(features["pitch_histogram"])

    scores = {}

    for name, maq in MAQAMAT.items():
        template = np.array(maq["intervals_cents"])

        # ── Method 1: Interval matching ──
        # For each observed scale degree, find closest template degree
        interval_score = 0
        for obs in observed_cents:
            diffs = np.abs(template - obs)
            min_diff = np.min(diffs)
            # Score inversely proportional to distance (max 50 cents tolerance)
            if min_diff <= 50:
                interval_score += 1 - (min_diff / 50)

        # Normalize by number of template notes matched
        interval_score /= max(len(template), 1)

        # ── Method 2: Histogram correlation ──
        # Build a synthetic histogram from the template
        synth_hist = np.zeros(120)
        for cent in template:
            bin_idx = int(cent / 10)
            if bin_idx < 120:
                # Gaussian spread around each scale degree
                for offset in range(-3, 4):
                    idx = (bin_idx + offset) % 120
                    synth_hist[idx] += np.exp(-0.5 * (offset / 1.5) ** 2)
        synth_hist /= synth_hist.sum() + 1e-9

        corr = np.corrcoef(histogram, synth_hist)[0, 1]
        corr = max(0, corr)  # clamp negative

        # ── Method 3: Characteristic interval check ──
        # Some maqamat have "signature" intervals that are strong identifiers
        char_bonus = check_characteristic_intervals(observed_cents, name)

        # Weighted combination
        total = (interval_score * 0.4) + (corr * 0.4) + (char_bonus * 0.2)
        scores[name] = total

    # Normalize to percentages
    max_score = max(scores.values()) if scores else 1
    results = []
    for name, score in scores.items():
        pct = int((score / max_score) * 100) if max_score > 0 else 0
        pct = min(98, max(5, pct))  # clamp to 5-98%
        results.append({
            "name": name,
            "ar": MAQAMAT[name]["ar"],
            "confidence": pct,
            "jins": MAQAMAT[name]["jins_lower"],
            "characteristic": MAQAMAT[name]["characteristic"],
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def check_characteristic_intervals(observed, maqam_name):
    """Check for signature intervals that strongly indicate a specific maqam."""
    bonus = 0

    if maqam_name == "Hijaz":
        # Look for augmented 2nd (~300 cents gap between 2nd and 3rd degree)
        for i in range(len(observed) - 1):
            gap = observed[i + 1] - observed[i]
            if 250 < gap < 350:
                bonus += 0.5
                break

    elif maqam_name == "Bayati" or maqam_name == "Sikah":
        # Look for 3/4 tone (~150 cents) in lower jins
        has_three_quarter = any(130 < c < 170 for c in observed if c < 400)
        if has_three_quarter:
            bonus += 0.4

    elif maqam_name == "Saba":
        # Look for the diminished-feeling 4th (~400 cents)
        has_dim4 = any(380 < c < 420 for c in observed)
        if has_dim4:
            bonus += 0.5

    elif maqam_name == "Ajam":
        # Should NOT have any 3/4 tones — pure Western major
        has_three_quarter = any(130 < c < 170 for c in observed[:4])
        if not has_three_quarter:
            bonus += 0.3

    return min(1.0, bonus)


# ─── Genre Detection ─────────────────────────────────────────────────

def detect_genre(features):
    """Simple heuristic genre detection based on audio characteristics."""
    duration = features["duration_sec"]
    n_frames = features["n_voiced_frames"]
    voice_density = n_frames / max(duration * 43, 1)  # ~43 frames/sec at default hop

    # Very high voice density + sustained phrases = likely recitation
    if voice_density > 0.7:
        return "Quran recitation"
    elif voice_density > 0.5:
        return "Vocal performance"
    elif voice_density > 0.3:
        return "Instrumental (oud/qanun)"
    else:
        return "Mixed/ensemble"


# ─── API Routes ───────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    """Main detection endpoint. Accepts audio file upload."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if len(audio_bytes) == 0:
        return jsonify({"error": "Empty audio file"}), 400

    if len(audio_bytes) > 30 * 1024 * 1024:
        return jsonify({"error": "File too large (max 30 MB)"}), 400

    try:
        # Extract features
        features = extract_features(audio_bytes)

        # Classify maqam
        results = classify_maqam(features)

        # Detect genre
        genre = detect_genre(features)

        # Build response
        primary = results[0]
        alternatives = results[1:4]

        return jsonify({
            "success": True,
            "maqam": {
                "name": primary["name"],
                "ar": primary["ar"],
                "confidence": primary["confidence"],
                "jins": primary["jins"],
                "characteristic": primary["characteristic"],
            },
            "root_note": f"{features['root_name']}{features['root_octave']}",
            "genre": genre,
            "alternatives": [
                {"name": a["name"], "ar": a["ar"], "confidence": a["confidence"]}
                for a in alternatives
            ],
            "debug": {
                "scale_degrees_cents": features["scale_degree_cents"],
                "duration_sec": round(features["duration_sec"], 1),
                "voiced_frames": features["n_voiced_frames"],
            },
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "maqamat_supported": list(MAQAMAT.keys())})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
