# Conversation Analyzer

Analyze stereo WAV recordings of two-person conversations. Speaker A is on the left channel, Speaker B is on the right.

Computes turn-taking, silence, and interruption statistics — with emphasis on **turn-taking latency** (response time) and **interruption yielding latency** — and generates a self-contained HTML report with interactive Plotly charts.

## Install

```sh
pip install -e .
```

## Usage

```sh
conversation-analyzer recording.wav
```

This produces `recording_report.html` in the current directory.

### Options

```
-o, --output PATH        Output HTML report path (default: <input>_report.html)
-t, --threshold FLOAT    RMS energy threshold for VAD (default: auto-detect)
-a, --speaker-a TEXT     Label for left channel (default: "Speaker A")
-b, --speaker-b TEXT     Label for right channel (default: "Speaker B")
--frame-size INT         VAD frame size in ms (default: 30)
--min-speech INT         Min speech segment duration in ms (default: 200)
--min-silence INT        Min silence gap to split segments in ms (default: 300)
```

## Output

A single self-contained HTML file with:

- Summary statistics table (per-speaker)
- Timeline view of speech activity
- Talk time pie chart
- Turn duration histogram
- Cumulative talk time chart
- Response time (turn-taking latency) histogram
- Yielding latency histogram

## Development

```sh
pip install -e ".[dev]"
pytest
```
