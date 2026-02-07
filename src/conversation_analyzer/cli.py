"""CLI entry point for conversation-analyzer."""

from pathlib import Path

import click

from .audio import load_wav
from .report import generate_report
from .stats import compute_stats
from .vad import detect_speech


@click.command()
@click.argument("wav_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(), default=None,
              help="Output HTML report path (default: <input>_report.html)")
@click.option("-t", "--threshold", type=float, default=None,
              help="RMS energy threshold (default: auto-detect)")
@click.option("-a", "--speaker-a", default="Human",
              help="Label for left channel")
@click.option("-b", "--speaker-b", default="Maya",
              help="Label for right channel")
@click.option("--frame-size", type=int, default=30,
              help="VAD frame size in ms")
@click.option("--min-speech", type=int, default=200,
              help="Min speech segment duration in ms")
@click.option("--min-silence", type=int, default=300,
              help="Min silence duration to split segments in ms")
def main(
    wav_file: str,
    output: str | None,
    threshold: float | None,
    speaker_a: str,
    speaker_b: str,
    frame_size: int,
    min_speech: int,
    min_silence: int,
) -> None:
    """Analyze a stereo WAV recording of a two-person conversation."""
    wav_path = Path(wav_file)

    if output is None:
        output = str(wav_path.with_suffix("")) + "_report.html"

    click.echo(f"Loading {wav_path.name}...")
    sample_rate, left, right = load_wav(wav_path)
    duration_sec = len(left) / sample_rate

    click.echo(f"  Sample rate: {sample_rate} Hz")
    click.echo(f"  Duration: {duration_sec:.1f}s")

    click.echo("Running voice activity detection...")
    segments_a = detect_speech(
        left, sample_rate,
        frame_ms=frame_size, threshold=threshold,
        min_speech_ms=min_speech, min_silence_ms=min_silence,
    )
    segments_b = detect_speech(
        right, sample_rate,
        frame_ms=frame_size, threshold=threshold,
        min_speech_ms=min_speech, min_silence_ms=min_silence,
    )

    click.echo(f"  {speaker_a}: {len(segments_a)} speech segments")
    click.echo(f"  {speaker_b}: {len(segments_b)} speech segments")

    click.echo("Computing statistics...")
    stats = compute_stats(segments_a, segments_b, duration_sec, speaker_a, speaker_b)

    click.echo("Generating report...")
    generate_report(stats, output)

    click.echo(f"Report written to {output}")
