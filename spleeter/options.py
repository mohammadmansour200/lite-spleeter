#!/usr/bin/env python
# coding: utf8

"""This modules provides spleeter command as well as CLI parsing methods."""

from os.path import join
from tempfile import gettempdir

from typer import Argument, Exit, Option, echo
from typer.models import List, Optional

from .audio import Codec

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

AudioInputArgument: List[str] = Argument(
    ...,
    help="List of input audio file path",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
)

AudioInputOption: Optional[str] = Option(
    None, "--inputs", "-i", help="(DEPRECATED) placeholder for deprecated input option"
)

AudioAdapterOption: str = Option(
    "spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter",
    "--adapter",
    "-a",
    help="Name of the audio adapter to use for audio I/O",
)

AudioOutputOption: str = Option(
    join(gettempdir(), "separated_audio"),
    "--output_path",
    "-o",
    help="Path of the output directory to write audio files in",
)


AudioCodecOption: Codec = Option(
    Codec.MP3, "--codec", "-c", help="Audio codec to be used for the separated output"
)

AudioBitrateOption: str = Option(
    "128k", "--bitrate", "-b", help="Audio bitrate to be used for the separated output"
)


MWFOption: bool = Option(
    False, "--mwf", help="Whether to use multichannel Wiener filtering for separation"
)


VerboseOption: bool = Option(False, "--verbose", help="Enable verbose logs")


def version_callback(value: bool):
    if value:
        from importlib.metadata import version

        echo(f"lite-spleeter Version: {version('lite-spleeter')}")
        raise Exit()


VersionOption: bool = Option(
    None,
    "--version",
    callback=version_callback,
    is_eager=True,
    help="Return lite-spleeter version",
)
