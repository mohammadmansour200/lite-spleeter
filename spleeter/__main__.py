#!/usr/bin/env python
# coding: utf8

"""
Python oneliner script usage.

USAGE: python -m spleeter separate ...

Notes:
    All critical import involving TF, numpy or Pandas are deported to
    command function scope to avoid heavy import on CLI evaluation,
    leading to large bootstraping time.
"""

from typing import List, Optional

# pyright: reportMissingImports=false
# pylint: disable=import-error
from typer import Exit, Typer

from . import SpleeterError
from .audio import Codec
from .options import (
    AudioAdapterOption,
    AudioBitrateOption,
    AudioCodecOption,
    AudioInputArgument,
    AudioInputOption,
    AudioOutputOption,
    MWFOption,
    VerboseOption,
    VersionOption,
)
from .utils.logging import configure_logger, logger

# pylint: enable=import-error

spleeter: Typer = Typer(add_completion=False, no_args_is_help=True, short_help="-h")
""" CLI application. """


@spleeter.callback()
def default(
    version: bool = VersionOption,
) -> None:
    pass


@spleeter.command(no_args_is_help=True)
def separate(
    deprecated_files: Optional[str] = AudioInputOption,
    files: List[str] = AudioInputArgument,
    adapter: str = AudioAdapterOption,
    bitrate: str = AudioBitrateOption,
    codec: Codec = AudioCodecOption,
    output_path: str = AudioOutputOption,
    mwf: bool = MWFOption,
    verbose: bool = VerboseOption,
) -> None:
    """
    Separate audio file(s)
    """
    from .audio.adapter import AudioAdapter
    from .separator import Separator

    configure_logger(verbose)
    if deprecated_files is not None:
        logger.error(
            "⚠️ -i option is not supported anymore, audio files must be supplied "
            "using input argument instead (see spleeter separate --help)"
        )
        raise Exit(20)
    audio_adapter: AudioAdapter = AudioAdapter.get(adapter)
    separator: Separator = Separator(MWF=mwf)

    for filename in files:
        separator.separate_to_file(
            filename,
            output_path,
            audio_adapter=audio_adapter,
            codec=codec,
            bitrate=bitrate,
            synchronous=False,
        )
    separator.join()


def entrypoint():
    """Application entrypoint."""
    try:
        spleeter()
    except SpleeterError as e:
        logger.error(e)


if __name__ == "__main__":
    entrypoint()
