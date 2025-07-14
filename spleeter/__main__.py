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
import os.path
from typing import List, Optional

import validators
# pyright: reportMissingImports=false
# pylint: disable=import-error
from typer import Exit, Typer

from spleeter.utils.downloader import Downloader
from . import SpleeterError
from .options import (
    AudioAdapterOption,
    AudioBitrateOption,
    AudioInputArgument,
    AudioInputOption,
    AudioOutputOption,
    MWFOption,
    VerboseOption,
    VersionOption, MediaTypeOption, QualityOption,
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
    output_path: str = AudioOutputOption,
    mwf: bool = MWFOption,
    verbose: bool = VerboseOption,
    media_type: str = MediaTypeOption,
    quality: str = QualityOption
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
        is_url = validators.url(filename)
        audio_descriptor = filename
        if is_url:
            if media_type is None:
                raise SpleeterError("No media type is specified, it can either be 'audio' or 'video'")
            downloaded_output_path = os.path.join(output_path, 'tmp')
            downloader = Downloader(
                                    output_dir=downloaded_output_path,
                                    media_type=media_type,
                                    quality=quality
                                    )
            downloaded_file_name = downloader.download(url=filename)
            audio_descriptor = os.path.abspath(os.path.join(downloaded_output_path, downloaded_file_name))

        separator.separate_to_file(audio_descriptor=audio_descriptor, destination=output_path, audio_adapter=audio_adapter, bitrate=bitrate,
                                   synchronous=True)
    separator.join()


def entrypoint():
    """Application entrypoint."""
    try:
        spleeter()
    except SpleeterError as e:
        logger.error(e)


if __name__ == "__main__":
    entrypoint()
