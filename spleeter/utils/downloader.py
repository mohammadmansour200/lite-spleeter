import os

from typing import Any

import yt_dlp


class Downloader:
    def __init__(self, output_dir: str, media_type: str, quality: str):
        self.output_dir = output_dir
        self.media_type = media_type
        self.quality = quality
        self.is_output_video = self.media_type == "video"
        self._initialize_youtube_dl()

    def download(self, url: str) -> str:
        self.youtube_dl.download(url)
        url_data = self.youtube_dl.extract_info(url, download=False)

        self._initialize_youtube_dl()

        filename = f"{url_data['id']}.{url_data['ext']}" if self.is_output_video else f"{url_data['id']}.mp3"
        return filename

    def _initialize_youtube_dl(self) -> None:
        self.youtube_dl = yt_dlp.YoutubeDL(self._config())

    def _config(self) -> dict[str, Any]:
        config = {
            'ignoreerrors': True,
            'noplaylist':True,
            'outtmpl': os.path.join(self.output_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'verbose': False,
        }

        is_high_quality = self.quality == "high"

        if not self.is_output_video and is_high_quality:
            config['extract_audio'] = True
            config['postprocessors'] = [
                {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }
            ]

        if self.is_output_video and not is_high_quality:
            config['format'] = 'bestvideo[height<=720]+bestaudio' if self.quality == "medium" else 'bestvideo[height<=360]+bestaudio'

        if not self.is_output_video and not is_high_quality:
            config['format'] = 'bestaudio'
            config['extract_audio'] = True
            config['postprocessors']= [
                {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '128' if self.quality == "medium" else '96',
                }
            ]

        return config

