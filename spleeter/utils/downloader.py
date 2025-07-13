import os

from typing import Any

import yt_dlp


class Downloader:
    def __init__(self, output_dir: str, media_type: str, quality: str):
        self.output_dir = output_dir
        self.media_type = media_type
        self.quality = quality
        self._initialize_youtube_dl()

    def download(self, url: str) -> str:
        self.youtube_dl.download(url)
        url_data = self.youtube_dl.extract_info(url, download=False)

        self._initialize_youtube_dl()

        filename = f"{url_data['id']}.mp4" if self.media_type == "video" else f"{url_data['id']}.m4a"
        download_path = os.path.abspath(os.path.join(self.output_dir, filename))
        return download_path

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
        if self.quality == "medium":
            config['format'] =  'best[height<=720]' if self.media_type == 'video' else 'best[abr<=160]'

        if self.quality == "low":
            config['format'] =  'best[height<=360]' if self.media_type == 'video' else 'best[abr<=96]'

        return config

