#!/usr/bin/env python
# coding: utf8

"""
Module that provides a class wrapper for source separation.

Examples:

```python
>>> from spleeter.separator import Separator
>>> separator = Separator()
>>> separator.separate(waveform, lambda instrument, data: ...)
>>> separator.separate_to_file(...)
```
"""

import atexit
import os
from multiprocessing import Pool
from os.path import dirname, basename, join, exists, splitext
import shutil
from typing import Any, Dict, Generator, List, Optional

# pyright: reportMissingImports=false
# pylint: disable=import-error
import numpy as np
import tensorflow as tf

from .audio.ffmpeg import get_audio_duration, merge_media_files

from . import SpleeterError
from .audio import Codec
from .audio.adapter import AudioAdapter
from .audio.convertor import to_stereo
from .model import EstimatorSpecBuilder, InputProviderFactory, model_fn
from .types import AudioDescriptor
from .utils.configuration import load_configuration

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


class DataGenerator(object):
    """
    Generator object that store a sample and generate it once while called.
    Used to feed a tensorflow estimator without knowing the whole data at
    build time.
    """

    def __init__(self) -> None:
        """Default constructor."""
        self._current_data = None

    def update_data(self, data) -> None:
        """Replace internal data."""
        self._current_data = data

    def __call__(self) -> Generator:
        """Generation process."""
        buffer = self._current_data
        while buffer:
            yield buffer
            buffer = self._current_data


def create_estimator(params: Dict, MWF: bool) -> tf.Tensor:
    """
    Initialize tensorflow estimator that will perform separation

    Parameters:
        params (Dict):
            A dictionary of parameters for building the model
        MWF (bool):
            Wiener filter enabled?

    Returns:
        tf.Tensor:
            A tensorflow estimator
    """
    # Load model.
    package_dirname = dirname(__file__)
    params["model_dir"] = join(package_dirname, "pretrained_models", "2stems")

    params["MWF"] = MWF

    # Setup config
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config = tf.estimator.RunConfig(session_config=session_config)

    # Setup estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=params["model_dir"], params=params, config=config
    )
    return estimator


class Separator(object):
    """A wrapper class for performing separation."""

    def __init__(
        self,
        MWF: bool = False,
        multiprocess: bool = True,
    ) -> None:
        """
        Default constructor.

        Parameters:
            MWF (bool):
                (Optional) `True` if MWF should be used, `False` otherwise.
            multiprocess (bool):
                (Optional) Enable multi-processing.
        """
        self._params = load_configuration()
        self._sample_rate = self._params["sample_rate"]
        self._MWF = MWF
        self._estimator = create_estimator(self._params, self._MWF)
        self._tf_graph = tf.Graph()
        self._prediction_generator: Optional[Generator] = None
        self._input_provider = None
        self._builder = None
        self._features = None
        self._session = None
        if multiprocess:
            self._pool: Optional[Any] = Pool()
            atexit.register(self._pool.close)
        else:
            self._pool = None
        self._tasks: List = []
        self._data_generator = DataGenerator()

    def _get_prediction_generator(self) -> Generator:
        """
        Lazy loading access method for internal prediction generator
        returned by the predict method of a tensorflow estimator.

        Returns:
            Generator:
                Generator of prediction.
        """
        if self._prediction_generator is None:

            def get_dataset():
                return tf.data.Dataset.from_generator(
                    self._data_generator,
                    output_types={"waveform": tf.float32, "audio_id": tf.string},
                    output_shapes={"waveform": (None, 2), "audio_id": ()},
                )

            self._prediction_generator = self._estimator.predict(
                get_dataset, yield_single_examples=False
            )
        return self._prediction_generator

    def join(self, timeout: int = 200) -> None:
        """
        Wait for all pending tasks to be finished.

        Parameters:
            timeout (int):
                (Optional) Task waiting timeout.
        """
        while len(self._tasks) > 0:
            task = self._tasks.pop()
            task.get()
            task.wait(timeout=timeout)

    def _get_input_provider(self):
        if self._input_provider is None:
            self._input_provider = InputProviderFactory.get(self._params)
        return self._input_provider

    def _get_features(self):
        if self._features is None:
            provider = self._get_input_provider()
            self._features = provider.get_input_dict_placeholders()
        return self._features

    def _get_builder(self):
        if self._builder is None:
            self._builder = EstimatorSpecBuilder(self._get_features(), self._params)
        return self._builder

    def _get_session(self):
        if self._session is None:
            saver = tf.compat.v1.train.Saver()
            model_directory: str = self._params["model_dir"]
            latest_checkpoint = tf.train.latest_checkpoint(model_directory)
            self._session = tf.compat.v1.Session()
            saver.restore(self._session, latest_checkpoint)
        return self._session

    def _separate_tensorflow(
        self, waveform: np.ndarray, audio_descriptor: AudioDescriptor
    ) -> Dict:
        """
        Performs source separation over the given waveform with tensorflow
        backend.

        Parameters:
            waveform (np.ndarray):
                Waveform to be separated (as a numpy array)
            audio_descriptor (AudioDescriptor):
                Audio descriptor to be used.

        Returns:
            Dict:
                Separated waveforms.
        """
        if not waveform.shape[-1] == 2:
            waveform = to_stereo(waveform)
        prediction_generator = self._get_prediction_generator()
        # NOTE: update data in generator before performing separation.
        self._data_generator.update_data(
            {"waveform": waveform, "audio_id": np.array(audio_descriptor)}
        )
        # NOTE: perform separation.
        prediction = next(prediction_generator)
        prediction.pop("audio_id")
        return prediction

    def separate(
        self,
        waveform: np.ndarray,
        audio_descriptor: Optional[str] = "",
    ) -> Dict:
        """
        Performs separation on a waveform.

        Parameters:
            waveform (np.ndarray):
                Waveform to be separated (as a numpy array)
            audio_descriptor (Optional[str]):
                (Optional) string describing the waveform (e.g. filename).

        Returns:
            Dict:
                Separated waveforms.
        """
        return self._separate_tensorflow(waveform, audio_descriptor)

    def separate_to_file(
        self,
        audio_descriptor: AudioDescriptor,
        destination: str,
        audio_adapter: Optional[AudioAdapter] = None,
        codec: Codec = Codec.MP3,
        bitrate: str = "128k",
        synchronous: bool = True,
        all: bool = False,
    ) -> None:
        """
        Performs source separation and export result to file using
        given audio adapter.

        Filename format should be a Python formattable string that could
        use following parameters :

        - {instrument}
        - {filename}
        - {foldername}
        - {codec}.

        Parameters:
            audio_descriptor (AudioDescriptor):
                Describe song to separate, used by audio adapter to
                retrieve and load audio data, in case of file based
                audio adapter, such descriptor would be a file path.
            destination (str):
                Target directory to write output to.
            audio_adapter (AudioAdapter):
                (Optional) Audio adapter to use for I/O.
            codec (Codec):
                (Optional) Export codec.
            bitrate (str):
                (Optional) Export bitrate.
            synchronous (bool):
                (Optional) True is should by synchronous.
            all (bool):
                (Optional) If True, output both vocals and accompaniment.
                If False, only output vocals. Default is False.
        """

        if audio_adapter is None:
            audio_adapter = AudioAdapter.default()

        total_duration = get_audio_duration(audio_descriptor)

        root, _ = splitext(basename(audio_descriptor))

        # Segemented processing should only be for audio files over 30 seconds.
        if total_duration > 30:
            segment_files = []
            segment_duration = 30
            temp_folder_path = os.path.abspath(os.path.join(destination, "tmp"))
            if not exists(temp_folder_path):
                os.makedirs(temp_folder_path)

            # Calculate total segments
            total_segments = int(np.ceil(total_duration / segment_duration))
            instruments_to_process = None

            for segment_index in range(total_segments):
                segment_dir = os.path.join(temp_folder_path, f"segment_{segment_index}")
                if not exists(segment_dir):
                    os.makedirs(segment_dir)

                offset = segment_index * segment_duration
                duration_to_process = min(segment_duration, total_duration - offset)

                try:
                    # Load audio segment
                    waveform, _ = audio_adapter.load(
                        audio_descriptor,
                        offset=offset,
                        duration=duration_to_process,
                        sample_rate=self._sample_rate,
                    )

                    # 分离音频
                    sources = self.separate(waveform, audio_descriptor)
                    if not all:
                        sources.pop("accompaniment", None)

                    # 在第一次循环时确定需要处理的乐器
                    if instruments_to_process is None:
                        instruments_to_process = list(sources.keys())

                    # 为每个乐器创建单独的任务
                    for instrument, data in sources.items():
                        output_path = os.path.join(segment_dir, f"{instrument}.{codec}")
                        if self._pool:
                            # 只传递必要的数据
                            task = self._pool.apply_async(
                                audio_adapter.save,
                                (output_path, data, self._sample_rate, codec, bitrate)
                            )
                            self._tasks.append(task)
                        else:
                            audio_adapter.save(output_path, data, self._sample_rate, codec, bitrate)

                        segment_path = os.path.join(f"segment_{segment_index}", f"{instrument}.{codec}")
                        segment_files.append(segment_path)

                    # 计算并打印进度
                    progress = ((segment_index + 1) / total_segments) * 100
                    print(f"Processing: {progress:.2f}% complete")

                except Exception as e:
                    print(f"Error processing segment {segment_index}: {str(e)}")
                    if exists(temp_folder_path):
                        shutil.rmtree(temp_folder_path)
                    raise

            # 等待所有保存任务完成
            if self._tasks:
                for task in self._tasks:
                    try:
                        task.get()
                    except Exception as e:
                        print(f"Error in save task: {str(e)}")
                        if exists(temp_folder_path):
                            shutil.rmtree(temp_folder_path)
                        raise

            # 合并文件
            try:
                for instrument in instruments_to_process:
                    segment_paths = [f for f in segment_files if instrument in f]
                    merge_media_files(
                        root,
                        segment_paths,
                        temp_folder_path,
                        codec,
                        instrument
                    )

                    # 移动合并后的文件到目标目录
                    src = os.path.join(temp_folder_path, f"{root}_{instrument}.{codec}")
                    dst = os.path.join(destination, f"{root}_{instrument}.{codec}")
                    if os.path.exists(src):
                        shutil.move(src, dst)

            except Exception as e:
                print(f"Error during merging: {str(e)}")
                if exists(temp_folder_path):
                    shutil.rmtree(temp_folder_path)
                raise

            # 清理临时文件
            if exists(temp_folder_path):
                shutil.rmtree(temp_folder_path)

        else:
            waveform, _ = audio_adapter.load(
                audio_descriptor,
                sample_rate=self._sample_rate,
            )
            sources = self.separate(waveform, audio_descriptor)
            if not all:
                sources.pop("accompaniment", None)  # 使用 pop 的第二个参数避免 KeyError
            self.save_to_file(
                sources,
                destination,
                codec,
                audio_adapter,
                bitrate,
                synchronous,
                root,
            )

        print("File created successfuly")

    def save_to_file(
        self,
        sources: Dict,
        destination: str,
        codec: Codec = Codec.MP3,
        audio_adapter: Optional[AudioAdapter] = None,
        bitrate: str = "128k",
        synchronous: bool = True,
        filename: str = None,
    ) -> None:
        """
        Export dictionary of sources to files.

        Parameters:
            sources (Dict):
                Dictionary of sources to be exported. The keys are the name
                of the instruments, and the values are `N x 2` numpy arrays
                containing the corresponding intrument waveform, as
                returned by the separate method
            destination (str):
                Target directory to write output to.
            codec (Codec):
                (Optional) Export codec.
            audio_adapter (Optional[AudioAdapter]):
                (Optional) Audio adapter to use for I/O.
            bitrate (str):
                (Optional) Export bitrate.
            synchronous (bool):
                (Optional) True is should by synchronous.
        """
        if audio_adapter is None:
            audio_adapter = AudioAdapter.default()

        generated = []
        for instrument, data in sources.items():
            if filename:
                path = join(destination, f"{filename}_{instrument}.{codec}")
            else:
                path = join(destination, f"{instrument}.{codec}")

            directory = dirname(path)
            if not exists(directory):
                os.makedirs(directory)
            
            if path in generated:
                raise SpleeterError(f"Separated source path conflict: {path}")
            
            generated.append(path)
            if self._pool:
                task = self._pool.apply_async(
                    audio_adapter.save, (path, data, self._sample_rate, codec, bitrate)
                )
                self._tasks.append(task)
            else:
                audio_adapter.save(path, data, self._sample_rate, codec, bitrate)
            
        if synchronous and self._pool:
            self.join()
