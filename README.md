# `lite-spleeter`: Lightweight musical instruments removing library

## About
This is a lightweight [Spleeter](https://github.com/deezer/spleeter) (a library that uses AI/ML to extract vocals from audio files) version that works with media files of any length, with very low and fixed RAM usage (around 1.5 GB RAM usage on my computer).

## Features
• Progress indicator

• Very lightweight on low-end Computers

• Smaller package and simpler

• No numpy errors

• Asynchronous processing for better performance

• Efficient memory management

## How does it work?
It processes 30-second segments sequentially using asynchronous processing. After processing all segments, it concatenates them. This approach of processing chunks instead of the whole audio file helps keep memory usage low while maintaining good performance.

## Get started
*Make sure you have [ffmpeg](https://www.ffmpeg.org/download.html) installed.*

Download package:
```bash
pip install lite-spleeter
```

## Usage

### Basic Usage
# Extract vocals only (default)
```bash
lite-spleeter separate audio_example.mp3 -o audio_output_path
```

# Extract both vocals and accompaniment
```bash
lite-spleeter separate audio_example.mp3 -o audio_output_path --all
```

The `--all` flag determines what stems are extracted:
- Without `--all`: Only extracts vocals (default)
- With `--all`: Extracts both vocals and accompaniment

### Batch Processing
`separate` command builds the model each time it is called. If you have several files to separate, it is recommended to process them in a single call:

# Extract vocals only from multiple files
```bash
lite-spleeter separate <path/to/audio1.mp3> <path/to/audio2.wav> <path/to/audio3.ogg> -o audio_output_path
```

# Extract both vocals and accompaniment from multiple files
```bash
lite-spleeter separate <path/to/audio1.mp3> <path/to/audio2.wav> <path/to/audio3.ogg> -o audio_output_path --all
```

### Help
To get help on the different options available with the separate command, type:
```bash
lite-spleeter separate --help
```

Read the original [Spleeter repo](https://github.com/deezer/spleeter) for more info.