`lite-spleeter`: Lightweight musical instruments removing library

## About
This is a lightweight [Spleeter](https://github.com/deezer/spleeter) (a library that uses AI/ML to extract vocals from audio files) version that works with media files of any length, with very low and fixed RAM usage (around 1.5 GB RAM usage on my computer).

## Features
• Progress indicator

• Very lightweight on low-end Computers

• Smaller package and simpler

• No numpy errors

## How does it work?
It processes 30-second segments sequentially. After processing all segments, it concatenates them. This approach of processing chunks instead of the whole audio file helps keep memory usage low.

## Get started
*Make sure you have [ffmpeg](https://www.ffmpeg.org/download.html) installed.*

Download package:
```bash
pip install lite-spleeter
```



## Usage
```bash
lite-spleeter separate audio_example.mp3  -o audio_output_path
```

You can provide either a single or a list of files for batch processing

##### Batch processing
`separate` command builds the model each time it is called, this process may be long, in this case If you have several files to separate, it is then recommended to perform all separation with a single call to separate:
```bash
lite-spleeter separate <path/to/audio1.mp3> <path/to/audio2.wav> <path/to/audio3.ogg> -o audio_output_path
```


To get help on the different options available with the separate command, type:
```bash
lite-spleeter separate --help
```

Read the original [Spleeter repo](https://github.com/deezer/spleeter) for more info.