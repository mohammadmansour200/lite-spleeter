`lite-spleeter`: Lightweight instruments removing library

## About

This is a lite spleeter version that works with media files of any length, with very low and fixed RAM usage (around 1.5 GB RAM usage on my Computer), removed the test and evaluation command, made the package limited and hardcoded to the 2stems seperation flavor (Vocals / accompaniment separation) for simplicity, furthermore, I used tensorflow-cpu for a smaller package; besides that everyting remains as is.

## Features

• Progress indicator
• Very lightweight on low-end Computers
• Smaller package and simpler
• No numpy errors

## How does it work?

I limited it to processing 30 seconds only one after another, then it concatenates those splitted parts.

## Get started
*Make sure you have ffmpeg installed.*

Download package:
```bash
pip install lite-spleeter
```

Read the original [Spleeter repo](https://github.com/deezer/spleeter) for more info.

## Example usage
```bash
lite-spleeter separate <input file path> -o <output path>
```
