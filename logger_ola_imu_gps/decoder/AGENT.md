# Python decoder for the raw data files

## Overview

- the goal is to decode the raw data files generated from the OLA logger using python code
- see the spec in the `./SPEC.md` file; make sure to read it
- the OLA (OpenLogArtemis) logger runs a C/C++ bare metal firmware are logs to SD card
- the OLA logger logs an IMU (ISM330DHCX) and a GNSS (SAM-M10Q GPS, both logging some navigation data and the PPS 1Hz rising edge)
- in addition, the decoder combines the PPS and GPS data to generate a mapping from microseconds to UTC datetime. For this:
    - since the `micros()` uses `uint32_t` on the OLA MCU, the parser checks for possible wrapping: going through the ordered list of `micros_reading` for each kind of data, if a jump by more than `2**32 / 2` is detected, a "dealiasing additional offset" is increased by an additional `2**32` for all following `micros_reading` values
    - the PPS and GNSS data are cross compared; for each PPS `micros_reading`, the closest GPS `micros_reading` is found and the associated `posix_timestamp + microseconds/1e6` is used to determine which second start the PPS entry corresponds to; this is used to generate a dataset of "unwrapped micros" from the PPS data vs "matching UTC second" from the GNSS data; a linear regression is then established to go from `micros_reading` to "UTC posix timestamp"; to avoid numerical inaccuracies, the offset (lowest micros observed within the file) is subtracted from the micros MCU timestamps data in all linear regression tasks
    - the linear regression is used to generate a `utc_timestamp_from_pps_regression` variable that is added to the dataclass of each kind of data, applying the linear regression to the `micros_reading` for each data entry

## Code and data organization

- the `./decoder.py` module contains all the utilities to easily decode a single data file at a time
- several example data files are available (e.g., `DATA_BOOT_0000_TIME_20260204T193000.dat`)
- the `./example_decode.py` script contains an example of how to decode the data file using the decoder functions
- the `./simple_example.py` script is the **recommended starting point** for end users - shows clean, simple workflow
- the `./test_decoder.py` module contains the tests to run
- write and keep up to date a README.md file to explain how to install the mamba env, run the code, etc.
- as you need more packages, make sure these are all registered in the mamba environment config file

## User-Facing API Design

The decoder provides two equivalent workflows for end users:

### Workflow 1: CLI Tool (for quick inspection)
```bash
python decoder_cli.py -p DATA_FILE.dat
```
- Decodes file and creates .npz
- Shows 8 comprehensive plots
- Good for quick data inspection

### Workflow 2: Python API (recommended for analysis)
```python
from decoder import decode_file, load_data_as_arrays

# Step 1: Decode
result = decode_file(Path("data.dat"))

# Step 2: Load as numpy arrays (EASIEST)
data = load_data_as_arrays(result['file'])

# Step 3: Use the data
plot(data['imu_utc'], data['imu_acc_x'])
```

**Key principle:** Both workflows produce **identical .npz files**. The CLI is for quick inspection, the Python API is for custom analysis.

### API Functions for End Users

1. **`decode_file(path)`** - Decode binary .dat file → returns dict with 'file' (npz path)
2. **`load_data_as_arrays(npz_path)`** - Load .npz → returns dict of numpy arrays (RECOMMENDED)
3. **`load_and_combine_segments(npz_path)`** - Load .npz → returns dict of dataclass arrays (advanced)

The recommended function is `load_data_as_arrays()` which provides:
- All sensor data as numpy arrays (ready for matplotlib, scipy, etc.)
- Clean naming: `imu_acc_x`, `gnss_latitude`, `pps_utc`, etc.
- Outlier flags: `imu_acc_x_outlier`, `gnss_latitude_outlier`, etc.
- Header info as scalars: `imu_odr`, `firmware_commit`, etc.

**Design goals:**
- Simple and obvious for beginners
- Follows standard numpy/scipy conventions
- Minimal boilerplate code
- Works identically whether using CLI or Python API

## Code practices

Everything for parsing etc is written in python

- have tests to run with pytest
- have logging with loguru
- write simple, idiomatic python code
- use typing / type annotation for the functions
- use formal checkers for the code (install these in the mamba environment):
  - check spell in the code with `pylint`
  - check and lint the code with `ruff` + `pylint` + `flake8`
  - check code complexity with `complexipy`
  - generally, double check your code for good style, ease of understanding, good API / interface, safety and speed
- run test after each code change
- use established packages:
  - struct and struct.unpack for raw data handling (reading int16_t, long unsigned int, etc; long unsigned int is 32 bits)
  - dataclasses to model C/C++ structs
  - numpy for the arrays and dump the data as npy; use the type "object" for the arrays to contain the dataclasses objects; enrich the dataclass with "scaled" values for the IMU (float scaled values by the sensors sensitivity in addition to raw int16_t readings)
  - scipy for linear regression
  - gnuplotlib for asicc plots in terminal
  - matplotlib for interactive plots
  - click to build cli tools
  - no more package imports needed
- code defensively: user asserts to document and check all assumptions about incoming data, especially line lengths / raw binary data size; if some assert fail, log an error with enough details to understand what happened and exit with an error (raise an exception)
- make sure that the asserts are really useful and not tautological - asserts should really check that the data match what is expected, i.e. check that the input from the "messy real world" (in particular, sd card files written by the logger) match the assumptions about the structure of the file and content
- if you are in doubt, or anything is unclear or ambiguous or badly explained or defined, do not guess - ask me (the user) for more information
- if you can see several ways to implement the same thing, and you are unsure which one to choose, feel free to ask me (the user) for more information / chat together

## Code setup and environment management: mamba

- use mamba; it should be pre installed on the system; do not install it yourself, do not add channels; fail if no mamba and ask user help
- use an environment.yml to define the necessary packages
- use, or create if necessary, a dedicated environment for working on this, with name: "ola_ism330dhcx_samm10q_decoder"; use only this env for running code here
- use only conda-forge channel (this should already be set by the mamba install available)

## Code architecture and conventions for the decoder

- magic constants should be ALL_CAPS_CONSTANTS defined at the start to make it easy to edit
- avoid object oriented if not necessary, make the code based on simple functions
- pass the ALL_CAPS_CONSTANTS as default args, it is ok to have many default args to the functions
- use individual functions to parse each raw data kind (i.e. parse 1 line of raw data)
- write a single function to parse a single file and write the data to disk
- when dumping to disk, save as a single compressed .npz file containing numpy arrays for each data kind (pps, gnss, imu), with each array containing dataclass objects matching the object kind
- after parsing, summarize the information: both file duration based on the micros information (take the first and last micros from IMU measurements), number of messages of each kind parsed, corresponding effective frequency

## Misc

- at the end of each update to the code, go through the files and check that the documentation and readme are up to date
- when writing code, check that all guidelines present here are followed
- at the end of each code update, do a small review of the code, look for good possibilities for refactoring and improvement, and if there are clear improvement possibilities, work on implementing them
- at the end of each code update, make sure everything works - tests, spell check, linting, etc

