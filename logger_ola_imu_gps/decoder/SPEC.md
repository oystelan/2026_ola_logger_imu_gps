# Specification of the data format on the SD card

The information below is from the PlatformIO C++ project in the sibling directory: see `../logger_ola_imu_gps` for more details about the code running on the logger. The data files are pre allocated to be a total of `10 * 1024 * 1024` bytes, and may contain empty data or trash data at the end of the file.

The default logging rates are:
  - ISM330DHCX: around 437 Hz
  - GNSS position: around 10 Hz
  - PPS: around 1 Hz

The default sd data file duration is 15 minutes, and new files are started by the logger at UTC minutes 00, 15, 30, 45. If a file is started at boot, it may start at a different minute than these, but will end at the next such UTC minute.

Filename format: this is generated in C++:

```cpp
    snprintf(filename_buffer, sizeof(filename_buffer),
             "DATA_BOOT_%04u_TIME_%02u%02u%02uT%02u%02u%02u.dat",
             boot_count,
             common_working_struct_YMDHMS.year,
             common_working_struct_YMDHMS.month,
             common_working_struct_YMDHMS.day,
             common_working_struct_YMDHMS.hour,
             common_working_struct_YMDHMS.minute,
             common_working_struct_YMDHMS.second);
```

Data file format:

- header: (the actual string and numerical values may change depending on compile, options etc): taken from an example file:

```
Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: 391b428a3e869543ebd2caf1626f845730858f8b
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10
```

The header contains the scaling from int value to physical units for the accelerometer and the gyroscope. For the Acc (accelerometer), mg means "milli-g", g is the acceleration of gravity. For the Gyr (gyroscope), the mdps means "milli degree per second".

For the GPS, the units are as follow:
  - latitude and longitude: these are the int rounding of "degree decimal * 10^7"
  - for the vel (velocity) variables: these are the in mmps (millimeters per second) per LSB
  - the posix_timestamp field has unit seconds, and it can be combined with the microseconds value to generate a timestamp with microsecond accuracy

- actual data entries format:
    - 1 entry per line: each line is a single record of data
    - first 4 chars that indicate the kind of the line are set in C++ as:

      - for the PPS entries:

```cpp
        entry_kind[0] = '\n';
        entry_kind[1] = 'P';
        entry_kind[2] = 'P';
        entry_kind[3] = 'S';
```

      - for the GPS entries:

```cpp
        entry_kind[0] = '\n';
        entry_kind[1] = 'G';
        entry_kind[2] = 'P';
        entry_kind[3] = 'S';
```

      - for the IMU entries:

```cpp
        entry_kind[0] = '\n';
        entry_kind[1] = 'I';
        entry_kind[2] = 'M';
        entry_kind[3] = 'U';
```

    - then a raw dump of the C++ respective struct (unsigned long is 32 bits) follows as (in all the following, micros_reading is the microcontroller arduino core `micros()` output):

      - for the PPS entries:

```cpp
struct PPS_fix {
  unsigned long micros_reading;
};
```

      - for the GNSS entries:

```cpp
struct GNSS_reading {
  unsigned long micros_reading;
  int32_t latitude;
  int32_t longitude;
  uint32_t posix_timestamp;
  uint32_t microseconds;
  int32_t NED_vel_north;
  int32_t NED_vel_east;
  int32_t NED_vel_down;
  uint8_t fix_type;
};
```

      - for the IMU entries:

```cpp
struct IMU_reading{
  unsigned long micros_reading;
  uint16_t counter;
  int16_t acc_x;
  int16_t acc_y;
  int16_t acc_z;
  int16_t gyr_x;
  int16_t gyr_y;
  int16_t gyr_z;
};  
```

Note: The IMU struct has 2 bytes of padding added by the C compiler for alignment, making the actual on-disk size 20 bytes (18 bytes struct + 2 bytes padding) before the newline marker.

For the IMU entries, the scaling from the `int16_t` to actual float values are to be performed using the sensitivity values provided in the header.

- Footer:

```
\n\nLog stop OLA ISM330DHCX SAM-M10Q logger\n
```

- Finally the file is pre-allocated for speed, so the rest of the file after the footer may be either all `\0` or even garbage, depending on how the SD card had been formatted or not beforehands.
