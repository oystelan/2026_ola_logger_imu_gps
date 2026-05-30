#ifndef TIME_MANAGER
#define TIME_MANAGER

#include "Arduino.h"

#include "kiss_posix_time_utils.hpp"
#include "kiss_posix_time_extras.hpp"

#include "firmware_configuration.h"
#include "user_configuration.h"
#include "print_utils.h"

#include "TimeLib.h"

// a class for managing time
// this is some wrappers around kiss_posix_time and the RTC HAL that allow to use the RTC
// to keep track of posix time and convert back and forth between posix time, calendar
// time.
class TimeManager{
  public:
    TimeManager();

    // set the posix timestamp to the given value in the TimeManager
    // this will allow the TimeManager to keep track of time, relatively
    // to this initial set value
    void set_posix_timestamp(kiss_time_t const crrt_posix_timestamp);

    // get the current posix timestamp from the time manager
    // note that this is only valid if posix_timestamp_is_valid!!
    kiss_time_t get_posix_timestamp(void) const;

    // whether the current posix timestamp provided by TimeManager is valid
    // it is valid if it has been set at least once
    // if not the timestamp is valid, it will actually count time since the
    // start of the board, relative to the unix epoch
    bool posix_timestamp_is_valid(void) const;

    // perform some serial print of the information in the TimeManager
    // will display posix and calendar time, as well as status
    void print_status(void) const;

    // set the low level RTC properties through the HAL to allow it to count
    // seconds in an interrupt-driven way
    void setup_RTC(void);

    // Read the battery-backed Apollo3 hardware RTC date/time registers and
    // convert to a POSIX timestamp. Returns 0 if the H/W RTC value is
    // implausible (year < 2025) — i.e. it's the post-power-loss default,
    // not a value we previously wrote. With a coin cell on VBAT the H/W
    // RTC keeps counting through power-off; this is how we recover that
    // counter on the next boot.
    kiss_time_t read_hw_rtc_posix(void) const;

    // Write the given POSIX timestamp into the H/W RTC date/time registers.
    // Call this whenever we receive an authoritative time source (e.g. GNSS
    // UTC) so the H/W RTC remains battery-backed and accurate across power
    // cycles. Does NOT update the software `posix_timestamp` counter — caller
    // should also call set_posix_timestamp().
    void write_hw_rtc_posix(kiss_time_t crrt_posix_timestamp);

    // Minimum POSIX-derived year considered "plausibly initialised" by
    // read_hw_rtc_posix(). Used to distinguish a battery-preserved RTC value
    // from the chip's post-power-loss default (often 2000-01-01).
    static constexpr int RTC_SANITY_YEAR_MIN = 2025;

  private:
    // have we ever set a posix timestamp?
    bool posix_is_set;
};

// isr for the rtc, used to do interrupt based seconds counting
extern "C" void arm_rtc_isr(void);

// we use one single TimeManager instance for the board
extern TimeManager board_time_manager;

extern "C" void arm_rtc_isr(void);

struct struct_YMDHMS{
  int year;
  int month;
  int day;
  int hour;
  int minute;
  int second;
};

// this is using conventions where month and day start at 1, as is common
time_t& posix_timestamp_from_YMDHMS(int year, int month, int day, int hour, int minute, int second);
struct_YMDHMS& YMDHMS_from_posix_timestamp(time_t posix_timestamp);

extern TimeManager board_time_manager;
extern volatile kiss_time_t posix_timestamp; // a timestamp

extern tmElements_t common_working_time_struct;
extern time_t common_working_posix_timestamp;
extern struct_YMDHMS common_working_struct_YMDHMS;


#endif
