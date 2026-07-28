#include "gnss_manager.h"

SFE_UBLOX_GNSS gnss;

GNSS_Manager gnss_manager;

void turn_gnss_on(void){
    pinMode(PIN_QWIIC_PWR, OUTPUT);
    digitalWrite(PIN_QWIIC_PWR, HIGH);
}

void turn_gnss_off(void){
    pinMode(PIN_QWIIC_PWR, OUTPUT);
    digitalWrite(PIN_QWIIC_PWR, LOW);
}

void print_gnss_status(SFE_UBLOX_GNSS &g){
  wdt.restart();

  // getFixType() returns the latest PVT (auto or polled); the other PVT-derived
  // getters below reuse that same epoch, so this is a coherent snapshot.
  uint8_t fix = g.getFixType();
  SERIAL_USB->print(F("[gnss] fixType="));
  SERIAL_USB->print(fix);
  SERIAL_USB->print(F(" (0=none 2=2D 3=3D 5=time-only)  fixOK="));
  SERIAL_USB->print(g.getGnssFixOk());
  SERIAL_USB->print(F("  SIV="));
  SERIAL_USB->print(g.getSIV());
  SERIAL_USB->print(F("  PDOP="));
  SERIAL_USB->print(g.getPDOP() / 100.0, 2);
  SERIAL_USB->print(F("  | timeValid="));
  SERIAL_USB->print(g.getTimeValid());
  SERIAL_USB->print(F(" fullyResolved="));
  SERIAL_USB->print(g.getTimeFullyResolved());
  SERIAL_USB->print(F(" confirmedTime="));
  SERIAL_USB->println(g.getConfirmedTime());
  SERIAL_USB->flush();

  wdt.restart();

  // Per-satellite carrier-to-noise (UBX-NAV-SAT). With a GPS repeater this is
  // the key diagnostic: it shows whether any usable signal is reaching the
  // antenna. Typical thresholds: C/N0 < ~20 dB-Hz = too weak to use; a fix
  // generally needs >=4 sats at >~25-30 dB-Hz.
  if (g.getNAVSAT()){
    uint8_t n = g.packetUBXNAVSAT->data.header.numSvs;
    uint8_t with_signal = 0, used = 0, max_cno = 0;
    for (uint8_t i = 0; i < n; i++){
      uint8_t cno = g.packetUBXNAVSAT->data.blocks[i].cno;
      if (cno > 0) with_signal++;
      if (cno > max_cno) max_cno = cno;
      if (g.packetUBXNAVSAT->data.blocks[i].flags.bits.svUsed) used++;
    }
    SERIAL_USB->print(F("[gnss] NAV-SAT: "));
    SERIAL_USB->print(n); SERIAL_USB->print(F(" listed, "));
    SERIAL_USB->print(with_signal); SERIAL_USB->print(F(" with signal, "));
    SERIAL_USB->print(used); SERIAL_USB->print(F(" used in nav, max C/N0="));
    SERIAL_USB->print(max_cno); SERIAL_USB->println(F(" dB-Hz"));
    // list the usable ones (gnssId: 0=GPS 1=SBAS 2=Galileo 3=BeiDou 5=QZSS 6=GLONASS)
    for (uint8_t i = 0; i < n; i++){
      uint8_t cno = g.packetUBXNAVSAT->data.blocks[i].cno;
      if (cno >= 15){
        SERIAL_USB->print(F("    gnssId="));
        SERIAL_USB->print(g.packetUBXNAVSAT->data.blocks[i].gnssId);
        SERIAL_USB->print(F(" svId="));
        SERIAL_USB->print(g.packetUBXNAVSAT->data.blocks[i].svId);
        SERIAL_USB->print(F(" C/N0="));
        SERIAL_USB->print(cno);
        SERIAL_USB->print(F(" qual="));
        SERIAL_USB->print(g.packetUBXNAVSAT->data.blocks[i].flags.bits.qualityInd);
        SERIAL_USB->print(F(" used="));
        SERIAL_USB->println(g.packetUBXNAVSAT->data.blocks[i].flags.bits.svUsed);
      }
    }
  }
  else{
    SERIAL_USB->println(F("[gnss] NAV-SAT poll failed"));
  }
  SERIAL_USB->flush();
  wdt.restart();
}

bool GNSS_Manager::get_a_fix(unsigned long timeout_seconds, bool set_RTC_time, bool perform_full_start, bool perform_full_stop){
  wdt.restart();
  good_fit = false;

  SERIAL_USB->println(F("attempt gnss fix"));

  if (perform_full_start){
    // power things up and connect to the GNSS; if fail several time, restart the board
    bool gnss_startup {false};
    for (int i=0; i<5; i++){
      Wire1.begin();
      SERIAL_USB->println(F("Wire1 started"));
      turn_gnss_on();
      delay(1000); // Give it time to power up
      wdt.restart();

      SERIAL_USB->println(F("gnss powered up"));
      SERIAL_USB->flush();

      if (!gnss.begin(Wire1)){
        SERIAL_USB->println(F("problem starting GNSS"));
        
        // power things down
        turn_gnss_off();
        Wire1.end();
        delay(500);
        continue;
      }
      else{
        SERIAL_USB->println(F("success starting GNSS"));
        gnss_startup = true;
        break;
      }
      wdt.restart();
    }

    wdt.restart();

    if (!gnss_startup){
      SERIAL_USB->println(F("failed to start GNSS; reboot"));
      while (1){;}
    }

    gnss.setI2COutput(COM_TYPE_UBX); // Limit I2C output to UBX (disable the NMEA noise)

    delay(100);

    // If we are going to change the dynamic platform model, let's do it here.
    // Possible values are:
    // PORTABLE,STATIONARY,PEDESTRIAN,AUTOMOTIVE,SEA,AIRBORNE1g,AIRBORNE2g,AIRBORNE4g,WRIST,BIKE
    if (!gnss.setDynamicModel(DYN_MODEL_STATIONARY)){
      SERIAL_USB->println(F("GNSS could not set dynamic model"));
    }

    wdt.restart();
  }

  // now ready take a measurement
  byte gnss_fix_status {0};
  wdt.restart();

  SERIAL_USB->println(F("attempt GNSS fix, remaining time to fix timeout:"));
  delay(5);
  for (int i=0; i<timeout_seconds;i++){
    SERIAL_USB->print(F("-"));
  }
  SERIAL_USB->println();
  delay(10);

  unsigned int status_tick {0};
  for (unsigned long start_millis=millis(); (gnss_fix_status != 3) && (millis() - start_millis < timeout_seconds * 1000UL); ){
    wdt.restart();
    SERIAL_USB->print(F("-"));
    delay(500);
    gnss_fix_status = gnss.getFixType();
    // Every ~5 s, dump full receiver status so a no-fix situation (e.g. indoors
    // / under a repeater) is diagnosable instead of just a row of dashes.
    if (++status_tick % 10 == 0){
      SERIAL_USB->println();
      print_gnss_status(gnss);
    }
  }
  SERIAL_USB->println();

  wdt.restart();

  if (gnss_fix_status == 3){
    // read the data and store it here
    good_fit = true;
    milliseconds = gnss.getMillisecond();
    second = gnss.getSecond();
    minute = gnss.getMinute();
    hour = gnss.getHour();
    day = gnss.getDay();
    month = gnss.getMonth();
    year = gnss.getYear();
    latitude = gnss.getLatitude();
    longitude = gnss.getLongitude();
    altitude = gnss.getAltitudeMSL();
    speed = gnss.getGroundSpeed();
    satellites = gnss.getSIV();
    course = gnss.getHeading();
    pdop = gnss.getPDOP();

    SERIAL_USB->print(F("we got a gnss fix:"));
    SERIAL_USB->print(year); SERIAL_USB->print(F("-")); SERIAL_USB->print(month); SERIAL_USB->print(F("-")); SERIAL_USB->print(day);
      SERIAL_USB->print(F(" ")); SERIAL_USB->print(hour); SERIAL_USB->print(F(":")); SERIAL_USB->print(minute); SERIAL_USB->print(F(":"));
      SERIAL_USB->print(second); SERIAL_USB->print(F(" ")); SERIAL_USB->print(latitude); SERIAL_USB->print(F(",")); SERIAL_USB->print(longitude);
      SERIAL_USB->println();

    wdt.restart();

    // compute the corresponding posix timestamp
    common_working_posix_timestamp = posix_timestamp_from_YMDHMS(
      year, month, day, hour, minute, second
    );

    SERIAL_USB->print(F("we computed a posix timestamp: ")); SERIAL_USB->println((unsigned long)common_working_posix_timestamp);
    
    posix_timestamp = common_working_posix_timestamp;

    if (set_RTC_time){
      board_time_manager.set_posix_timestamp(common_working_posix_timestamp);
    }
  }
  else{
    SERIAL_USB->println(F("GNSS timed out without fix"));
  }

  wdt.restart();

  // power things down
  if (perform_full_stop){
    turn_gnss_off();
    Wire1.end();
  }

  wdt.restart();

  return good_fit;
}

bool GNSS_Manager::get_and_push_fix(unsigned long timeout_seconds){
  wdt.restart();
  
  SERIAL_USB->println(F("start with GNSS buffer:"));
  
  if(get_a_fix(timeout_seconds, true, true, false)){
    number_of_GPS_fixes += 1;

    // clear the accumulators
    crrt_accumulator_latitude.clear();
    crrt_accumulator_longitude.clear();
    crrt_accumulator_posix_timestamp.clear();

    // push the current fix
    crrt_accumulator_latitude.push_back(latitude);
    crrt_accumulator_longitude.push_back(longitude);
    crrt_accumulator_posix_timestamp.push_back(posix_timestamp);

    // to be on the safe side, get a few extra fixes, and do some filtering on it to keep only "good" fixes;
    // maybe this helps avoid the "bad fix" problems (?)
    wdt.restart();
    delay(5000); // give a bit of time for the GPS reading to "warm up"?
    wdt.restart();

    // sample of few extra fixes for the n-sigma filtering; we sample at most enough to fill
    // the accumulators, and we use at most 60 seconds
    int crrt_fix_nbr {1};
    unsigned long millis_start = millis();
    while ( (crrt_fix_nbr < size_accumulators-1) && (millis() - millis_start < 60000UL)){
      if(get_a_fix(5UL, true, false, false)){
        crrt_fix_nbr += 1;
        crrt_accumulator_latitude.push_back(latitude);
        crrt_accumulator_longitude.push_back(longitude);
        crrt_accumulator_posix_timestamp.push_back(posix_timestamp);
        wdt.restart();
      }
    }

    // we turn off by hand, since we did not perform full start stop in the loop
    turn_gnss_off();
    Wire1.end();

    // then, get the values for the filtered lat, lon, timestamp
    long crrt_latitude = accurate_sigma_filter<long>(crrt_accumulator_latitude, 2.0);
    long crrt_longitude = accurate_sigma_filter<long>(crrt_accumulator_longitude, 2.0); 
    long crrt_posix_timestamp = accurate_sigma_filter<long>(crrt_accumulator_posix_timestamp, 2.0); 

    fix_information crrt_fix {crrt_posix_timestamp, crrt_latitude, crrt_longitude};


    SERIAL_USB->print(F("pushed fix: "));
    SERIAL_USB->print(crrt_fix.posix_timestamp);
    SERIAL_USB->print(F(" | "));
    SERIAL_USB->print(crrt_fix.latitude);
    SERIAL_USB->print(F(" | "));
    SERIAL_USB->print(crrt_fix.longitude);
    SERIAL_USB->println();

    return true;
  }

  turn_gnss_off();
  Wire1.end();
  
  return false;
}

