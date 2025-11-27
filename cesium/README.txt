To generate the telemetry JSON, place the DJI CSV file in the current directory and run:
python telemtry_log_loader.py

Columns of Interest 

We only care about a small subset of the DJI log columns for syncing telemetry with detections.

1. Timestamp data
   - CUSTOM.date [local]        = Local calendar date for each log row (e.g., 11/25/2025).
   - CUSTOM.updateTime [local]  = Local clock time for each row, centisecond precision (e.g., 7:05:08.97 PM).
   - timestamp_epoch            = Derived field in input.json; Unix epoch seconds computed from
                                  CUSTOM.date [local] + CUSTOM.updateTime [local] for precise time
                                  alignment with YOLO detections.

2. Latitude / longitude (position)
   - OSD.latitude               = Aircraft GPS latitude (degrees).
   - OSD.longitude              = Aircraft GPS longitude (degrees).
   - HOME.latitude              = Home-point latitude (degrees); useful as a reference origin.
   - HOME.longitude             = Home-point longitude (degrees).
   - APPGPS.latitude            = Mobile-device GPS latitude (fallback / debug).
   - APPGPS.longitude           = Mobile-device GPS longitude (fallback / debug).

3. Orientation data
   - OSD.yaw                    = Aircraft yaw (degrees, -180 to 180).
   - OSD.yaw [360]              = Aircraft yaw in [0, 360) degrees; easiest to use for heading.
   - OSD.pitch                  = Aircraft pitch (degrees).
   - OSD.roll                   = Aircraft roll (degrees).
   - OSD.directionOfTravel      = Direction of travel over ground (degrees).
   - HOME.aircraftHeadDirection = Aircraft heading relative to home point.
   - GIMBAL.pitch               = Gimbal pitch angle (degrees).
   - GIMBAL.roll                = Gimbal roll angle (degrees).
   - GIMBAL.yaw                 = Gimbal yaw angle (degrees).
   - GIMBAL.yaw [360]           = GIMBAL yaw in [0, 360) degrees; useful for camera frustum direction.

4. Altitude / height (how high up the drone is)
   - OSD.altitude [ft]          = Aircraft altitude in feet (primary field for “how high up” in the log).
   - OSD.height [ft]            = Flight height in feet (typically relative to takeoff/home).
   - OSD.heightMax [ft]         = Maximum height reached so far in this flight (feet).
   - OSD.vpsHeight [ft]         = Height from the Vision Positioning System (feet), relative to ground below.
   - HOME.height [ft]           = Aircraft height relative to home when last updated (feet).
   - HOME.heightLimit [ft]      = Configured maximum flight height limit (feet).
   - HOME.goHomeHeight [ft]     = Target altitude during Return-to-Home (feet).
   - HOME.forceLandingHeight [ft]= Height at which forced landing behavior activates (feet).
   - DETAILS.maxHeight [ft]     = Maximum flight height reached in this flight (feet, summary).

5. Summary / totals (end-of-flight stats)
   - DETAILS.totalTime [s]      = Total flight time.
   - DETAILS.totalDistance [ft] = Total distance traveled.
   - DETAILS.maxHorizontalSpeed [MPH] = Max horizontal speed.
   - DETAILS.maxVerticalSpeed [MPH]   = Max vertical speed.
   - DETAILS.photoNum           = Number of photos taken.
   - DETAILS.videoTime [s]      = Total time recording video.
   - DETAILS.maxHeight [ft]     = Maximum height during flight.