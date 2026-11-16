# Reused local sources

- `vendor/Rcb4BaseLib.py`: copied unchanged from `/Users/yuchenzhou/Desktop/khr3hv-exploration/Rcb4Lib/Rcb4BaseLib.py`. Kondo Kagaku, original MIT notice retained in file.
- `vendor/desktop_power_monitor_reference.py`: copied from the same desktop project's `shunt/python/power_monitor.py`. CSV voltage/current/clipping interpretation reused in `common.py`.
- `firmware/KHR3HV_PowerMonitor_Transfer/`: derived from that project's M5INA226 Arduino sketch. Added fresh-only emission, freshness timestamp, protocol announcement, clipping flag.
- PC-control configuration transition follows `outputs/humanoid-energy-paper/hardware_experiment/acquire.py`. This package implements its own timed trajectory loop and does not change archived experiments.

No claim of new hardware calibration or Windows execution validation.
