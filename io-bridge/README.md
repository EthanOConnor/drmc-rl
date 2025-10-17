Real-console I/O bridge (later):

- Video in: HDMI capture frames to pixel→state network
- Controller out: Microcontroller (RP2040/Arduino) emulating NES 4021 shift-register protocol ($4016/$4017)
- Sync button bits to VBlank at 60 Hz; many community references available
