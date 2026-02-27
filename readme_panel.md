## Overview

This viewer displays induced-current and electronics waveforms from a pixel-based particle
detector. Deposited charge drifts toward a pixel anode plane, inducing a current signal on
each pixel. The front-end electronics integrate that current and record a hit once the
accumulated charge crosses a configurable threshold; the integrated value is then digitised
by an ADC.

## Pixel Map (left panel)

Each square marker represents one pixel positioned at its physical (z, y) coordinates on
the anode plane. Colour encodes the **peak absolute induced current** for that pixel in the
loaded event — brighter orange indicates a larger signal. Hover over a pixel to see its ID,
position, and peak current. **Click a pixel** to load its waveforms in the right panel.

## Waveform Panel (right panel)

- **Pixel Signal I (top):** Induced current on the pixel as a function of time (e-/µs) from
  the drifting charge.
- **True Charge Q (middle):** True integrated charge recorded by the electronics
  simulation (e-).
- **Reco Charge Q (bottom):** Reconstructed integrated charge including noise and signal
  processing effects (e-).

## Annotations & Overlays

- **Dashed cyan trace:** Cumulative integral of the pixel signal (sig_sum), overlaid on the
  True Q and Reco Q subplots for comparison with the electronics output.
- **Amber dotted lines:** ADC hit times - the moments in time where the electronics recorded a hit.
- **∫ badges:** Numerical time-integral of each waveform, displayed in the top-left corner
  of each subplot. For the pixel signal / induced current this will integrate to the total number
  of electrons seen by the pixel.
