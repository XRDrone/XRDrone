# Runtime UI and Terminal Reference

This document explains the text you may see on the video/output window and in the terminal during normal pipeline use.

## On-screen labels

### Detection labels

When tracked-box display is not active, the label format is:

```text
<class> <confidence>%
```

When tracked-box display is active, the label format becomes:

```text
<class> #<track_id> <confidence>%
```

If the continuity layer is coasting a track forward, the label includes:

```text
[hold]
```

### DJI overlay

If `DJI_MENU_OVERLAY_ENABLED_DEFAULT` is on, the pipeline can draw a pre-made RGBA overlay image across the frame.

### Window title

The OpenCV window title is:

```text
Live Pipeline
```

## Terminal messages

You may see messages when:

- recording is enabled or disabled
- the active input source is switched
- tracking is toggled
