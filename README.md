# cvTagging
A realtime tagging tool for videos based on `OpenCV`


# Config File:
`config.ini`:
* `InputVideo`: video source
* `LabelFile`: classes to label (one class per line)
* `OutputDir`: where(directory) to store the info of tags(filename will be <video-filename>.tag), the output format is: `label-name` `<tab>` `frame-number` `<tab>` `x` `<tab>` `y` `<tab>` `width` `<tab>` `height` per line.


# Key Mapping:
* `p`: taggle tagging/playing mode.
* `,`, `.`: switch between tags which are created.
* `x`: delete tag which is selected.
* `[`, `]`: switch between labels(show at top-right corner).
* `z`: print debug messages.
* `w`, `a`, `s`, `d`: move the bounding box up/left/down/right-ward of the selected tag.
* `-`, `+`: enlarge/shrink the bounding box of the selected tag.
* `q`, `ESC`: quit


# Dependencies:
`OpenCV`, `boost` libraries


# TODO:
* Add OpenGL support

