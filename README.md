# cvTagging
A realtime tagging tool for videos based on `OpenCV`

![demo](demo.gif)


## Config File
`config.ini`:

* `InputVideo`: video source
* `LabelFile`: classes to label (one class per line)
* `OutputDir`: where(directory) to store the info of tags(filename will be `<video-filename>.tag`), the output format is: `label-name` `<tab>` `frame-number` `<tab>` `x` `<tab>` `y` `<tab>` `width` `<tab>` `height` per line.

## Build
In prompt: `$ cd src && make`


## Usage
1. In prompt: `$ ./e2tagging`
2. Start video playing by pressing `p`.
3. Press `p` to enter `tagging mode`.
3. Press `[`, `]` to switch label classes.
4. Use mouse(lift-button) to create bounding box of the tag.
5. Press `p` to continue video playing.
6. The tagging infos are automatically saved.


## Key Mapping
* `p`: taggle tagging/playing mode. In tagging mode, use mouse(left-button) to define the bounding box of the tag.
* `,`, `.`: switch between tags which are created.
* `x`: delete tag which is selected.
* `[`, `]`: switch between labels(show at top-right corner).
* `z`: print debug messages.
* `w`, `a`, `s`, `d`: move the bounding box up/left/down/right-ward of the selected tag.
* `-`, `+`: enlarge/shrink the bounding box of the selected tag.
* `q`, `ESC`: quit


## Dependencies
`OpenCV`, `boost` libraries


## TODO
* Add OpenGL support

