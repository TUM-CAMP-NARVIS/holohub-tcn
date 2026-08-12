# TCN Utils

The `TcnUtils` collects useful operators / utility functions

## Overview


## Features


## Usage

### Basic Usage with File Path

```python
```

## `rotate.rotate180`

180° rotation as a plain function (`arr[::-1, ::-1]`), working on numpy or cupy and importing neither,
so it is host-testable and usable **inline** inside another operator's `compute()`.

`RotateImage180Op` is the graph-node form and calls it. Prefer the function when rotating individual
tensors out of a multi-camera entity — `tcn_langsam` does this for upside-down cameras — because a
graph node per camera would mean splitting and re-merging the entity for no gain.

It returns a reversed **view**; callers handing the result to a tensor consumer must
`ascontiguousarray` it. The rotation is its own inverse and involves no resampling, which is what makes
"rotate in, rotate the result back out" exact.
