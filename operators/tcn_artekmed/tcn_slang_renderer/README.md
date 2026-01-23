# TCN Slang Renderer Operator

A Holoscan operator that combines window management with Slang shader compilation for Vulkan-based rendering.

## Overview

The TCN Slang Renderer operator creates a window with Vulkan context and uses Slang shaders for rendering. It integrates:
- **Window management** (similar to HoloViz) using GLFW
- **Vulkan pipeline** for graphics rendering
- **Slang shader compilation** for flexible shader development

## Features

- Creates a windowed Vulkan rendering context
- Supports custom Slang shaders via source string or file
- Default triangle rendering with colored vertices
- Full operator lifecycle: `initialize`, `setup`, `start`, `compute`, `stop`

## Parameters

- `width` (uint32_t, default: 800): Window width in pixels
- `height` (uint32_t, default: 600): Window height in pixels
- `window_title` (string, default: "TCN Slang Renderer"): Window title
- `shader_source` (string, optional): Slang shader source code
- `shader_source_file` (string, optional): Path to Slang shader file

## Default Behavior

When no shader is provided, the operator renders a colored triangle using a default Slang shader with:
- Vertex shader that positions three vertices
- Fragment shader that colors each vertex (red, green, blue)

## Build Requirements

- Holoscan SDK
- Vulkan SDK
- GLFW3
- Slang compiler (automatically fetched)
- GCC 13.0.0 or newer

## Usage Example

```cpp
#include <tcn_slang_renderer/tcn_slang_renderer.hpp>

auto renderer = make_operator<TcnSlangRenderOp>(
  "tcn_renderer",
  Arg("width", 1920u),
  Arg("height", 1080u),
  Arg("window_title", "My Renderer")
);
```

## Implementation Details

The operator uses a pimpl pattern to hide Vulkan and Slang implementation details:
- **GLFW** for cross-platform window creation
- **Vulkan** for low-level graphics rendering
- **Slang** for shader compilation (with SPIRV target)

### Lifecycle Methods

- `initialize()`: Base operator initialization
- `setup()`: Parameter setup and implementation creation
- `start()`: Window, Vulkan, and pipeline initialization
- `compute()`: Per-frame rendering and event polling
- `stop()`: Resource cleanup

## Future Enhancements

- Dynamic shader reloading
- Input handling (keyboard/mouse)
- Integration with Holoscan tensor inputs for data visualization
- Custom geometry rendering beyond triangles
- Slang shader hot-reload
