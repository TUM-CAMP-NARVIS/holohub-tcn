# tcn_shm_ssg_inference

Test to evaluate shm based data streaming

## Overview

This application is built using Holoscan SDK version 3.7.0 and supports the following platforms:


## Prerequisites

- Holoscan SDK 3.7.0
- CUDA (if using GPU acceleration)
- Docker (for containerized deployment)

## Installation

1. Clone this repository

2. Install dependencies:

3. Build the application:

## Usage

### Running the Application

```bash
./holohub run tcn_shm_ssg_inference
```

By default, the `./holohub build` and `./holohub run` commands will build and run the application in a containerized environment using the `standard` mode.

For local development without containers, use the `--local` flag:

```bash
./holohub run tcn_shm_ssg_inference --local
```

Note that for the `--local` flag, the relevant custom dependencies (e.g. `requirements.txt` for Python) will be ignored and need to be installed manually.

### For containerized deployment

The application includes a Dockerfile for containerized deployment:

```bash
# Build the container
./holohub build-container tcn_shm_ssg_inference

# Run the containerized application
./holohub run-container tcn_shm_ssg_inference
```

For custom Docker builds:

```bash
# Build with custom base image
./holohub build-container tcn_shm_ssg_inference --base-image nvcr.io/nvidia/clara-holoscan/holoscan:v3.7.0-dgpu

# Run with specific GPU type
./holohub run-container tcn_shm_ssg_inference --gpu-type dgpu
```

## Development

### Project Structure

```
tcn_shm_ssg_inference/
├── CMakeLists.txt
├── Dockerfile
├── README.md
├── requirements.txt
├── src/
│   └── main.py
├── include/
├── tests/
└── docs/
```

### Adding New Operators

1. Create a new operator class in `src/operators/`
2. Include the operator in `src/main.py`
3. Update the pipeline configuration

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Authors

- Ulrich Eck - TU Munich

## Acknowledgments

- NVIDIA Holoscan Team
- Open source community contributors
