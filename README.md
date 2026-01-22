# Sharp Web UI

A modern web interface for **SHARP: Sharp Monocular View Synthesis in Less Than a Second**.
Easily generate, view, and export 3D Gaussian Splats from a single image.

## Getting Started

### Prerequisites

*   **Node.js**: Required for the frontend and development scripts.
*   **Python 3.10+**: Required for the backend model.

### Installation

The easiest way to install is to use the provided installation scripts:

**On macOS/Linux:**
```bash
./install.sh
```

**On Windows:**
```powershell
.\install.bat
```

The script will automatically set up a Python virtual environment, install all dependencies, and prepare the project.

### Running the App

Once installed, you can start the application using the run scripts:

**On macOS/Linux:**
```bash
./run.sh
```

**On Windows:**
```powershell
.\run.bat
```

This will launch both the Python backend and the React frontend.


*   **Frontend**: `http://localhost:5173`
*   **Backend**: `http://localhost:8000`

> **Note**: On the first run, the application will automatically download the necessary model checkpoints (approx. same size as the manual download). This may take a few moments.

## Features

*   **Instant Generation**: Drag & drop an image to generate a 3D scene in seconds.
*   **Interactive Viewer**: integrated Splat viewer to inspect results immediately.
*   **Export Options**:
    *   Download as `.ply` (Standard Gaussian Splat)
    *   Download as `.ksplat` (Compressed, web-ready)
    *   Convert to Mesh (`.obj` / `.glb`)

---

## About the SHARP Model

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

![](data/teaser.jpg)

This project is built on top of the research paper: **Sharp Monocular View Synthesis in Less Than a Second**
by _Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy, Tian Fang, Yanghai Tsin, Stephan Richter and Vladlen Koltun_.

We present SHARP, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. This is done in less than a second on a standard GPU via a single feedforward pass through a neural network. The 3D Gaussian representation produced by SHARP can then be rendered in real time, yielding high-resolution photorealistic images for nearby views.

### CLI Usage (Advanced)

If you prefer using the command line interactively:

```bash
# Activate environment
source .venv/bin/activate

# Check installation
sharp --help

# Run prediction
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

### Citation

If you find this work useful, please cite the original paper:

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama\"{e}l Delaunoy and Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code and [LICENSE_MODEL](LICENSE_MODEL) for the released models.
