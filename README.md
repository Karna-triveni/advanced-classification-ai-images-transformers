# Advanced Classification of AI-Generated Images through Transformers

## About

This project detects image tampering using a **copy-move forgery detection** approach based on the **Discrete Cosine Transform (DCT)**. It identifies regions within an image that have been copied and pasted elsewhere — a common technique used to manipulate or forge images.

The detector works by:
1. Splitting the image into overlapping blocks
2. Applying a DCT to each block and quantizing the coefficients
3. Sorting blocks lexicographically to find near-identical blocks
4. Computing shift vectors between matched blocks and thresholding on how often each shift occurs
5. Highlighting the matched (forged) regions on the original image

The project includes both a **command-line tool** and a **Flask-based web app** with a simple frontend for uploading an image and viewing the detected forgery regions.

## Project Structure

```
├── app.py                  # Flask web app / REST API for forgery detection
├── code/
│   ├── cmf_detect.py        # CLI script for copy-move forgery detection
│   ├── quant_matrix.py       # Quantization matrices for DCT (QF 50/75/90)
│   └── utils/
│       └── helper_utils.py   # Image I/O, DCT, sorting, thresholding helpers
├── templates/
│   └── index.html            # Web UI served by the Flask app
├── images/                   # Sample forged images for testing
├── requirements.txt
└── run_command.txt           # Example CLI invocation
```

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/Karna-triveni/advanced-classification-ai-images-transformers.git
   cd advanced-classification-ai-images-transformers
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option A: Command line

```bash
python code/cmf_detect.py --img images/forged1.png --block_size 8 --qf 0.75 --shift_thresh 10 --stride 1
```

**Arguments:**
| Argument | Description | Default |
|---|---|---|
| `--img` | Path to the image to analyze | required |
| `--block_size` | Size of the sliding block window | `8` |
| `--qf` | Quality factor for the quantization matrix (`0`, `0.5`, `0.75`, `0.9`) | `0.75` |
| `--shift_thresh` | Minimum repeated-shift count to flag a region as forged | `10` |
| `--stride` | Step size for the sliding window | `1` |

### Option B: Web app

```bash
python app.py
```
Then open `http://localhost:5000` in your browser to upload an image through the web UI and view the highlighted forgery regions.

**API endpoint:**
- `POST /api/detect` — accepts a base64-encoded image and detection parameters, returns the annotated result image plus detection statistics
- `GET /api/health` — health check

## Sample Results

Forged regions are highlighted with matched blocks marked in red and green on the output image.

## Notes

- Detection accuracy depends heavily on `block_size`, `qf`, and `shift_thresh` — tune these per image.
- Test images are provided in the `images/` folder to try the tool out of the box.
