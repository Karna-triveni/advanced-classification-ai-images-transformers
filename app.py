from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from scipy import fft
import cv2
from operator import itemgetter
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

class QuantizationMatrix:
    """Quantization matrices for different quality factors"""
    
    Q50 = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

    Q75 = np.array([[8, 6, 5, 8, 12, 20, 26, 31],
                   [6, 6, 7, 10, 13, 29, 30, 28],
                   [7, 7, 8, 12, 20, 29, 35, 28],
                   [7, 9, 11, 15, 26, 44, 40, 31],
                   [9, 11, 19, 28, 34, 55, 52, 39],
                   [12, 18, 28, 32, 41, 52, 57, 46],
                   [25, 32, 39, 44, 52, 61, 60, 52],
                   [36, 46, 48, 49, 56, 50, 52, 50]])

    Q90 = np.array([[3, 2, 2, 3, 5, 8, 10, 12],
                    [2, 2, 3, 4, 5, 12, 12, 11],
                    [3, 3, 3, 5, 8, 11, 14, 11],
                    [3, 3, 4, 6, 10, 17, 16, 12],
                    [4, 4, 7, 11, 14, 22, 21, 15],
                    [5, 7, 11, 13, 16, 12, 23, 18],
                    [10, 13, 16, 17, 21, 24, 24, 21],
                    [14, 18, 19, 20, 22, 20, 20, 20]])

    Qrand = np.array([[4, 4, 6, 11, 24, 24, 24, 24],
                      [4, 5, 6, 16, 24, 24, 24, 24],
                      [6, 6, 14, 24, 24, 24, 24, 24],
                      [11, 16, 24, 24, 24, 24, 24, 24],
                      [24, 24, 24, 24, 24, 24, 24, 24],
                      [24, 24, 24, 24, 24, 24, 24, 24],
                      [24, 24, 24, 24, 24, 24, 24, 24],
                      [24, 24, 24, 24, 24, 24, 24, 24]])

    def get_qm(self, qf=0.75):
        if qf == 0.5:
            return self.Q50
        elif qf == 0.75:
            return self.Q75
        elif qf == 0:
            return self.Qrand
        elif qf == 0.9:
            return self.Q90
        else:
            return self.Q75


def read_img_from_base64(base64_string):
    """Convert base64 string to image array"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    
    # Convert to numpy array
    original_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    overlay = original_image.copy()
    img_array = np.array(image)
    height, width = img_array.shape
    
    return img_array, original_image, overlay, width, height


def create_quantize_dct(img, width, height, block_size, stride, Q_8x8):
    """Create sliding windows, apply DCT, and quantize"""
    quant_row_matrices = []

    for i in range(0, height - block_size, stride):
        for j in range(0, width - block_size, stride):
            block = img[i: i + block_size, j: j + block_size]
            
            # DCT transform
            dct_matrix = fft.dct(block)

            # Quantization of DCT coefficients
            quant_block = np.round(np.divide(dct_matrix, Q_8x8))
            block_row = list(quant_block.flatten())

            # Store left-corner pixel coordinates and block
            quant_row_matrices.append([(i, j), block_row])
    
    return quant_row_matrices


def lexographic_sort(quant_row_matrices):
    """Find matched blocks using lexicographic sort"""
    sorted_blocks = sorted(quant_row_matrices, key=itemgetter(1))

    matched_blocks = []
    shift_vec_count = {}

    for i in range(len(sorted_blocks) - 1):
        if sorted_blocks[i][1] == sorted_blocks[i + 1][1]:
            point1 = sorted_blocks[i][0]
            point2 = sorted_blocks[i + 1][0]

            # Calculate shift vector
            s = np.linalg.norm(np.array(point1) - np.array(point2))

            # Increment count for shift vector
            shift_vec_count[s] = shift_vec_count.get(s, 0) + 1
            matched_blocks.append([sorted_blocks[i][1], sorted_blocks[i + 1][1],
                                point1, point2, s])
    
    return shift_vec_count, matched_blocks


def shift_vector_thresh(shift_vec_count, matched_blocks, shift_thresh):
    """Apply shift vector threshold"""
    matched_pixels_start = []
    for sf in shift_vec_count:
        if shift_vec_count[sf] > shift_thresh:
            for row in matched_blocks:
                if sf == row[4]:
                    matched_pixels_start.append([row[2], row[3]])
    
    return matched_pixels_start


def create_result_image(overlay, original_image, matched_pixels_start, block_size):
    """Create the result image with highlighted forgery regions"""
    alpha = 0.5

    for starting_points in matched_pixels_start:
        p1 = starting_points[0]
        p2 = starting_points[1]

        # Highlight matched regions in red and green
        overlay[p1[0]: p1[0] + block_size, p1[1]: p1[1] + block_size] = (0, 0, 255)
        overlay[p2[0]: p2[0] + block_size, p2[1]: p2[1] + block_size] = (0, 255, 0)

    result = cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0)
    
    return result


def image_to_base64(image):
    """Convert numpy image array to base64 string"""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(image_rgb)
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"


@app.route('/api/detect', methods=['POST'])
def detect_forgery():
    """Main endpoint for forgery detection"""
    try:
        data = request.json
        
        # Get parameters
        image_data = data.get('image')
        block_size = int(data.get('blockSize', 8))
        qf = float(data.get('qualityFactor', 0.75))
        shift_thresh = int(data.get('shiftThresh', 10))
        stride = int(data.get('stride', 1))
        
        # Validate parameters
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get quantization matrix
        Q_8x8 = QuantizationMatrix().get_qm(qf)
        
        # Read image from base64
        img, original_image, overlay, width, height = read_img_from_base64(image_data)
        
        # Apply DCT and quantization
        quant_row_matrices = create_quantize_dct(img, width, height, block_size, stride, Q_8x8)
        
        # Lexicographic sort to find matches
        shift_vec_count, matched_blocks = lexographic_sort(quant_row_matrices)
        
        # Apply shift vector threshold
        matched_pixels_start = shift_vector_thresh(shift_vec_count, matched_blocks, shift_thresh)
        
        # Create result image
        result_image = create_result_image(overlay, original_image, matched_pixels_start, block_size)
        
        # Convert result to base64
        result_base64 = image_to_base64(result_image)
        
        # Calculate statistics
        num_forgeries = len(matched_pixels_start)
        total_shift_vectors = len(shift_vec_count)
        max_shift_count = max(shift_vec_count.values()) if shift_vec_count else 0
        
        return jsonify({
            'success': True,
            'result_image': result_base64,
            'statistics': {
                'forgery_regions': num_forgeries,
                'total_shift_vectors': total_shift_vectors,
                'max_shift_count': max_shift_count,
                'forgery_detected': num_forgeries > 0
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Image Forgery Detection API'
    })


@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Image Forgery Detection API',
        'endpoints': {
            '/api/detect': 'POST - Detect forgery in images',
            '/api/health': 'GET - Health check'
        }
    })


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)