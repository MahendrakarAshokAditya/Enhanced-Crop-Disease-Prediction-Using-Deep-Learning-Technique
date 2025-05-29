#app.py
from flask import Flask, render_template, request, jsonify, url_for, redirect
import os
import base64
import io
import json
import uuid
import datetime
from PIL import Image, ImageOps
import numpy as np
from model_handler import DiseaseDetector
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['HISTORY_FILE'] = 'static/data/history.json'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure data directory exists
os.makedirs(os.path.dirname(app.config['HISTORY_FILE']), exist_ok=True)

# Initialize the disease detector
detector = DiseaseDetector(model_path='best.pt')

# Load history data or create empty history
def load_history():
    try:
        if os.path.exists(app.config['HISTORY_FILE']):
            with open(app.config['HISTORY_FILE'], 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading history: {str(e)}")
        return []

# Save history data
def save_history(history):
    try:
        with open(app.config['HISTORY_FILE'], 'w') as f:
            json.dump(history, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving history: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def history():
    history_items = load_history()
    return render_template('history.html', history_items=history_items)

@app.route('/history/<item_id>')
def history_detail(item_id):
    history_items = load_history()
    item = next((item for item in history_items if item['id'] == item_id), None)
    
    if not item:
        return render_template('history_detail.html', item=None)
    
    # Find related items (same disease or plant type)
    related_items = [i for i in history_items if 
                    (i['disease_name'] == item['disease_name'] or 
                     i['plant_type'] == item['plant_type']) and 
                    i['id'] != item['id']]
    related_items = sorted(related_items, key=lambda x: x['date'], reverse=True)[:4]
    
    # Add current item to related items for timeline display
    all_related = related_items.copy()
    all_related.append(item)
    all_related = sorted(all_related, key=lambda x: x['date'])
    
    return render_template('history_detail.html', 
                           item=item, 
                           related_items=all_related, 
                           all_items=history_items)

@app.route('/history/<item_id>/data')
def history_item_data(item_id):
    history_items = load_history()
    item = next((item for item in history_items if item['id'] == item_id), None)
    
    if not item:
        return jsonify({'success': False, 'error': 'Item not found'})
    
    return jsonify({'success': True, 'item': item})

@app.route('/history/<item_id>/delete', methods=['POST'])
def delete_history_item(item_id):
    history_items = load_history()
    
    # Find the item to delete
    item_index = next((i for i, item in enumerate(history_items) if item['id'] == item_id), None)
    
    if item_index is None:
        return jsonify({'success': False, 'error': 'Item not found'})
    
    # Get the image path to delete the file
    item = history_items[item_index]
    image_path = item.get('image_path', '')
    
    # Remove from history
    history_items.pop(item_index)
    save_history(history_items)
    
    # Delete image file if it exists and is in uploads folder
    if image_path and os.path.exists(image_path) and app.config['UPLOAD_FOLDER'] in image_path:
        try:
            os.remove(image_path)
        except Exception as e:
            print(f"Error deleting image file: {str(e)}")
    
    return jsonify({'success': True})

@app.route('/save_detection', methods=['POST'])
def save_detection():
    data = request.json
    
    if not data or 'result' not in data or 'image' not in data:
        return jsonify({'success': False, 'error': 'Invalid data'})
    
    try:
        # Generate unique ID
        item_id = str(uuid.uuid4())
        
        # Save the image
        img_data = base64.b64decode(data['image'].split(',')[1])
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{item_id}.png")
        
        with open(img_path, 'wb') as f:
            f.write(img_data)
        
        # Get result data
        result = data['result']
        
        # Determine confidence color
        confidence = result['confidence'] * 100
        if confidence > 80:
            confidence_color = '#27ae60'
        elif confidence > 60:
            confidence_color = '#f39c12'
        else:
            confidence_color = '#e74c3c'
        
        # Create history item
        history_item = {
            'id': item_id,
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'disease_name': result['disease_name'],
            'confidence': confidence,
            'confidence_color': confidence_color,
            'plant_type': result['disease_name'].split(' ')[0],  # Extract plant type from disease name
            'description': result['info']['description'],
            'symptoms': result['info']['symptoms'],
            'treatment': result['info']['treatment'],
            'prevention': result['info']['prevention'],
            'image_path': img_path
        }
        
        # Add to history
        history = load_history()
        history.append(history_item)
        save_history(history)
        
        return jsonify({'success': True, 'id': item_id})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read image file
        img = Image.open(file.stream)
        
        # Resize to 480x480 if needed
        if img.size != (480, 480):
            img = ImageOps.fit(img, (480, 480), Image.LANCZOS)
        
        # Save the original image temporarily
        img_io = io.BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        
        # Convert to base64 for display
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        # Detect disease
        result = detector.detect_disease(img)
        
        if result is None:
            return jsonify({
                'success': False,
                'image': img_base64,
                'error': 'No disease detected or error in processing'
            })
        
        # Create a copy of the image to draw bounding boxes
        img_with_boxes = img.copy()
        img_with_boxes = np.array(img_with_boxes)
        
        # Convert to BGR for OpenCV
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        
        # Draw bounding box
        if 'bounding_box' in result:
            x1, y1, x2, y2 = [int(coord) for coord in result['bounding_box']]
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            confidence = result['confidence'] * 100
            label = f"{result['disease_name']}: {confidence:.1f}%"
            cv2.putText(img_with_boxes, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert back to RGB for PIL
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        img_with_boxes = Image.fromarray(img_with_boxes)
        
        # Save the image with boxes
        img_boxes_io = io.BytesIO()
        img_with_boxes.save(img_boxes_io, format='PNG')
        img_boxes_io.seek(0)
        
        # Convert to base64 for display
        img_boxes_base64 = base64.b64encode(img_boxes_io.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'image_with_boxes': img_boxes_base64,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Get base64 image from request
        image_data = request.json.get('image', '')
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Resize to 480x480 if needed
        if img.size != (480, 480):
            img = ImageOps.fit(img, (480, 480), Image.LANCZOS)
        
        # Detect disease
        result = detector.detect_disease(img)
        
        # Re-encode image for response
        img_io = io.BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        if result is None:
            return jsonify({
                'success': False,
                'image': img_base64,
                'error': 'No disease detected or error in processing'
            })
        
        # Create a copy of the image to draw bounding boxes
        img_with_boxes = img.copy()
        img_with_boxes = np.array(img_with_boxes)
        
        # Convert to BGR for OpenCV
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        
        # Draw bounding box
        if 'bounding_box' in result:
            x1, y1, x2, y2 = [int(coord) for coord in result['bounding_box']]
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            confidence = result['confidence'] * 100
            label = f"{result['disease_name']}: {confidence:.1f}%"
            cv2.putText(img_with_boxes, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert back to RGB for PIL
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        img_with_boxes = Image.fromarray(img_with_boxes)
        
        # Save the image with boxes
        img_boxes_io = io.BytesIO()
        img_with_boxes.save(img_boxes_io, format='PNG')
        img_boxes_io.seek(0)
        
        # Convert to base64 for display
        img_boxes_base64 = base64.b64encode(img_boxes_io.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'image_with_boxes': img_boxes_base64,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model at startup
    detector.load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)