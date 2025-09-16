from flask import Flask, render_template, request, jsonify, send_from_directory
from fraud_detector import assess_ethereum_address
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/api/check_address', methods=['POST'])
def check_address():
    """API endpoint to check Ethereum address risk"""
    try:
        data = request.get_json()
        address = data.get('address', '')
        
        # Validate address format
        if not address.startswith('0x') or len(address) != 42:
            return jsonify({
                'error': 'Invalid Ethereum address format. Should start with "0x" and be 42 characters long.'
            }), 400
        
        # Get fraud probability from your model
        fraud_probability = assess_ethereum_address(address)
        
        # Convert to percentage
        fraud_percentage = fraud_probability * 100
        
        # Determine risk level and color
        if fraud_probability > 0.7:
            risk_level = "High Risk"
            color = "#ff4444"
        elif fraud_probability >= 0.4:
            risk_level = "Medium Risk"
            color = "#ffaa44"
        else:
            risk_level = "Low/No Risk"
            color = "#44cc44"
            
        return jsonify({
            'address': address,
            'fraud_probability': round(fraud_percentage, 2),
            'risk_level': risk_level,
            'color': color
        })
        
    except Exception as e:
        return jsonify({
            'error': f"Error processing address: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)