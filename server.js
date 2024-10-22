const express = require('express');
const multer = require('multer');
const tesseract = require('tesseract.js');
const cors = require('cors');
require('dotenv').config(); // For loading environment variables

const app = express();
app.use(cors()); // Enable CORS for frontend-backend communication

// Set up multer for handling file uploads
const upload = multer({ dest: 'uploads/' });

// Test endpoint to ensure the server is working
app.get('/api/ocr/services/config', (req, res) => {
  res.json({
    message: "OCR service is running",
    supportedLanguages: [
      "English", "Hindi", "Urdu", "Tamil", "Telugu", 
      "Kannada", "Bengali", "Gujarati", "Odia", "Marathi"
    ]
  });
});

// OCR processing route
app.post('/api/ocr/services/new-ocr', upload.single('file'), (req, res) => {
  const file = req.file; // Get the uploaded file
  const { document_parser, text_recognizer } = req.query; // Get query parameters

  if (!file) {
    return res.status(400).send({ error: 'No file uploaded.' });
  }

  // Perform OCR using Tesseract with the selected recognizer (language)
  tesseract.recognize(file.path, text_recognizer || 'eng') // Default to English if not provided
    .then(result => {
      res.json({
        success: true,
        result: {
          detections: result.data.text.split('\n') // Split text into lines for easier processing
        }
      });
    })
    .catch(error => {
      console.error('OCR error:', error);
      res.status(500).send({ error: 'OCR processing failed.' });
    });
});

// Start the server
const port = process.env.PORT || 8000;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});