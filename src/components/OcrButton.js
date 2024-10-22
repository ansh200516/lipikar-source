// OcrButton.js
import React from 'react';
import { Button } from '@mui/material';
import { motion } from 'framer-motion';

const OcrButton = ({ handleSubmit }) => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.5, duration: 0.8 }}
    >
      <Button
        variant="contained"
        color="primary"
        onClick={handleSubmit}
        style={{
          marginTop: '20px',
          padding: '10px 30px',
          background: 'linear-gradient(90deg, #007bff, #0056b3)',
          transition: 'background 0.3s ease-in-out',
        }}
        onMouseEnter={(e) => {
          e.target.style.background = 'linear-gradient(90deg, #0056b3, #007bff)';
        }}
        onMouseLeave={(e) => {
          e.target.style.background = 'linear-gradient(90deg, #007bff, #0056b3)';
        }}
      >
        Start OCR!
      </Button>
    </motion.div>
  );
};

export default OcrButton;