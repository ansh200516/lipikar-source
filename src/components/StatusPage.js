// StatusPage.js
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Container,
  Typography,
  CircularProgress,
  Box,
  Button,
  Paper,
} from '@mui/material';
import { motion } from 'framer-motion';
import { CheckCircleOutline, ErrorOutline } from '@mui/icons-material';
import './StatusPage.css';

const StatusPage = () => {
  const { id: inferenceId } = useParams();
  const [status, setStatus] = useState('Processing...');
  const [ocrLink, setOcrLink] = useState('');
  const BACKEND_URL = 'http://maverick.cse.iitd.ac.in:5000';

  useEffect(() => {
    const intervalId = setInterval(async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/status?inference_id=${inferenceId}`);
        const result = await response.json();

        if (result.success && result.status === 'done') {
          clearInterval(intervalId);
          setStatus('Completed');
          setOcrLink(`${BACKEND_URL}${result.result.ocr_result_link}`);
        } else if (result.status === 'error') {
          clearInterval(intervalId);
          setStatus('Failed');
        }
      } catch (error) {
        console.error('Error polling for OCR result:', error);
        clearInterval(intervalId);
        setStatus('Error fetching status');
      }
    }, 3000);

    return () => clearInterval(intervalId);
  }, [inferenceId, BACKEND_URL]);

  const getStatusIcon = () => {
    if (status === 'Completed') {
      return <CheckCircleOutline className="status-icon success" />;
    } else if (status === 'Failed' || status === 'Error fetching status') {
      return <ErrorOutline className="status-icon error" />;
    } else {
      return <CircularProgress size={80} />;
    }
  };

  return (
    <Container maxWidth="sm" className="status-container">
      <Paper elevation={3} className="status-paper">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
          className="status-content"
        >
          {getStatusIcon()}
          <Typography variant="h4" style={{ marginTop: '20px' }}>
            OCR Status: {status}
          </Typography>
          {ocrLink && (
            <div style={{ marginTop: '30px' }}>
              <Button
                variant="contained"
                color="primary"
                href={ocrLink}
                target="_blank"
                rel="noopener noreferrer"
              >
                View OCR Result
              </Button>
            </div>
          )}
          <div style={{ marginTop: '30px' }}>
            <Button
              variant="outlined"
              color="secondary"
              onClick={() => {
                window.location.href = '/';
              }}
            >
              Back to Home
            </Button>
          </div>
        </motion.div>
      </Paper>
    </Container>
  );
};

export default StatusPage;