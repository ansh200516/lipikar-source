// App.js
import React, { useState, useEffect } from 'react';
import {
  createTheme,
  ThemeProvider,
  CssBaseline,
  Container,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  LinearProgress,
  Grid,
} from '@mui/material';
import FileUpload from './components/FileUpload';
import LanguageSelector from './components/LanguageSelector';
import OcrButton from './components/OcrButton';
import LoadingScreen from './components/LoadingScreen';
import { motion } from 'framer-motion';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import StatusPage from './components/StatusPage';
import './App.css';

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [language, setLanguage] = useState('');
  const [outputType, setOutputType] = useState('');
  const [files, setFiles] = useState([]);
  const [inferenceId, setInferenceId] = useState(null);
  const [estimatedTime, setEstimatedTime] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [copySuccess, setCopySuccess] = useState('');
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');

  const BACKEND_URL = 'http://maverick.cse.iitd.ac.in:5000'; // Update with the correct backend URL and port

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#007bff',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#fff',
      },
      text: {
        primary: darkMode ? '#fff' : '#000',
      },
    },
  });

  const handleThemeChange = () => {
    setDarkMode(!darkMode);
  };

  const handleFileUpload = (uploadedFiles) => {
    setFiles(uploadedFiles);
  };

  const handleSubmit = async () => {
    if (!files.length || !language || !outputType) {
      alert('Please upload files, select a language, and choose an output type!');
      return;
    }

    setIsLoading(true);

    const formData = new FormData();

    // Check if the first file is a PDF
    if (files.length === 1 && files[0].type === 'application/pdf') {
      formData.append('file_type', 'pdf');
      formData.append('files', files[0]); // Append the PDF file
    } else {
      formData.append('file_type', 'images');
      files.forEach((file) => {
        formData.append('files', file); // Append each image file
      });
    }

    formData.append('language', language); // Language code
    formData.append('output_type', outputType); // Selected output type

    // Log FormData entries for debugging
    for (let pair of formData.entries()) {
      console.log(`${pair[0]}:`, pair[1]);
    }

    try {
      const response = await fetch(`${BACKEND_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      if (result.success) {
        setInferenceId(result.inference_id);
        setEstimatedTime(result.estimated_time); // Capture estimated time
        setIsLoading(false); // Stop loading spinner
        setStatus('processing');
        // Start polling for progress
        pollProgress(result.inference_id);
      } else {
        alert('File upload failed. Please try again.');
        setIsLoading(false);
      }
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Error uploading files. Please try again.');
      setIsLoading(false);
    }
  };

  // Function to format estimated time
  const formatEstimatedTime = (seconds) => {
    if (seconds < 60) {
      return `${seconds} seconds`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes} minute${minutes > 1 ? 's' : ''}${
        remainingSeconds > 0 ? ` and ${remainingSeconds} second${remainingSeconds > 1 ? 's' : ''}` : ''
      }`;
    }
  };

  // Polling function to check progress
  const pollProgress = (inferenceId) => {
    const intervalId = setInterval(async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/status?inference_id=${inferenceId}`);
        const result = await response.json();
        if (result.success) {
          setProgress(result.progress);
          setStatus(result.status);
          if (result.progress >= 100 || result.status === 'done' || result.status === 'error') {
            clearInterval(intervalId);
          }
        } else {
          clearInterval(intervalId);
        }
      } catch (error) {
        console.error('Error fetching progress:', error);
        clearInterval(intervalId);
      }
    }, 1000); // Poll every 3 seconds
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {isLoading && <LoadingScreen />}
      <Router>
        <Routes>
          <Route
            path="/"
            element={
              <div className="app-container">
                <div className="top-bar">
                <img src="/lipi_logo.png" alt="Logo" />
                </div>
                <Container maxWidth="lg" style={{ padding: '100px 0' }}>
                  <Grid container spacing={2}>
                    <Grid item xs={12} style={{ textAlign: 'right' }}>
                      <FormControlLabel
                        control={<Switch checked={darkMode} onChange={handleThemeChange} />}
                        label="Dark Mode"
                      />
                    </Grid>

                    <Grid item xs={12}>
                      <motion.h1
                        initial={{ opacity: 0, y: -50 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 1 }}
                        style={{
                          background: 'linear-gradient(90deg, #ff0066, #6600ff)',
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                          textAlign: 'center',
                        }}
                      >
                        Upload Files for OCR
                      </motion.h1>
                    </Grid>

                    <Grid item xs={12}>
                      <FileUpload onFileUpload={handleFileUpload} />
                    </Grid>

                    {/* Wrap the LanguageSelector and Select in a container Grid */}
                    <Grid item xs={12}>
                      <Grid container spacing={2} alignItems="center">
                        <Grid item xs={12} sm={6}>
                          <LanguageSelector language={language} setLanguage={setLanguage} />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <Select
                            value={outputType}
                            onChange={(e) => setOutputType(e.target.value)}
                            displayEmpty
                            fullWidth
                            // Ensure consistent height
                            variant="outlined"
                          >
                            <MenuItem value="" disabled>
                              Select Output Type
                            </MenuItem>
                            <MenuItem value="text">Text</MenuItem>
                            <MenuItem value="pdf">PDF</MenuItem>
                          </Select>
                        </Grid>
                      </Grid>
                    </Grid>

                    <Grid item xs={12} style={{ textAlign: 'center' }}>
                      <OcrButton handleSubmit={handleSubmit} />
                    </Grid>

                    {/* Display the status URL, progress bar, and copy option */}
                    {inferenceId && (
                      <>
                        <Grid item xs={12}>
                          <p style={{ textAlign: 'center' }}>
                            Your OCR process has started! You can check the status at the following URL:
                          </p>
                        </Grid>
                        <Grid item xs={12}>
                          <Grid container justifyContent="center" alignItems="center" spacing={2}>
                            <Grid item xs={12} sm={8}>
                              <TextField
                                variant="outlined"
                                value={`${window.location.origin}/status/${inferenceId}`}
                                InputProps={{
                                  readOnly: true,
                                }}
                                fullWidth
                              />
                            </Grid>
                            <Grid item xs={12} sm="auto">
                              <Button
                                variant="contained"
                                color="primary"
                                onClick={() => {
                                  navigator.clipboard.writeText(
                                    `${window.location.origin}/status/${inferenceId}`
                                  );
                                  setCopySuccess('Copied!');
                                  setTimeout(() => setCopySuccess(''), 2000);
                                }}
                                fullWidth
                              >
                                Copy
                              </Button>
                            </Grid>
                          </Grid>
                        </Grid>
                        {copySuccess && (
                          <Grid item xs={12}>
                            <p style={{ color: 'green', textAlign: 'center' }}>{copySuccess}</p>
                          </Grid>
                        )}

                        {/* Display the estimated time */}
                        {estimatedTime && (
                          <Grid item xs={12}>
                            <p style={{ marginTop: '20px', textAlign: 'center' }}>
                              Estimated time to complete OCR:{' '}
                              <strong>{formatEstimatedTime(estimatedTime)}</strong>
                            </p>
                            <p style={{ fontSize: 'small', color: 'gray', textAlign: 'center' }}>
                              Note: This is an estimated time. Actual processing time may vary.
                            </p>
                          </Grid>
                        )}

                        {/* Display the progress bar */}
                        <Grid item xs={12}>
                          <div style={{ marginTop: '20px' }}>
                            <LinearProgress variant="determinate" value={progress} />
                            <p style={{ textAlign: 'center' }}>
                              {status === 'done'
                                ? 'Processing complete!'
                                : status === 'error'
                                ? 'An error occurred during processing.'
                                : `Processing: ${progress}%`}
                            </p>
                          </div>
                        </Grid>
                      </>
                    )}
                  </Grid>
                </Container>
              </div>
            }
          />
          <Route path="/status/:id" element={<StatusPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;