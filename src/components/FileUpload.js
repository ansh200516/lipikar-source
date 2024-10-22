// FileUpload.js
import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import './FileUpload.css';
import * as pdfjsLib from 'pdfjs-dist/build/pdf';
pdfjsLib.GlobalWorkerOptions.workerSrc = `${process.env.PUBLIC_URL}/pdf.worker.min.js`;

const FileUpload = ({ onFileUpload }) => {
  const [files, setFiles] = useState([]); // Holds image files or a single PDF file
  const [previews, setPreviews] = useState([]); // Stores previews (images or PDF pages)
  const [uploadType, setUploadType] = useState('image'); // 'image' or 'pdf'
  const [currentIndex, setCurrentIndex] = useState(0); // For navigating previews

  const { getRootProps, getInputProps } = useDropzone({
    accept: uploadType === 'image' ? 'image/*' : 'application/pdf',
    multiple: uploadType === 'image', // Multiple files only for images
    onDrop: (acceptedFiles) => {
      if (uploadType === 'image') {
        const uploadedFiles = [...files, ...acceptedFiles].slice(0, 50); // Limit to 50 images
        setFiles(uploadedFiles);
        const newPreviews = uploadedFiles.map((file) => URL.createObjectURL(file));
        setPreviews(newPreviews);
        setCurrentIndex(0);
        onFileUpload(uploadedFiles); // Pass image files to App.js
      } else if (uploadType === 'pdf') {
        const pdfFile = acceptedFiles[0];
        if (pdfFile) {
          setFiles([pdfFile]); // Store the original PDF file
          renderPdfForPreview(pdfFile);
          setCurrentIndex(0);
          onFileUpload([pdfFile]); // Pass PDF file to App.js
        }
      }
    },
  });

  // Render PDF pages for preview
  const renderPdfForPreview = async (pdfFile) => {
    const fileReader = new FileReader();
    fileReader.onload = async function (e) {
      const typedArray = new Uint8Array(e.target.result);
      try {
        const pdf = await pdfjsLib.getDocument({ data: typedArray }).promise;
        const totalPages = pdf.numPages;
        const renderedPages = [];
        for (let i = 1; i <= totalPages && i <= 50; i++) { // Limit to 50 pages
          const dataURL = await renderPdfPageToImage(pdf, i);
          renderedPages.push(dataURL);
        }
        setPreviews(renderedPages);
      } catch (error) {
        console.error('Error reading PDF:', error);
        alert('Failed to read PDF file.');
      }
    };
    fileReader.readAsArrayBuffer(pdfFile);
  };

  // Convert a single PDF page to an image data URL for preview
  const renderPdfPageToImage = async (pdf, pageNumber) => {
    const page = await pdf.getPage(pageNumber);
    const viewport = page.getViewport({ scale: 1.5 });
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.height = viewport.height;
    canvas.width = viewport.width;

    const renderContext = {
      canvasContext: context,
      viewport: viewport,
    };
    await page.render(renderContext).promise;
    const dataURL = canvas.toDataURL('image/jpeg'); // Get the image data URL
    return dataURL;
  };

  const handleUploadTypeChange = (type) => {
    setUploadType(type);
    setFiles([]); // Clear files when changing upload type
    setPreviews([]); // Clear previews
    setCurrentIndex(0);
  };

  const removeCurrentFile = () => {
    if (uploadType === 'image' && files.length > 0) {
      const updatedFiles = [...files];
      const updatedPreviews = [...previews];
      updatedFiles.splice(currentIndex, 1);
      updatedPreviews.splice(currentIndex, 1);
      setFiles(updatedFiles);
      setPreviews(updatedPreviews);
      setCurrentIndex((prev) => (prev > 0 ? prev - 1 : 0));
      onFileUpload(updatedFiles); // Update the files passed to App.js
    } else if (uploadType === 'pdf' && files.length > 0) {
      // Removing the PDF file
      setFiles([]);
      setPreviews([]);
      setCurrentIndex(0);
      onFileUpload([]); // Update the files passed to App.js
    }
  };

  const nextFile = () => {
    if (currentIndex < previews.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const previousFile = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const renderImagePreview = () => {
    if (previews.length === 0) return null;

    const currentImage = previews[currentIndex];
    const totalItems = previews.length;

    return (
      <div className="file-preview-wrapper">
        <div className="file-preview">
          <img
            src={currentImage}
            alt={`Preview ${currentIndex}`}
            className="preview-image"
          />
          <p className="image-count">
            {`Showing ${currentIndex + 1} of ${totalItems} ${uploadType === 'image' ? 'images' : 'pages'}`}
          </p>
          <div className="button-group">
            <button onClick={previousFile} disabled={currentIndex === 0} className="nav-button left">
              Previous
            </button>
            <button onClick={removeCurrentFile} className="remove-button center">
              Remove
            </button>
            <button
              onClick={nextFile}
              disabled={currentIndex === totalItems - 1}
              className="nav-button right"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <motion.div
      className="file-upload-container"
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.6 }}
    >
      <div className="upload-type-selector">
        <button
          onClick={() => handleUploadTypeChange('image')}
          className={`upload-type-button ${uploadType === 'image' ? 'active' : ''}`}
        >
          Upload Images
        </button>
        <button
          onClick={() => handleUploadTypeChange('pdf')}
          className={`upload-type-button ${uploadType === 'pdf' ? 'active' : ''}`}
        >
          Upload PDF
        </button>
      </div>

      <div {...getRootProps({ className: 'dropzone' })}>
        <input {...getInputProps()} />
        <p>
          {uploadType === 'image'
            ? 'Drag and drop up to 50 images'
            : 'Drag and drop a PDF (max 50 pages)'}
        </p>
      </div>

      {renderImagePreview()}
    </motion.div>
  );
};

export default FileUpload;