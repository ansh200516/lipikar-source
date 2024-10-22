// LanguageSelector.js
import React from 'react';
import { MenuItem, Select } from '@mui/material';

const LanguageSelector = ({ language, setLanguage }) => {
  // List of supported languages with their codes
  const languages = [
    { name: 'English', code: 'english' },
    { name: 'Hindi', code: 'hindi' },
    { name: 'Bengali', code: 'bengali' },
    { name: 'Gujarati', code: 'gujarati' },
    { name: 'Kannada', code: 'kannada' },
    { name: 'Malayalam', code: 'malayalam' },
    { name: 'Marathi', code: 'marathi' },
    { name: 'Punjabi', code: 'punjabi' },
    { name: 'Tamil', code: 'tamil' },
    { name: 'Kashmiri', code: 'kashmiri' },
    { name: 'Maithili', code: 'maithili' },
    { name: 'Nepali', code: 'nepali' },
    { name: 'Konkani', code: 'konkani' },
    { name: 'Odia', code: 'odia' },
  ];

  return (
    <Select
      value={language}
      onChange={(e) => setLanguage(e.target.value)}
      displayEmpty
      fullWidth
      // Remove margin to match the Output Type Select
      variant="outlined"
    >
      <MenuItem value="" disabled>
        Select Language
      </MenuItem>
      {languages.map((lang) => (
        <MenuItem key={lang.code} value={lang.code}>
          {lang.name}
        </MenuItem>
      ))}
    </Select>
  );
};

export default LanguageSelector;