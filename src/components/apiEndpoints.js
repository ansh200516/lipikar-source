const apiBaseUrl = "http://localhost:8000/api/ocr";

const apiEndpoints = {
  services: {
    getConfig: () => {
      return `${apiBaseUrl}/services/config`;
    },
    newOcr: (documentParser, textRecognizer) => {
      return `${apiBaseUrl}/services/new-ocr/?document_parser=${documentParser}&text_recognizer=${textRecognizer}`;
    },
  },
};

module.exports = apiEndpoints;
