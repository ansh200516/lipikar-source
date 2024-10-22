const availableTextRecognizers = [
  "Assamese",
  "Bengali",
  "English",
  "Gujarati",
  "Hindi",
  "Kannada",
  "Malayalam",
  "Manipuri",
  "Oriya",
  "Punjabi",
  "Santali",
  "Tamil",
  "Telugu",
  "Urdu",
];

const lttrMap = {
  Assamese: "assamese_iitd",
  Bengali: "bengali_iitd",
  English: "eng",
  Gujarati: "gujarati_iitd",
  Hindi: "hindi_iitd",
  Kannada: "kannada_iitd",
  Malayalam: "malayalam_iitd",
  Manipuri: "manipuri_meitei_iitd",
  Oriya: "oriya_iitd",
  Punjabi: "punjabi_iitd",
  Santali: "santali_ol_chiki_iitd",
  Tamil: "tamil_iitd",
  Telugu: "telugu_iitd",
  Urdu: "urdu_iitd",
};

const languageToTextRecognizer = (language) => {
  return lttrMap[language] || 'eng';  // Default to English ('eng') if no match
};

// Export the function
module.exports = { languageToTextRecognizer };