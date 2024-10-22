import fetch, { FormData, fileFromSync } from "node-fetch";
import path from "path";
import apiEndpoints from "./apiEndpoints.js";
import { languageToTextRecognizer } from "../../config.js";

const getParserForRecognizer = (tr) => {
  if (tr === "urdu_iitd") {
    return "tess_line_level";
  }
  return "tess_word_level";
};

// Get OCR service configuration
async function getServiceConfig() {
  try {
    const endpoint = apiEndpoints.services.getConfig();

    console.log("Requesting OCR from: ", endpoint);

    const response = await fetch(endpoint, {
      method: "GET",
      headers: {
        "X-LIPIKAR-SERVICE-API-KEY": process.env.SERVICE_API_KEY,  // Use your API key if required
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Error response body:", errorText);
      throw new Error("Failed to fetch service configuration.");
    }

    const responseData = await response.json();
    return responseData;
  } catch (error) {
    console.log(error);
    return false;
  }
}

// Process OCR for the uploaded image
async function getDetectionsForImage(imagePath, documentParser, textRecognizer) {
  try {
    const endpoint = apiEndpoints.services.newOcr(documentParser, textRecognizer);

    console.log("Requesting OCR from: ", endpoint);

    const formData = new FormData();
    formData.append("file", fileFromSync(imagePath, "image/jpeg"));

    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
      headers: {
        "X-LIPIKAR-SERVICE-API-KEY": process.env.SERVICE_API_KEY,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Error response body:", errorText);
      throw new Error("Failed to process OCR.");
    }

    const responseData = await response.json();
    return responseData.result.detections;
  } catch (error) {
    console.log(error);
    return false;
  }
}

export { getServiceConfig, getDetectionsForImage };