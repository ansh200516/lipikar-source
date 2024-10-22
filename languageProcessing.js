import { availableTextRecognizers, availableDocumentParsers, availableOutputFormats } from "./config.js";

function generateHelpMessage(availableTextRecognizers, availableDocumentParsers, availableOutputFormats) {
  // let helpMessage = `Send a photo with a 3 digit code to perform OCR. \n\n`;
  // helpMessage += "Languages: \n";
  // for (let i = 0; i < availableTextRecognizers.length; i++) {
  //   helpMessage += `    ${i + 1}) ${availableTextRecognizers[i]} \n`;
  // }
  // helpMessage += "Parsers: \n";
  // for (let i = 0; i < availableDocumentParsers.length; i++) {
  //   helpMessage += `    ${i + 1}) ${availableDocumentParsers[i]} \n`;
  // }
  // helpMessage += "Output Formats: \n";
  // for (let i = 0; i < availableOutputFormats.length; i++) {
  //   helpMessage += `    ${i + 1}) ${availableOutputFormats[i]} \n`;
  // }

  // return helpMessage;

  let helpMessage = "Send a photo with language caption to perform OCR. \n\n";
  helpMessage += "Languages: \n";
  for (let i = 0; i < availableTextRecognizers.length; i++) {
    helpMessage += `    ${i + 1}) ${availableTextRecognizers[i]} \n`;
  }

  return helpMessage;
}

function parseOcrConfigCode(ocrConfigCode) {
  ocrConfigCode = ocrConfigCode.toLowerCase();

  let identifiedTextRecognizer = "";

  const lcAvailableTextRecognizers = availableTextRecognizers;
  for (const tr of lcAvailableTextRecognizers) {
    if (tr.toLowerCase() === ocrConfigCode) {
      identifiedTextRecognizer = tr;
    }
  }

  if (identifiedTextRecognizer === "") {
    return {
      success: false,
      error: "Undefined language",
    };
  }

  // // TODO: return types
  // if (ocrConfigCode.length !== 3) {
  //   return {
  //     success: false,
  //     error: "OCR Config Code should be exactly 3 characters long.",
  //   };
  // }

  // const textRecognizerIndex = parseInt(ocrConfigCode[0]) - 1;
  // if (textRecognizerIndex < 0 || textRecognizerIndex >= availableTextRecognizers.length) {
  //   return {
  //     success: false,
  //     error: "Invalid Language number.",
  //   };
  // }

  // const documentParserIndex = parseInt(ocrConfigCode[1]) - 1;
  // if (documentParserIndex < 0 || documentParserIndex >= availableDocumentParsers.length) {
  //   return {
  //     success: false,
  //     error: "Invalid Parser number.",
  //   };
  // }

  // const outputFormatIndex = parseInt(ocrConfigCode[2]) - 1;
  // if (outputFormatIndex < 0 || outputFormatIndex >= availableOutputFormats.length) {
  //   return {
  //     success: false,
  //     error: "Invalid Output Format number.",
  //   };
  // }

  return {
    success: true,
    result: {
      textRecognizer: identifiedTextRecognizer,
      documentParser: "foo", //availableDocumentParsers[documentParserIndex],
      outputFormat: "bar", // availableOutputFormats[outputFormatIndex],
    },
  };
}

const defaultHelpMessage = generateHelpMessage(
  availableTextRecognizers,
  availableDocumentParsers,
  availableOutputFormats
);

const helpCommandMessage = "Send /help for help.";
const invalidInstructionMessage = "Send a photo with a language name caption to start OCR. \n\n" + helpCommandMessage;
const photoButNoConfigInstructionMessage =
  "Please caption the photo with a language name to perform OCR. Send /help for help.";

export {
  generateHelpMessage,
  parseOcrConfigCode,
  defaultHelpMessage,
  invalidInstructionMessage,
  helpCommandMessage,
  photoButNoConfigInstructionMessage,
};
