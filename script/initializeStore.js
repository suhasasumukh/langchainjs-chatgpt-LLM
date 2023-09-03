import glob from 'glob';
import fs from 'fs'
import { CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { config } from 'dotenv';
// runner for cli
import { execSync } from 'child_process';

config();

const client = new PineconeClient();
let pineconeIndex;
if(process.env.PINECONE=="true") {
await client.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});
console.log("Pinecone Initialized");
pineconeIndex = client.Index(process.env.PINECONE_INDEX);
}



const data = [];
const pdfs = await new Promise((resolve, reject) =>
  glob("training/**/*.pdf", (err, files) => err ? reject(err) : resolve(files))
);

for(const pdf of pdfs) {
  // convert to txt by running the cli command `pdftotext <filename>.pdf`
  execSync(`pdftotext ${pdf}`);
}

const files = await new Promise((resolve, reject) =>
  glob("training/**/*.txt", (err, files) => err ? reject(err) : resolve(files))
);

if(files.length == 0) {
  throw new Error("No training files found.  Please add some training files to the training directory.");
}

for (const file of files) {
  data.push(fs.readFileSync(file, 'utf-8'));
}

console.log(`Added ${files.length} files to data.  Splitting text into chunks...`);

const textSplitter = new CharacterTextSplitter({
  chunkSize: 2000,
  chunkOverlap: 100,
  separator: " "
});

let docs = [];
for (const d of data) {
  const docOutput = textSplitter.splitText(d);
  if(docOutput.length > 1) {
  docs = [...docs, ...docOutput];
  } else {
  docs.push(d);
  }
}


console.log("Initializing Store...");

let store;
try {
if(pineconeIndex) {
  await PineconeStore.fromTexts(
    docs,
    docs.map((doc,i) => {return {id: i+1}}),
    new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY
    }),
    {
      pineconeIndex,
    }
  )
} else {
 store = await HNSWLib.fromTexts(
  docs,
  docs.map((doc,i) => {return {id: i+1}}),
  new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY
  })
)
}

console.clear();
console.log("Saving Vectorstore");

if(!pineconeIndex) {
  await store.save("vectorStore");
}

console.clear();
console.log("VectorStore saved");
} catch(e) {
  console.log(e);
  process.exit();
}
