import { LLM } from "llama-node";
import { LLamaCpp } from "llama-node/dist/llm/llama-cpp.js";
import express from "express";
import path from "path";

const app = express();
const llama = new LLM(LLamaCpp);

app.listen(3000, () => {
  console.log("server started http://127.0.0.1:3000");
});

app.get("/", async (req, res) => {
  res.sendFile(path.join(path.resolve(), "/index.html"));
});

app.get("/ask", async (req, res) => {
  const userQuestion = "Who are you?";
  res.writeHead(200, { "content-type": "text/html" });
  await llama.load({
    modelPath: "model/airoboros-13b-gpt4.ggmlv3.q4_0.bin",
    enableLogging: false,
    nCtx: 1024,
    seed: 0,
    f16Kv: false,
    logitsAll: false,
    vocabOnly: false,
    useMlock: false,
    embedding: false,
    useMmap: true,
    nGpuLayers: 0,
  });
  await llama.createCompletion(
    {
      prompt: userQuestion,
      nThreads: 4,
      nTokPredict: 2048,
      topk: 40,
      topP: 0.1,
      temp: 0.3,
      repeatPenalty: 1,
    },
    (response) => {
      res.write(response.token);
      res.end();
    }
  );
});
