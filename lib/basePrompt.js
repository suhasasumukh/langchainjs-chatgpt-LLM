const basePrompt =
[`You are a helpful AI assistant.
Use the following pieces of MemoryContext to answer the human.
---
MemoryContext: {context}
---`, `Human: {prompt}`];

export default basePrompt;