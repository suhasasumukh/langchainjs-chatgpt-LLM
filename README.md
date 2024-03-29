# LangChain.js LLM Template

This is a LangChain LLM template that allows you to train your custom AI model on any data you want.
Now supports gpt-3.5 and gpt-4!

## Setup
1. Provide all the information you want your LLM to be trained on in the `training` directory in markdown files.  Folder depth doesn't matter.
2. Add your OpenAI API key in environment vars via the kay `OPENAI_API_KEY`.
3. Run `yarn train` or `npm train` to set up your vector store.
4. Modify the base prompt in `lib/basePrompt.js`
5. Run index.js, and start playing around with it!

## Overview of this Model

LangChain LLM template, you have full control over the training process, allowing you to fine-tune your model to suit your specific needs and applications. Whether you're looking to create a language model that generates creative content, provides intelligent recommendations, or assists with complex tasks, this template offers the flexibility to mold the AI model to your requirements.

By utilizing gpt-3.5 or gpt-4, you benefit from state-of-the-art language processing capabilities, enabling your custom AI model to deliver unparalleled performance. Seamlessly integrate the LangChain LLM template into your projects, applications, or research endeavors, and witness the remarkable outcomes of your trained AI model.

## What is it capable of doing?

The template's flexibility allows for training on diverse datasets, enabling the customization of models for specific use cases. It can be applied to chatbots, virtual assistants, language translation, sentiment analysis, content generation, and much more. Developers can fine-tune the models to optimize performance and align with their desired outcomes.

## Getting Started
<br>
<p>To get started with langchainjs-chatgpt-LLM, follow these steps:</p>

#### Clone the repository:
```bash
git clone https://github.com/suhasasumukh/langchainjs-chatgpt-LLM.git
```

#### Install the required dependencies:
```bash
npm install
```
#### Configure the model parameters and settings as per your requirements

#### Run the application:
```bash
npm start
```
