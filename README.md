# ArtMuse

ArtMuse is an AI-powered art assistant that helps users generate, analyze, and transform visual concepts and images. It uses natural language processing and advanced image processing techniques to understand user requests and produce creative results.

## Features

- Generate visual concepts from text descriptions
- Extract style and information from existing images
- Transform images based on text prompts
- Generate images from text descriptions
- Answer general queries related to art and visual concepts

## Components

1. **ArtMuse**: The main class that orchestrates the entire process.
2. **ConversationManager**: Manages the conversation flow between the user and ArtMuse.
3. **MemoryManager**: Handles storage and retrieval of conversation history.
4. **ImageProcessor**: Processes images using various AI models.

## Dependencies

Python 3.11

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your Groq API key as an environment variable:
   ```
   export GROQ_API_KEY=your_api_key_here
   ```

3. Run the main script:
   ```
   python main.py
   ```

## Usage

After running the main script, you can interact with ArtMuse through the command line. Here are some example interactions:

1. Generate a visual concept:
   ```
   You: Create a visual concept for a futuristic cityscape
   ```

2. Analyze an existing image:
   ```
   You: What can you tell me about the style of image0.jpeg?
   ```

3. Transform an image:
   ```
   You: Transform image0.jpeg into a watercolor painting
   ```

4. Generate an image from text:
   ```
   You: Generate an image of a serene mountain landscape at sunset
   ```

5. Ask a general art-related question:
   ```
   You: What are the key characteristics of impressionist painting?
   ```

To exit the program, simply type 'exit'.

## Logging

ArtMuse logs its activities in the `artmuse.log` file. You can check this file for debugging information and to understand the system's decision-making process.

## Contributing

Contributions to ArtMuse are welcome! Please feel free to submit pull requests or open issues to discuss potential improvements or report bugs.

## License

GPL 3