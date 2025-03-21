// This module handles the content display (projects and blog posts)
document.addEventListener('DOMContentLoaded', () => {
    const mainContent = document.getElementById('mainContent');
    const contentView = document.getElementById('contentView');
    const contentContainer = document.getElementById('contentContainer');
    const backButton = document.getElementById('backButton');
    
    // Database of content
    const contentDatabase = {
        project: {
            project1: {
                title: "Natural Language Processing API",
                date: "October 2023",
                tags: ["Python", "Flask", "HuggingFace", "NLP", "API"],
                content: `
                    <div class="content-body">
                        <p>This project is a robust API for text analysis, sentiment detection, and language translation using transformer models from the HuggingFace ecosystem.</p>
                        
                        <h2>Project Overview</h2>
                        <p>The Natural Language Processing (NLP) API provides a set of endpoints that allow developers to integrate advanced text processing capabilities into their applications without needing to understand the underlying machine learning models.</p>
                        
                        <p>The API supports the following functions:</p>
                        <ul>
                            <li>Text classification (sentiment analysis, topic detection)</li>
                            <li>Named entity recognition</li>
                            <li>Language translation between 100+ languages</li>
                            <li>Text summarization</li>
                            <li>Question answering</li>
                        </ul>
                        
                        <h2>Technical Architecture</h2>
                        <p>The API is built using Flask and integrates with HuggingFace's Transformers library to provide state-of-the-art NLP capabilities. The architecture follows a microservices approach, with each NLP function running as a separate service that can be scaled independently based on usage patterns.</p>
                        
                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <pre><code class="language-python">from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get('text', '')
    
    // Initialize sentiment analysis pipeline
    sentiment_analyzer = pipeline('sentiment-analysis')
    
    // Analyze the text
    result = sentiment_analyzer(text)
    
    return jsonify({
        'text': text,
        'sentiment': result[0]['label'],
        'score': result[0]['score']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)</code></pre>
                        </div>
                        
                        <h2>Performance Optimization</h2>
                        <p>To ensure quick response times, the API implements several optimizations:</p>
                        <ul>
                            <li>Model caching to avoid reloading models for each request</li>
                            <li>Batch processing for handling multiple inputs efficiently</li>
                            <li>Request queuing for handling traffic spikes</li>
                            <li>Model quantization to reduce memory footprint</li>
                        </ul>
                        
                        <h2>API Documentation</h2>
                        <p>The API is fully documented using Swagger, allowing developers to quickly understand and integrate the various endpoints. Authentication is handled through API keys, with rate limiting applied based on the subscription tier.</p>
                        
                        <div class="interactive-demo">
                            <div class="interactive-demo-header">Try the Sentiment Analysis</div>
                            <div class="mb-3">
                                <textarea id="sentimentText" class="w-full p-2 border border-gray-300 rounded" rows="3" placeholder="Enter text to analyze sentiment...">I absolutely love this new product! It's changed the way I work.</textarea>
                            </div>
                            <button id="analyzeSentiment" class="px-4 py-2 bg-black text-white rounded hover:bg-gray-800">Analyze Sentiment</button>
                            <div id="sentimentResult" class="mt-3 p-3 bg-gray-50 rounded hidden"></div>
                        </div>
                        
                        <h2>Future Enhancements</h2>
                        <p>The roadmap for this project includes:</p>
                        <ul>
                            <li>Adding custom model fine-tuning through the API</li>
                            <li>Supporting more languages for low-resource regions</li>
                            <li>Implementing streaming responses for real-time applications</li>
                            <li>Adding multi-modal capabilities (text + image analysis)</li>
                        </ul>
                    </div>
                `
            },
            project2: {
                title: "Image Generation App",
                date: "August 2023",
                tags: ["React", "Node.js", "Stable Diffusion", "Deep Learning"],
                content: `
                    <div class="content-body">
                        <p>This web application generates creative images from text prompts using Stable Diffusion models, giving users an intuitive interface to create custom artwork without artistic skills.</p>
                        
                        <h2>Project Overview</h2>
                        <p>The Image Generation App allows users to create unique images by simply describing what they want to see. The application leverages the power of Stable Diffusion, a state-of-the-art latent diffusion model that can generate detailed images from text descriptions.</p>
                        
                        <h2>Key Features</h2>
                        <ul>
                            <li>Text-to-image generation with customizable parameters</li>
                            <li>Style transfer and image editing capabilities</li>
                            <li>Personal gallery to save and organize generated images</li>
                            <li>Image enhancement and upscaling</li>
                            <li>Sharing options for social media integration</li>
                        </ul>
                        
                        <h2>Technical Implementation</h2>
                        <p>The application is built with a React frontend and a Node.js backend, with the image generation powered by Stable Diffusion models running on GPU servers. The architecture is designed to be scalable, allowing for multiple simultaneous generation requests.</p>
                        
                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <pre><code class="language-javascript">// React component for text-to-image generation
function ImageGenerator() {
  const [prompt, setPrompt] = useState("");
  const [generatedImage, setGeneratedImage] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  
  async function generateImage() {
    if (!prompt.trim()) return;
    
    setIsGenerating(true);
    
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });
      
      const data = await response.json();
      setGeneratedImage(data.imageUrl);
    } catch (error) {
      console.error('Error generating image:', error);
    } finally {
      setIsGenerating(false);
    }
  }
  
  return (
    <div className="image-generator">
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Describe the image you want to generate..."
      />
      <button 
        onClick={generateImage}
        disabled={isGenerating}
      >
        {isGenerating ? 'Generating...' : 'Generate Image'}
      </button>
      
      {generatedImage && (
        <div className="result">
          <img src={generatedImage} alt="Generated from prompt" />
        </div>
      )}
    </div>
  );
}</code></pre>
                        </div>
                        
                        <h2>User Interface</h2>
                        <p>The user interface is designed to be intuitive and accessible, allowing both beginners and experienced users to create stunning images. The main generation screen provides a simple text input for the prompt, along with advanced options for those who want more control over the generation process.</p>
                        
                        <h2>Optimization Techniques</h2>
                        <p>To ensure fast generation times even under high load, the application employs several optimization techniques:</p>
                        <ul>
                            <li>Request queuing and prioritization</li>
                            <li>Model optimization through half-precision computation</li>
                            <li>Progressive image loading for immediate feedback</li>
                            <li>Load balancing across multiple GPU instances</li>
                        </ul>
                        
                        <div class="interactive-demo">
                            <div class="interactive-demo-header">Try a Sample Prompt</div>
                            <div class="mb-3">
                                <input type="text" id="imagePrompt" class="w-full p-2 border border-gray-300 rounded" placeholder="Enter a description..." value="A futuristic cityscape at sunset with flying cars">
                            </div>
                            <button id="generateImage" class="px-4 py-2 bg-black text-white rounded hover:bg-gray-800">Generate Image</button>
                            <div id="imageResult" class="mt-3 hidden">
                                <div class="animate-pulse bg-gray-200 rounded-lg h-60 w-full"></div>
                                <p class="text-center text-sm text-gray-500 mt-2">The actual API is not connected in this demo.</p>
                            </div>
                        </div>
                        
                        <h2>Future Development</h2>
                        <p>The roadmap for this project includes:</p>
                        <ul>
                            <li>Advanced editing tools for more precise image manipulation</li>
                            <li>Video generation from text descriptions</li>
                            <li>Custom style training for consistent artistic output</li>
                            <li>API access for developers to integrate image generation into their applications</li>
                        </ul>
                    </div>
                `
            },
            project3: {
                title: "AI Chatbot Assistant",
                date: "June 2023",
                tags: ["TensorFlow", "FastAPI", "Vue.js", "NLP"],
                content: `
                    <div class="content-body">
                        <p>This conversational AI assistant helps users find information and complete tasks through natural dialogue, leveraging advanced language models to understand context and provide relevant responses.</p>
                        
                        <h2>Project Overview</h2>
                        <p>The AI Chatbot Assistant is designed to provide a natural, conversational interface for users to interact with systems and retrieve information. Unlike traditional chatbots that follow rigid, pre-defined paths, this assistant uses contextual understanding to provide more helpful and natural responses.</p>
                        
                        <img src="https://images.unsplash.com/photo-1635002952774-0a116223378e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" alt="Chatbot interface visualization" class="rounded-lg shadow-md">
                        
                        <h2>Key Capabilities</h2>
                        <ul>
                            <li>Natural language understanding for complex queries</li>
                            <li>Context retention across multiple conversation turns</li>
                            <li>Integration with external systems (calendar, email, databases)</li>
                            <li>Multi-language support for global accessibility</li>
                            <li>Personality customization to match brand voice</li>
                        </ul>
                        
                        <h2>Technical Architecture</h2>
                        <p>The assistant is built using a three-layer architecture:</p>
                        <ol>
                            <li><strong>Understanding layer:</strong> Processes incoming messages using TensorFlow-based NLP models to extract intent, entities, and context.</li>
                            <li><strong>Reasoning layer:</strong> Decides how to respond to the user's query based on available information and services.</li>
                            <li><strong>Generation layer:</strong> Creates natural, human-like responses that address the user's needs.</li>
                        </ol>
                        
                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <pre><code class="language-python">from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf

app = FastAPI()

// Load NLP models
intent_model = tf.saved_model.load("./models/intent_classifier")
entity_model = tf.saved_model.load("./models/entity_recognizer")
response_model = tf.saved_model.load("./models/response_generator")

class Message(BaseModel):
    text: str
    user_id: str
    conversation_id: Optional[str] = None

class Response(BaseModel):
    text: str
    suggested_actions: List[str] = []
    confidence: float

@app.post("/api/chat", response_model=Response)
async def process_message(message: Message):
    // Extract intent and entities
    intent = intent_model(message.text)
    entities = entity_model(message.text)
    
    // Generate appropriate response
    response_text, confidence = response_model(
        message.text, intent, entities, message.user_id, message.conversation_id
    )
    
    // Get suggested next actions
    suggested_actions = get_suggested_actions(intent, entities)
    
    return Response(
        text=response_text,
        suggested_actions=suggested_actions,
        confidence=confidence
    )</code></pre>
                        </div>
                        
                        <h2>Frontend Implementation</h2>
                        <p>The frontend is implemented using Vue.js to create a responsive and accessible chat interface. The design follows best practices for conversational UIs, including:</p>
                        <ul>
                            <li>Clear message threading and conversation history</li>
                            <li>Typing indicators to show when the assistant is responding</li>
                            <li>Message status indicators (sent, delivered, read)</li>
                            <li>Rich message formatting for complex responses (cards, carousels, buttons)</li>
                        </ul>
                        
                        <h2>Data Privacy and Security</h2>
                        <p>The assistant is designed with privacy as a core principle:</p>
                        <ul>
                            <li>All conversation data is encrypted both in transit and at rest</li>
                            <li>User data retention policies are configurable</li>
                            <li>The system can be deployed on-premises for sensitive applications</li>
                            <li>User consent mechanisms are built into the conversation flow</li>
                        </ul>
                        
                        <div class="interactive-demo">
                            <div class="interactive-demo-header">Try a Conversation</div>
                            <div class="bg-gray-50 p-3 rounded mb-3 min-h-[100px]" id="chatHistory">
                                <div class="mb-2">
                                    <div class="bg-gray-200 rounded-lg px-3 py-2 inline-block max-w-[80%]">
                                        <p class="text-sm">Hello! How can I assist you today?</p>
                                    </div>
                                </div>
                            </div>
                            <div class="flex">
                                <input type="text" id="chatInput" class="flex-1 p-2 border border-gray-300 rounded-l" placeholder="Type a message...">
                                <button id="sendMessage" class="px-4 py-2 bg-black text-white rounded-r hover:bg-gray-800">Send</button>
                            </div>
                        </div>
                        
                        <h2>Performance Metrics</h2>
                        <p>The assistant has been trained and evaluated on multiple benchmarks:</p>
                        <ul>
                            <li>Intent recognition accuracy: 94.3%</li>
                            <li>Entity extraction F1 score: 0.89</li>
                            <li>Response relevance (human evaluated): 4.6/5</li>
                            <li>Average response time: 230ms</li>
                        </ul>
                        
                        <h2>Future Roadmap</h2>
                        <p>Planned enhancements include:</p>
                        <ul>
                            <li>Voice interface for hands-free interaction</li>
                            <li>Improved contextual understanding for more complex conversations</li>
                            <li>Expanded domain-specific knowledge for targeted applications</li>
                            <li>Multi-modal capabilities (image recognition, document processing)</li>
                        </ul>
                    </div>
                `
            }
        },
        blog: {
            blog1: {
                title: "GPT-4 and the Future of AI",
                date: "April 12, 2023",
                author: "Alex Mitchell",
                readTime: "7 min read",
                content: `
                    <div class="content-body">
                        <p>With the release of GPT-4, we've reached another milestone in artificial intelligence. This exploration looks at GPT-4's capabilities and what they mean for the future of AI and its applications across various industries.</p>
                        
                        <h2>The Evolution of Language Models</h2>
                        <p>GPT-4 represents a significant leap forward in language model capabilities. While its architecture remains similar to its predecessor, GPT-3, the improvements in its performance, reliability, and versatility are substantial. GPT-4 demonstrates enhanced understanding of nuance, context, and intent, enabling it to produce more coherent, creative, and accurate outputs.</p>
                        
                        <p>The evolution of these models follows an interesting trajectory:</p>
                        <ul>
                            <li><strong>GPT-1 (2018):</strong> Introduced the basic transformer architecture for language modeling</li>
                            <li><strong>GPT-2 (2019):</strong> Scaled up the model by 10x, showing surprising emergent abilities</li>
                            <li><strong>GPT-3 (2020):</strong> Scaled up by another 100x, demonstrating impressive few-shot learning abilities</li>
                            <li><strong>GPT-4 (2023):</strong> Multi-modal capabilities and significant improvements in reasoning</li>
                        </ul>
                        
                        <h2>Key Capabilities of GPT-4</h2>
                        <p>GPT-4 introduces several noteworthy capabilities that set it apart from previous models:</p>
                        
                        <h3>Multimodal Understanding</h3>
                        <p>Unlike its predecessors, GPT-4 can process both text and images as inputs. This allows it to interpret visual information and provide responses based on the combined context, opening up many new application possibilities in fields like accessibility, education, and visual reasoning.</p>
                        
                        <h3>Enhanced Reasoning</h3>
                        <p>Perhaps the most significant improvement in GPT-4 is its reasoning capabilities. The model demonstrates a better understanding of logic, causality, and complex instructions. It can break down multi-step problems, maintain logical consistency across longer outputs, and handle nuanced scenarios more effectively.</p>
                        
                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <p class="font-medium mb-2">Example: Solving a physics problem</p>
                            <p><strong>Prompt:</strong> A ball is thrown upward with an initial velocity of 15 m/s from a height of 2 meters. Assuming gravity is 9.8 m/s², when will the ball hit the ground?</p>
                            <p><strong>GPT-4 Response:</strong> To solve this problem, I'll use the kinematic equation:</p>
                            <p>y = y₀ + v₀t - (1/2)gt²</p>
                            <p>Where:</p>
                            <ul>
                                <li>y is the final position (0 when the ball hits the ground)</li>
                                <li>y₀ is the initial height (2 meters)</li>
                                <li>v₀ is the initial velocity (15 m/s upward)</li>
                                <li>g is the acceleration due to gravity (9.8 m/s²)</li>
                                <li>t is the time it takes for the ball to hit the ground</li>
                            </ul>
                            <p>Substituting our values:</p>
                            <p>0 = 2 + 15t - (1/2)(9.8)t²</p>
                            <p>0 = 2 + 15t - 4.9t²</p>
                            <p>Rearranging to standard form:</p>
                            <p>4.9t² - 15t - 2 = 0</p>
                            <p>Using the quadratic formula: t = [-b ± √(b² - 4ac)]/2a</p>
                            <p>t = [15 ± √(15² - 4(4.9)(-2))]/2(4.9)</p>
                            <p>t = [15 ± √(225 + 39.2)]/9.8</p>
                            <p>t = [15 ± √264.2]/9.8</p>
                            <p>t = [15 ± 16.25]/9.8</p>
                            <p>t = 3.19 seconds or t = -0.13 seconds</p>
                            <p>Since we can't have negative time in this physical scenario, the ball hits the ground after approximately 3.19 seconds.</p>
                        </div>
                        
                        <h3>Improved Factuality</h3>
                        <p>GPT-4 shows reduced hallucinations and improved factual accuracy compared to GPT-3.5. This is crucial for applications requiring reliable information, though it's important to note that the model can still make errors and is not designed to be a replacement for expert knowledge or fact-checking.</p>
                        
                        <h2>Implications for Various Industries</h2>
                        
                        <h3>Healthcare</h3>
                        <p>GPT-4's enhanced capabilities could transform healthcare through improved medical documentation, patient triage, and information retrieval. Its ability to process both text and images opens up possibilities for analyzing medical images alongside clinical notes, providing more comprehensive assistance to healthcare professionals.</p>
                        
                        <p>However, significant challenges remain, particularly around safety, regulatory compliance, and the risk of misinformation. GPT-4 is not a medical device and should not be used to diagnose or treat patients without proper medical oversight.</p>
                        
                        <h3>Education</h3>
                        <p>In education, GPT-4 can provide personalized learning experiences, generate educational content, and assist teachers with administrative tasks. Its ability to understand complex concepts and explain them at different levels of complexity makes it a powerful tool for personalized education.</p>
                        
                        <p>The model can adapt to different learning styles and paces, providing explanations tailored to individual students' needs. It can also generate practice problems, offer immediate feedback, and help students explore concepts from multiple angles.</p>
                        
                        <h3>Legal</h3>
                        <p>The legal industry stands to benefit from GPT-4's enhanced reasoning and document processing capabilities. The model can assist with contract review, legal research, and document drafting, potentially reducing the time lawyers spend on routine tasks and allowing them to focus on more complex aspects of their work.</p>
                        
                        <h2>Ethical Considerations and Limitations</h2>
                        <p>While GPT-4 represents significant progress, it's essential to acknowledge its limitations and the ethical considerations surrounding its use:</p>
                        
                        <h3>Bias and Fairness</h3>
                        <p>Like all AI systems trained on human-generated data, GPT-4 can reflect and potentially amplify biases present in its training data. Efforts have been made to reduce harmful biases, but vigilance is required when deploying these models, especially in sensitive contexts.</p>
                        
                        <h3>Transparency and Explainability</h3>
                        <p>The complexity of large language models like GPT-4 makes them difficult to fully understand and audit. This lack of transparency can be problematic, especially in high-stakes decisions where explainability is crucial.</p>
                        
                        <h3>Environmental Impact</h3>
                        <p>The computational resources required to train models like GPT-4 have significant environmental implications. The AI community needs to prioritize more energy-efficient approaches to model development and deployment.</p>
                        
                        <h2>Looking Ahead: The Future of AI</h2>
                        <p>GPT-4 offers a glimpse into the future direction of AI development. Several trends are likely to shape the field in the coming years:</p>
                        
                        <h3>Integration with Other Systems</h3>
                        <p>Future AI systems will likely combine language models like GPT-4 with other specialized components, creating more comprehensive and capable systems that can interact with the world in more meaningful ways.</p>
                        
                        <h3>Specialized Domain Models</h3>
                        <p>While general-purpose models like GPT-4 are impressive, there's growing interest in developing specialized models fine-tuned for specific domains such as medicine, law, or scientific research, where domain-specific knowledge and constraints are crucial.</p>
                        
                        <h3>Human-AI Collaboration</h3>
                        <p>The most promising path forward involves humans and AI systems working together, leveraging the strengths of both. This collaborative approach can mitigate many of the risks associated with AI while maximizing its benefits.</p>
                        
                        <h2>Conclusion</h2>
                        <p>GPT-4 represents an important step forward in artificial intelligence, demonstrating improved capabilities across numerous dimensions. While it's not without limitations and risks, it opens up new possibilities for how we can use AI to enhance human creativity, productivity, and knowledge.</p>
                        
                        <p>As we continue to develop and deploy these powerful technologies, a thoughtful and responsible approach is essential. By considering the ethical implications, addressing limitations, and focusing on human-AI collaboration, we can work toward a future where AI systems like GPT-4 serve as tools that augment our capabilities and help us address important challenges.</p>
                    </div>
                `
            },
            blog2: {
                title: "Building a Python NLP Pipeline",
                date: "March 8, 2023",
                author: "Maya Johnson",
                readTime: "10 min read",
                content: `
                    <div class="content-body">
                        <p>Natural Language Processing (NLP) is one of the most exciting fields in artificial intelligence, enabling computers to understand, interpret, and generate human language. This tutorial walks through creating an efficient NLP pipeline in Python, from text preprocessing to implementing advanced models.</p>
                        
                        <h2>Understanding NLP Pipelines</h2>
                        <p>An NLP pipeline is a series of data processing steps that transform raw text into structured information that can be analyzed or used to train machine learning models. A well-designed pipeline creates a foundation for accurate and efficient text processing.</p>
                        
                        <p>A typical NLP pipeline includes these stages:</p>
                        <ol>
                            <li><strong>Text acquisition</strong> - collecting and loading text data</li>
                            <li><strong>Preprocessing</strong> - cleaning and normalizing text</li>
                            <li><strong>Feature extraction</strong> - converting text into numerical representations</li>
                            <li><strong>Modeling</strong> - applying algorithms to extract insights or make predictions</li>
                            <li><strong>Evaluation</strong> - assessing model performance</li>
                            <li><strong>Deployment</strong> - integrating the pipeline into applications</li>
                        </ol>

                        <h2>Setting Up Your Environment</h2>
                        <p>Before we begin building our pipeline, let's set up a Python environment with the necessary libraries:</p>

                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <pre><code class="language-bash"># Create a virtual environment
python -m venv nlp_env

# Activate the environment
# On Windows:
nlp_env\\Scripts\\activate
# On macOS/Linux:
source nlp_env/bin/activate

# Install required libraries
pip install numpy pandas scikit-learn nltk spacy transformers</code></pre>
                        </div>

                        <h2>Stage 1: Text Preprocessing</h2>
                        <p>Text preprocessing is crucial for removing noise and standardizing the input data. Let's implement a comprehensive preprocessing function using NLTK:</p>

                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <pre><code class="language-python">import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocesses text by performing the following steps:
    1. Convert to lowercase
    2. Remove punctuation and special characters
    3. Remove numbers
    4. Remove extra whitespaces
    5. Tokenize
    6. Remove stopwords
    7. Lemmatize
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

# Example usage
sample_text = "This is a sample text with numbers like 123 and punctuation marks! How will it be preprocessed?"
processed_tokens = preprocess_text(sample_text)
print(processed_tokens)</code></pre>
                        </div>

                        <p>This preprocessing function handles common tasks like lowercasing, removing punctuation, tokenizing, and lemmatizing. Depending on your specific requirements, you might want to customize this function to include or exclude certain steps.</p>

                        <h2>Stage 2: Feature Extraction</h2>
                        <p>Once our text is preprocessed, we need to convert it into numerical features that machine learning algorithms can understand. Let's implement three common approaches:</p>

                        <h3>Bag of Words (BoW)</h3>
                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <pre><code class="language-python">from sklearn.feature_extraction.text import CountVectorizer

def bow_features(texts, max_features=None):
    """
    Convert a collection of text documents to a matrix of token counts
    """
    # Join tokens back into strings for CountVectorizer
    processed_texts = [' '.join(preprocess_text(text)) for text in texts]
    
    # Create a CountVectorizer
    vectorizer = CountVectorizer(max_features=max_features)
    
    # Fit and transform the texts
    bow_matrix = vectorizer.fit_transform(processed_texts)
    
    return bow_matrix, vectorizer

# Example usage
texts = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

bow_matrix, bow_vectorizer = bow_features(texts)
print(bow_matrix.toarray())
print(bow_vectorizer.get_feature_names_out())</code></pre>
                        </div>

                        <h3>TF-IDF (Term Frequency-Inverse Document Frequency)</h3>
                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <pre><code class="language-python">from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(texts, max_features=None):
    """
    Convert a collection of text documents to a matrix of TF-IDF features
    """
    processed_texts = [' '.join(preprocess_text(text)) for text in texts]
    
    # Create a TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    return tfidf_matrix, vectorizer

# Example usage
tfidf_matrix, tfidf_vectorizer = tfidf_features(texts)
print(tfidf_matrix.toarray())
print(tfidf_vectorizer.get_feature_names_out())</code></pre>
                        </div>

                        <h3>Word Embeddings with spaCy</h3>
                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <pre><code class="language-python">import numpy as np
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_md')

def word_embeddings(texts):
    """
    Convert texts to word embeddings using spaCy
    Returns the average embedding for each document
    """
    embeddings = []
    
    for text in texts:
        doc = nlp(text)
        # Calculate the mean of word vectors for the document
        if len(doc) > 0:
            doc_vector = np.mean([token.vector for token in doc if not token.is_stop and not token.is_punct], axis=0)
        else:
            doc_vector = np.zeros(nlp.meta['vectors']['width'])
        embeddings.append(doc_vector)
    
    return np.array(embeddings)

# Example usage
embeddings = word_embeddings(texts)
print(embeddings.shape)
print(embeddings[0][:10])  # Print first 10 dimensions of the first document embedding</code></pre>
                        </div>

                        <h2>Stage 3: Building the Complete Pipeline</h2>
                        <p>Now let's assemble these components into a complete, flexible NLP pipeline using scikit-learn's Pipeline API:</p>

                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <pre><code class="language-python">from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC

# Custom transformer that applies the preprocessing function
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [' '.join(preprocess_text(text)) for text in X]

# Build a complete pipeline for text classification
def build_nlp_pipeline(feature_type='tfidf', classifier=None):
    """
    Build a complete NLP pipeline with preprocessing, feature extraction, and classification
    
    Parameters:
    -----------
    feature_type : str, {'bow', 'tfidf', 'embedding'}
        Type of feature extraction to use
    classifier : sklearn classifier, optional
        Classifier to use (default is SVC)
    
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        Complete NLP pipeline
    """
    if classifier is None:
        classifier = SVC(kernel='linear')
    
    preprocessor = TextPreprocessor()
    
    if feature_type == 'bow':
        vectorizer = CountVectorizer()
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    elif feature_type == 'embedding':
        vectorizer = FunctionTransformer(lambda x: word_embeddings(x))
    else:
        raise ValueError("feature_type must be one of 'bow', 'tfidf', or 'embedding'")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    return pipeline

# Example usage
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load a small subset of the 20 Newsgroups dataset
categories = ['alt.atheism', 'sci.space']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    twenty_train.data, twenty_train.target, test_size=0.25, random_state=42
)

# Build and train the pipeline
pipeline = build_nlp_pipeline(feature_type='tfidf')
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=twenty_train.target_names))</code></pre>
                        </div>

                        <h2>Stage 4: Advanced NLP with Transformer Models</h2>
                        <p>For many modern NLP tasks, transformer-based models like BERT provide state-of-the-art performance. Let's integrate these into our pipeline:</p>

                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <pre><code class="language-python">from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

def get_bert_embeddings(texts, max_length=512):
    """
    Get embeddings from a pre-trained BERT model
    """
    # Tokenize the texts
    encoded_inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        # Use the [CLS] token embeddings as the document embeddings
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embeddings

# Create a custom transformer using these embeddings
class BertTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_length=512):
        self.max_length = max_length
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return get_bert_embeddings(X, self.max_length)

# Update our pipeline builder to include BERT
def build_nlp_pipeline_with_bert(classifier=None):
    """
    Build an NLP pipeline using BERT embeddings
    """
    if classifier is None:
        classifier = SVC(kernel='linear')
    
    pipeline = Pipeline([
        ('bert_embeddings', BertTransformer()),
        ('classifier', classifier)
    ])
    
    return pipeline

# Example usage (with a smaller dataset due to computational requirements)
sample_size = 100  # Use a small sample for demonstration
X_train_sample = X_train[:sample_size]
y_train_sample = y_train[:sample_size]
X_test_sample = X_test[:sample_size]
y_test_sample = y_test[:sample_size]

# Build and train the BERT pipeline
bert_pipeline = build_nlp_pipeline_with_bert()
bert_pipeline.fit(X_train_sample, y_train_sample)

# Evaluate
y_pred_sample = bert_pipeline.predict(X_test_sample)
print(classification_report(y_test_sample, y_pred_sample, target_names=twenty_train.target_names))</code></pre>
                        </div>

                        <h2>Stage 5: Putting It All Together</h2>
                        <p>Finally, let's create a complete application that demonstrates our NLP pipeline in action, with a text classifier that can be easily adapted to different tasks:</p>

                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <pre><code class="language-python">import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

class TextClassifier:
    """
    A complete text classification system with flexible feature extraction and model selection
    """
    def __init__(self, feature_type='tfidf', use_bert=False, model_type='svm'):
        self.feature_type = feature_type
        self.use_bert = use_bert
        self.model_type = model_type
        self.pipeline = None
    
    def build_pipeline(self):
        """
        Build the appropriate pipeline based on configuration
        """
        if self.model_type == 'svm':
            classifier = SVC(probability=True)
        elif self.model_type == 'rf':
            classifier = RandomForestClassifier()
        else:
            raise ValueError("model_type must be one of 'svm' or 'rf'")
        
        if self.use_bert:
            self.pipeline = build_nlp_pipeline_with_bert(classifier)
        else:
            self.pipeline = build_nlp_pipeline(self.feature_type, classifier)
        
        return self.pipeline
    
    def train(self, X_train, y_train, param_grid=None):
        """
        Train the model, optionally using GridSearchCV for hyperparameter tuning
        """
        if self.pipeline is None:
            self.build_pipeline()
        
        if param_grid is not None:
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(self.pipeline, param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            # Train with default parameters
            self.pipeline.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_test, y_test, target_names=None):
        """
        Evaluate the model on test data
        """
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(report)
        return report

# Example usage
classifier = TextClassifier(feature_type='tfidf', model_type='svm')

# Define a parameter grid for tuning
param_grid = {
    'vectorizer__max_features': [1000, 2000, 3000],
    'classifier__C': [0.1, 1, 10],
}

# Train with parameter tuning
classifier.train(X_train, y_train, param_grid)

# Evaluate
classifier.evaluate(X_test, y_test, target_names=twenty_train.target_names)

# Make predictions on new data
new_texts = [
    "Space exploration has advanced significantly in recent decades.",
    "The debate between atheism and religion continues to be controversial."
]
predictions = classifier.predict(new_texts)
probabilities = classifier.predict_proba(new_texts)

# Display results
for text, pred, prob in zip(new_texts, predictions, probabilities):
    predicted_category = twenty_train.target_names[pred]
    confidence = prob[pred] * 100
    print(f"Text: {text}")
    print(f"Predicted category: {predicted_category}")
    print(f"Confidence: {confidence:.2f}%")
    print()</code></pre>
                        </div>

                        <h2>Conclusion</h2>
                        <p>In this tutorial, we've built a comprehensive NLP pipeline in Python, covering text preprocessing, feature extraction with traditional methods and transformer models, and classification. This flexible architecture can be adapted to various NLP tasks beyond classification, such as sentiment analysis, named entity recognition, or question answering.</p>
                        
                        <p>Key points to remember:</p>
                        <ul>
                            <li>Careful text preprocessing is essential for most NLP tasks and impacts model performance</li>
                            <li>Different feature extraction methods are suitable for different tasks and datasets</li>
                            <li>Modern transformer models offer state-of-the-art performance but come with higher computational costs</li>
                            <li>Using pipeline architecture improves code organization and prevents data leakage</li>
                            <li>Hyperparameter tuning is often necessary to achieve optimal performance</li>
                        </ul>
                        
                        <p>As you build your own NLP applications, remember to consider the specific requirements of your task and dataset. While transformer models like BERT often achieve the best performance, simpler approaches like TF-IDF can be more appropriate when computational resources are limited or when you need faster inference times.</p>

                        <div class="interactive-demo">
                            <div class="interactive-demo-header">Try Text Classification</div>
                            <div class="mb-3">
                                <textarea id="classifyText" class="w-full p-2 border border-gray-300 rounded" rows="3" placeholder="Enter text to classify...">Space exploration has advanced significantly in recent decades.</textarea>
                            </div>
                            <button id="runClassification" class="px-4 py-2 bg-black text-white rounded hover:bg-gray-800">Classify Text</button>
                            <div id="classificationResult" class="mt-3 p-3 bg-gray-50 rounded hidden"></div>
                        </div>
                    </div>
                `
            },
            blog3: {
                title: "The Ethics of AI Development",
                date: "February 22, 2023",
                author: "Jamal Rahman",
                readTime: "8 min read",
                content: `
                    <div class="content-body">
                        <p>As artificial intelligence becomes increasingly integrated into our daily lives, the ethical considerations surrounding its development have never been more important. This article explores the key ethical issues that AI developers should keep in mind when building these powerful systems.</p>
                        
                        <h2>The Growing Importance of AI Ethics</h2>
                        <p>Artificial intelligence is no longer confined to research labs or science fiction. It now powers recommendation systems, content moderation tools, hiring algorithms, medical diagnosis systems, and countless other applications that impact millions of people daily. With this widespread deployment comes a responsibility to consider the ethical implications of these systems.</p>
                        
                        <p>The field of AI ethics examines questions like:</p>
                        <ul>
                            <li>How can we ensure AI systems don't perpetuate or amplify social biases?</li>
                            <li>Who is responsible when an AI system makes a harmful decision?</li>
                            <li>How much transparency should AI systems provide about their decision-making?</li>
                            <li>What privacy safeguards should be in place for data used to train AI?</li>
                            <li>How can we ensure equitable access to AI benefits across society?</li>
                        </ul>
                        
                        <h2>Key Ethical Considerations for AI Developers</h2>
                        
                        <h3>1. Fairness and Bias</h3>
                        <p>AI systems learn from data, and if that data contains historical biases, the resulting systems can perpetuate or even amplify those biases. For example, hiring algorithms trained on historical hiring decisions might discriminate against women or minorities if those groups were underrepresented in past hiring.</p>
                        
                        <p><strong>Developer Responsibility:</strong> Carefully examine training data for potential biases, use diverse and representative datasets, implement fairness metrics, and regularly audit systems for biased outcomes across different demographic groups.</p>
                        
                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <h4 class="font-bold mb-2">Case Study: Amazon's Experimental Hiring Algorithm</h4>
                            <p>In 2018, Amazon scrapped an AI hiring tool that showed bias against women. The system was trained on resumes submitted to Amazon over a 10-year period, most of which came from men, reflecting the male dominance in the tech industry. The algorithm learned to penalize resumes that included the word "women's" (as in "women's chess club captain") and downgraded graduates from women's colleges.</p>
                            <p>This case illustrates how AI can amplify existing biases in training data and the importance of testing for fairness before deployment.</p>
                        </div>
                        
                        <h3>2. Transparency and Explainability</h3>
                        <p>Many advanced AI systems, particularly deep learning models, function as "black boxes" where it's difficult to understand how they reach specific conclusions. This opacity becomes problematic when these systems make significant decisions affecting people's lives.</p>
                        
                        <p><strong>Developer Responsibility:</strong> Prioritize explainable AI techniques when possible, provide meaningful explanations of how AI systems work to end users, and be transparent about limitations and potential errors.</p>
                        
                        <p>The right level of transparency depends on the application context:</p>
                        <ul>
                            <li>In healthcare or criminal justice, high explainability is essential</li>
                            <li>For recommendation systems, simpler explanations might suffice</li>
                            <li>For some applications, the ability to appeal or override decisions may be more important than detailed explanations</li>
                        </ul>
                        
                        <h3>3. Privacy and Data Governance</h3>
                        <p>AI development often requires large amounts of data, some of which may be sensitive or personal. Responsible AI development includes careful stewardship of this data.</p>
                        
                        <p><strong>Developer Responsibility:</strong> Obtain proper consent for data use, anonymize data when appropriate, secure data against breaches, minimize data collection to what's necessary, and comply with relevant regulations like GDPR or CCPA.</p>
                        
                        <p>Consider implementing techniques like:</p>
                        <ul>
                            <li>Differential privacy to provide mathematical guarantees against identifying individuals</li>
                            <li>Federated learning to train models without centralizing sensitive data</li>
                            <li>Data minimization strategies to reduce privacy risks</li>
                        </ul>
                        
                        <h3>4. Accountability and Oversight</h3>
                        <p>When AI systems make mistakes or cause harm, clear accountability structures should determine responsibility and provide recourse.</p>
                        
                        <p><strong>Developer Responsibility:</strong> Establish clear lines of accountability, implement robust testing and monitoring, provide mechanisms for users to appeal decisions, and consider third-party auditing for high-stakes applications.</p>
                        
                        <h3>5. Environmental and Social Impact</h3>
                        <p>AI development can have broader societal and environmental impacts that extend beyond the immediate application.</p>
                        
                        <p><strong>Developer Responsibility:</strong> Consider the energy consumption and carbon footprint of training and running AI models, assess potential job displacement effects, and evaluate how the technology might affect different communities.</p>
                        
                        <div class="bg-gray-100 p-4 rounded-lg mb-6">
                            <h4 class="font-bold mb-2">The Environmental Cost of AI</h4>
                            <p>Training a single large language model can generate as much carbon dioxide as five cars emit during their entire lifetimes. As models grow larger, their environmental impact increases dramatically.</p>
                            <p>Researchers are working on more efficient training methods, but developers should consider whether the benefits of increasingly large models justify their environmental costs.</p>
                        </div>
                        
                        <h2>Practical Frameworks for Ethical AI Development</h2>
                        <p>Moving from abstract principles to practical implementation requires structured approaches. Here are several frameworks that developers can adopt:</p>
                        
                        <h3>Ethics by Design</h3>
                        <p>Similar to "security by design" or "privacy by design," ethics by design integrates ethical considerations throughout the development lifecycle rather than treating them as an afterthought.</p>
                        
                        <p>Key practices include:</p>
                        <ul>
                            <li>Conducting ethical risk assessments at project inception</li>
                            <li>Incorporating diverse perspectives in design processes</li>
                            <li>Establishing ethical metrics alongside performance metrics</li>
                            <li>Using checklists and review processes at key development milestones</li>
                        </ul>
                        
                        <h3>The OECD AI Principles</h3>
                        <p>The Organisation for Economic Co-operation and Development (OECD) has established principles for responsible AI that have been adopted by over 40 countries. These principles emphasize:</p>
                        <ul>
                            <li>Inclusive growth, sustainable development and well-being</li>
                            <li>Human-centered values and fairness</li>
                            <li>Transparency and explainability</li>
                            <li>Robustness, security and safety</li>
                            <li>Accountability</li>
                        </ul>
                        
                        <h3>Impact Assessments</h3>
                        <p>Algorithmic Impact Assessments (AIAs) help organizations evaluate potential social, ethical, and legal impacts of AI systems before deployment. Similar to environmental impact assessments, AIAs involve:</p>
                        <ul>
                            <li>Identifying stakeholders who might be affected by the system</li>
                            <li>Assessing potential positive and negative impacts</li>
                            <li>Implementing mitigation strategies for identified risks</li>
                            <li>Establishing monitoring mechanisms for ongoing assessment</li>
                        </ul>
                        
                        <h2>Implementing Ethical AI in Practice</h2>
                        <p>Translating ethical principles into daily development practices requires concrete steps:</p>
                        
                        <h3>1. Diverse Teams</h3>
                        <p>Teams with diverse backgrounds and perspectives are better equipped to identify potential ethical issues. This diversity should include not just technical expertise but also backgrounds in social sciences, law, ethics, and other relevant fields.</p>
                        
                        <h3>2. Ethics Training</h3>
                        <p>AI developers should receive training in identifying and addressing ethical issues specific to AI. This includes understanding concepts like algorithmic bias, explainability, and privacy implications.</p>
                        
                        <h3>3. Documentation</h3>
                        <p>Thoroughly document decisions made during development, including:</p>
                        <ul>
                            <li>Data sources and collection methods</li>
                            <li>Preprocessing steps</li>
                            <li>Model selection rationale</li>
                            <li>Performance across different demographic groups</li>
                            <li>Limitations and potential risks</li>
                        </ul>
                        
                        <p>Tools like Model Cards and Datasheets for Datasets provide standardized formats for this documentation.</p>
                        
                        <h3>4. Ongoing Monitoring</h3>
                        <p>Ethical considerations don't end at deployment. Implement systems to:</p>
                        <ul>
                            <li>Monitor performance across different user groups</li>
                            <li>Track user feedback and complaints</li>
                            <li>Detect and address "concept drift" where model performance degrades over time</li>
                            <li>Regularly audit for bias or other ethical concerns</li>
                        </ul>
                        
                        <h3>5. User Feedback Mechanisms</h3>
                        <p>Create channels for users to report issues, appeal decisions, or provide feedback about AI systems. This feedback loop is essential for identifying unforeseen problems.</p>
                        
                        <div class="interactive-demo">
                            <div class="interactive-demo-header">AI Ethics Checklist</div>
                            <p class="text-sm mb-2">Use this interactive checklist to assess your AI project:</p>
                            <div class="mb-3">
                                <div class="flex items-center mb-2">
                                    <input type="checkbox" id="ethics1" class="mr-2">
                                    <label for="ethics1" class="text-sm">We've examined our training data for potential biases</label>
                                </div>
                                <div class="flex items-center mb-2">
                                    <input type="checkbox" id="ethics2" class="mr-2">
                                    <label for="ethics2" class="text-sm">We've tested model performance across diverse demographic groups</label>
                                </div>
                                <div class="flex items-center mb-2">
                                    <input type="checkbox" id="ethics3" class="mr-2">
                                    <label for="ethics3" class="text-sm">Our system provides appropriate explanations for its decisions</label>
                                </div>
                                <div class="flex items-center mb-2">
                                    <input type="checkbox" id="ethics4" class="mr-2">
                                    <label for="ethics4" class="text-sm">We've implemented proper consent and privacy protections</label>
                                </div>
                                <div class="flex items-center mb-2">
                                    <input type="checkbox" id="ethics5" class="mr-2">
                                    <label for="ethics5" class="text-sm">We've documented limitations and potential risks of our system</label>
                                </div>
                                <div class="flex items-center mb-2">
                                    <input type="checkbox" id="ethics6" class="mr-2">
                                    <label for="ethics6" class="text-sm">We have mechanisms for users to appeal decisions</label>
                                </div>
                                <div class="flex items-center mb-2">
                                    <input type="checkbox" id="ethics7" class="mr-2">
                                    <label for="ethics7" class="text-sm">We've considered environmental impacts of model training and deployment</label>
                                </div>
                                <div class="flex items-center mb-2">
                                    <input type="checkbox" id="ethics8" class="mr-2">
                                    <label for="ethics8" class="text-sm">We've consulted with diverse stakeholders about potential impacts</label>
                                </div>
                            </div>
                            <button id="checkEthics" class="px-4 py-2 bg-black text-white rounded hover:bg-gray-800">Evaluate Project</button>
                            <div id="ethicsResult" class="mt-3 p-3 bg-gray-50 rounded hidden"></div>
                        </div>
                        
                        <h2>Balancing Innovation and Ethics</h2>
                        <p>Some developers worry that ethical considerations might slow innovation or impose burdensome constraints. However, building ethics into AI development is increasingly seen as essential rather than optional:</p>
                        
                        <ul>
                            <li><strong>Legal requirements:</strong> Regulations like the EU's proposed AI Act are creating legal obligations for ethical AI</li>
                            <li><strong>Business necessity:</strong> Ethical failures can lead to reputational damage and lost trust</li>
                            <li><strong>Sustainability:</strong> Addressing ethical issues early prevents costly fixes or rewrites later</li>
                            <li><strong>Market advantage:</strong> Responsible AI can be a competitive differentiator</li>
                        </ul>
                        
                        <p>Rather than viewing ethics as a constraint, developers can see it as a design requirement that leads to more robust, trustworthy systems that better serve human needs.</p>
                        
                        <h2>Conclusion</h2>
                        <p>As AI systems become more powerful and more deeply integrated into society, the ethical dimensions of their development grow increasingly important. By adopting structured ethical frameworks, implementing practical steps throughout the development lifecycle, and fostering a culture that values responsible innovation, AI developers can create systems that not only perform well technically but also align with human values and societal needs.</p>
                        
                        <p>The most successful AI of the future will be not just technologically sophisticated but also ethically sound. By embracing this broader view of excellence in AI development, today's developers can help ensure that these powerful technologies fulfill their potential to benefit humanity.</p>
                    </div>
                `
            }
        }
    };
    
    // Content Manager to handle showing and hiding content
    window.contentManager = {
        scrollPosition: 0,
        
        showContent: function(type, id) {
            this.scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
            
            // Get the content to show
            const content = contentDatabase[type][id];
            if (!content) {
                console.error(`Content not found: ${type}/${id}`);
                return;
            }
            
            // Hide main content and show content view with fade effect
            mainContent.style.opacity = '0';
            mainContent.style.transition = 'opacity 0.3s ease';
            setTimeout(() => {
                mainContent.classList.add('hidden');
                contentView.classList.remove('hidden');
                contentView.style.opacity = '0';
                contentView.style.transition = 'opacity 0.3s ease';
                setTimeout(() => {
                    contentView.style.opacity = '1';
                }, 50);
            }, 300);
            
            // Scroll to top smoothly
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
            
            // Create content HTML
            let contentHTML = `
                <div class="content-header">
                    <button onclick="window.contentManager.hideContent()" class="mb-4 flex items-center text-gray-600 hover:text-gray-900">
                        <svg class="w-5 h-5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clip-rule="evenodd" />
                        </svg>
                        Back to Home
                    </button>
                    <h1 class="content-title">${content.title}</h1>
                    <div class="content-meta flex items-center text-gray-500 mt-4">
                        ${type === 'blog' && content.author ? `
                            <div class="content-meta-item mr-6">
                                <svg class="h-4 w-4 mr-1 inline" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd" />
                                </svg>
                                ${content.author}
                            </div>
                        ` : ''}
                        <div class="content-meta-item mr-6">
                            <svg class="h-4 w-4 mr-1 inline" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clip-rule="evenodd" />
                            </svg>
                            ${content.date}
                        </div>
                        ${type === 'blog' && content.readTime ? `
                            <div class="content-meta-item">
                                <svg class="h-4 w-4 mr-1 inline" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd" />
                                </svg>
                                ${content.readTime}
                            </div>
                        ` : ''}
                    </div>
                </div>
                
                ${content.content}
                
                ${content.tags ? `
                    <div class="content-tags mt-8">
                        ${content.tags.map(tag => `<span class="content-tag">${tag}</span>`).join('')}
                    </div>
                ` : ''}
            `;
            
            // Set content
            contentContainer.innerHTML = contentHTML;
            
            // Add event listeners for interactive elements
            this.setupInteractiveElements();

            // Add to browser history
            const state = { type, id };
            const url = `#${type}/${id}`;
            history.pushState(state, content.title, url);

            // Add popstate event listener if not already added
            if (!this.popStateListenerAdded) {
                window.addEventListener('popstate', this.handlePopState.bind(this));
                this.popStateListenerAdded = true;
            }
        },
        
        handlePopState: function(event) {
            if (event.state) {
                // 保存当前滚动位置
                this.scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
                // 显示内容
                this.showContent(event.state.type, event.state.id);
            } else {
                // 隐藏内容并恢复滚动位置
                contentContainer.innerHTML = '';
                contentView.classList.add('hidden');
                mainContent.classList.remove('hidden');
                mainContent.style.opacity = '1';
                
                // Restore scroll position smoothly
                window.scrollTo({
                    top: this.scrollPosition,
                    behavior: 'smooth'
                });
                
                // Update URL to remove hash if present
                if (window.location.hash) {
                    history.pushState(null, document.title, window.location.pathname + window.location.search);
                }
            }
        },
        
        hideContent: function() {
            // Fade out content view
            contentView.style.opacity = '0';
            contentView.style.transition = 'opacity 0.3s ease';
            
            setTimeout(() => {
                contentContainer.innerHTML = '';
                contentView.classList.add('hidden');
                mainContent.classList.remove('hidden');
                mainContent.style.opacity = '1';
                
                // Restore scroll position smoothly
                window.scrollTo({
                    top: this.scrollPosition,
                    behavior: 'smooth'
                });
                
                // Update URL to remove hash if present
                if (window.location.hash) {
                    history.pushState(null, document.title, window.location.pathname + window.location.search);
                }
            }, 300);
        },
        
        setupInteractiveElements: function() {
            // Example: Sentiment analyzer for NLP project
            const analyzeSentiment = document.getElementById('analyzeSentiment');
            if (analyzeSentiment) {
                analyzeSentiment.addEventListener('click', () => {
                    const text = document.getElementById('sentimentText').value;
                    const resultDiv = document.getElementById('sentimentResult');
                    
                    // Simple sentiment analysis demo (in a real app, this would call an API)
                    let sentiment = 'neutral';
                    let score = 0;
                    
                    const positiveWords = ['love', 'great', 'excellent', 'good', 'wonderful', 'amazing', 'happy', 'best'];
                    const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'worst', 'sad', 'disappointing', 'poor'];
                    
                    const lowerText = text.toLowerCase();
                    positiveWords.forEach(word => {
                        if (lowerText.includes(word)) score += 1;
                    });
                    negativeWords.forEach(word => {
                        if (lowerText.includes(word)) score -= 1;
                    });
                    
                    if (score > 0) sentiment = 'positive';
                    if (score < 0) sentiment = 'negative';
                    
                    // Display result
                    resultDiv.innerHTML = `
                        <div class="text-center">
                            <div class="text-lg font-medium mb-2">Sentiment: 
                                <span class="${sentiment === 'positive' ? 'text-green-600' : sentiment === 'negative' ? 'text-red-600' : 'text-gray-600'}">
                                    ${sentiment.toUpperCase()}
                                </span>
                            </div>
                            <div class="text-sm text-gray-500">
                                This is a simplified demo. A real sentiment analyzer would use more sophisticated algorithms.
                            </div>
                        </div>
                    `;
                    resultDiv.classList.remove('hidden');
                });
            }
            
            // Example: Image generator for Image Generation project
            const generateImage = document.getElementById('generateImage');
            if (generateImage) {
                generateImage.addEventListener('click', () => {
                    const prompt = document.getElementById('imagePrompt').value;
                    const resultDiv = document.getElementById('imageResult');
                    
                    // Show loading animation
                    resultDiv.classList.remove('hidden');
                    
                    // Simulate API call delay
                    setTimeout(() => {
                        resultDiv.innerHTML = `
                            <div class="text-center">
                                <p class="text-gray-700">Generated image based on: "${prompt}"</p>
                                <div class="mt-3">
                                    <img src="https://images.unsplash.com/photo-1614728894747-a83421e2b9c9?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80" alt="Generated image" class="rounded-lg mx-auto">
                                </div>
                                <p class="text-sm text-gray-500 mt-2">
                                    Note: This is a placeholder image. In a real application, this would be generated by an AI model.
                                </p>
                            </div>
                        `;
                    }, 1500);
                });
            }
            
            // Example: Chatbot for AI Assistant project
            const sendMessage = document.getElementById('sendMessage');
            if (sendMessage) {
                sendMessage.addEventListener('click', () => {
                    const chatInput = document.getElementById('chatInput');
                    const chatHistory = document.getElementById('chatHistory');
                    const userMessage = chatInput.value.trim();
                    
                    if (userMessage) {
                        // Add user message
                        chatHistory.innerHTML += `
                            <div class="mb-2 text-right">
                                <div class="bg-accent text-white rounded-lg px-3 py-2 inline-block max-w-[80%]">
                                    <p class="text-sm">${userMessage}</p>
                                </div>
                            </div>
                        `;
                        
                        // Clear input
                        chatInput.value = '';
                        
                        // Show typing indicator
                        chatHistory.innerHTML += `
                            <div class="mb-2 typing-indicator">
                                <div class="bg-gray-200 rounded-lg px-3 py-2 inline-block max-w-[80%]">
                                    <div class="flex space-x-1">
                                        <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                        <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                                        <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                                    </div>
                                </div>
                            </div>
                        `;
                        
                        // Scroll to bottom
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                        
                        // Simulate response after delay
                        setTimeout(() => {
                            // Remove typing indicator
                            const typingIndicator = chatHistory.querySelector('.typing-indicator');
                            if (typingIndicator) typingIndicator.remove();
                            
                            // Add bot response (in a real app, this would call an API)
                            let botResponse = "I'm sorry, I don't understand that question.";
                            
                            if (userMessage.toLowerCase().includes('hello') || userMessage.toLowerCase().includes('hi')) {
                                botResponse = "Hello! How can I help you today?";
                            } else if (userMessage.toLowerCase().includes('weather')) {
                                botResponse = "I don't have real-time weather data, but I'd be happy to help with other questions!";
                            } else if (userMessage.toLowerCase().includes('help')) {
                                botResponse = "I can answer questions about AI, assist with information, or just chat. What would you like to know?";
                            } else if (userMessage.toLowerCase().includes('ai') || userMessage.toLowerCase().includes('artificial intelligence')) {
                                botResponse = "Artificial Intelligence refers to systems that can perform tasks that typically require human intelligence. Is there something specific about AI you'd like to learn?";
                            }
                            
                            chatHistory.innerHTML += `
                                <div class="mb-2">
                                    <div class="bg-gray-200 rounded-lg px-3 py-2 inline-block max-w-[80%]">
                                        <p class="text-sm">${botResponse}</p>
                                    </div>
                                </div>
                            `;
                            
                            // Scroll to bottom
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }, 1000);
                    }
                });
                
                // Allow sending message with Enter key
                const chatInput = document.getElementById('chatInput');
                if (chatInput) {
                    chatInput.addEventListener('keypress', (e) => {
                        if (e.key === 'Enter') {
                            e.preventDefault();
                            sendMessage.click();
                        }
                    });
                }
            }
            
            // Example: Ethics checklist for the Ethics blog
            const checkEthics = document.getElementById('checkEthics');
            if (checkEthics) {
                checkEthics.addEventListener('click', () => {
                    const checkboxes = document.querySelectorAll('#checkEthics').length;
                    const checked = document.querySelectorAll('#checkEthics:checked').length;
                    const resultDiv = document.getElementById('ethicsResult');
                    
                    // Count checked items
                    let checkedCount = 0;
                    for (let i = 1; i <= 8; i++) {
                        if (document.getElementById(`ethics${i}`).checked) {
                            checkedCount++;
                        }
                    }
                    
                    // Calculate score and feedback
                    const percentage = Math.round((checkedCount / 8) * 100);
                    let feedback = '';
                    let color = '';
                    
                    if (percentage >= 88) {
                        feedback = "Excellent! Your project demonstrates strong ethical considerations.";
                        color = "text-green-600";
                    } else if (percentage >= 63) {
                        feedback = "Good progress, but there are still some ethical areas to address.";
                        color = "text-yellow-600";
                    } else {
                        feedback = "Your project needs significant work to address ethical concerns.";
                        color = "text-red-600";
                    }
                    
                    resultDiv.innerHTML = `
                        <div class="text-center">
                            <div class="text-2xl font-bold ${color}">${percentage}%</div>
                            <p class="mt-2">${feedback}</p>
                            <p class="text-sm text-gray-500 mt-4">This is a simplified assessment. A comprehensive ethics review would be more detailed.</p>
                        </div>
                    `;
                    resultDiv.classList.remove('hidden');
                });
            }
        }
    };
    
    // Add event listener to back button
    if (backButton) {
        backButton.addEventListener('click', (e) => {
            e.preventDefault();
            window.contentManager.hideContent();
        });
    }
});
