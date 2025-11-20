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
                date: "April 2025",
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
            }
        },
        blog: {
            blog1: {
                title: "Use Agent AI and Cursor to build a personal website",
                date: "April 12, 2025",
                author: "TTZ",
                readTime: "3 min read",
                content: `
                    <div class="content-body">
                        <p><em>The source code of this personal website can be found on <a href="https://github.com/TianzeTang0504/personalweb" target="_blank">GitHub</a>.</em></p>
                        <p>Although large language models (such as GPT) have shown strong capabilities in code and engineering, they still encounter some difficulties for complex multi-step tasks and work that requires external tools. Compared with humans, traditional large language models are more like a fully automatic encyclopedia. Many people have noticed this problem and hope that AI can truly solve problems in human ways of thinking.</p>
                        
                        <p>Some time ago, I noticed an AI called <strong><a href="https://flowith.io/blank" target="_blank">Flowith</a></strong>. The team claims that Flowith can conduct complete, engineering-style multi-step thinking for the questions you ask, and can complete a complete project from scratch. If it is really that easy to use, combined with Cursor, I should be able to build a personal website very quickly (It turned out to be true. Without knowing any front-end or back-end languages, it took me less than three hours to build and launch this website from scratch).</p>
                        
                        <h2>Use Flowith to get the whole workflow and a demo</h2>
                        <p>This part is what I really want to share from this project.
                        Before I use Flowith, I had the idea to build this website.
                        I also know that I can use GPT to help me.
                        But asking GPT to build a website is not a good idea because I need to know waht I should ask.
                        But I do not know website building at all. So I just give up.
                        </p>

                        <p>
                        So when I see Flowith which can complete a project from scratch, I thought it may work.
                        The actual experience is much better than I expected.
                        I only told Flowith "I want to build a personla website."
                        Then Flowith asked me 5 questions:
                        <ol>
                            <li>
                                <strong>What is the purpose of the website?</strong>
                                </li>
                            <li>
                                <strong>What is the design style of the website?</strong>
                                </li>
                            <li>
                                <strong>Do you need any specific functions?</strong>
                                </li>
                            <li>
                                <strong>Do you have any reference websites?</strong>
                                </li>
                            <li>
                                <strong>Do you have any experience with website building?</strong>
                                </li>
                                
                        </ol>
                        </p>
                        <p>
                        After answering these questions, Flowith started to work for like 15 mins.
                        It devided the project into 5 parts:
                        <ol>
                            <li>
                                <strong>How to generate the website?</strong>
                                </li>
                            <li>
                                <strong>How to design the website?</strong>
                                </li>
                            <li>
                                <strong>How to register the domain name?</strong>
                                </li>
                            <li>
                                <strong>Where to host the website?</strong>
                                </li>
                            <li>
                                <strong>Specific instructions for hosting the website.</strong>
                                </li>
                                
                        </ol>
                        </p>
                        <p>
                        For each part, Flowith will give me a detailed plan like comparison of domain providers, common mistakes when hosting a website, etc.
                        I believe that even though it is your first day on the Internet, you can at least know the basic process of building a website.
                        And the most important, <strong>you will know what questions to ask when you build a website.</strong>
                        </p>
                        <p>
                        At last, Flowith gave me a demo of the website which was already very good.
                        80% of the website you see now are the same as the demo.
                        </p>

                        
                        <h2>Use Cursor to complete the entire page</h2>
                        <p>The demo from Flowith is already very good.
                        I just need to use Cursor to adjust some details.
                        For example, I want to add a navigation bar to the website, I only need to tell Cursor to add a navigation bar on main page, and it will scan the whole project and complete the entire work.
                        It is so easy and convenient.
                        The only thing I need to do is to tell Cursor what I want to do.
                        However, here I have two suggestions for you:
                        <ol>
                            <li>
                                <strong>Make sure you have a good version control system like GitHub.</strong>
                                When you use Cursor to complete the entire page, it will generate a lot of changes.
                                Sometimes after many changes you find that the original code is better, but you can find it (I waste a lot of time on this).
                                If you don't have a good version control system, you may not be able to manage the changes and the code may be lost.
                                </li>
                            <li>
                                <strong>Make sure you at least understand the basic structure of the project and what is going on.</strong>
                                Even though I do not learn about js and css at all, I have basic knowledge about other languages.
                                So at least I know each file and parts' function.
                                When the project is complex, totally rely on Cursor may not be a good idea.
                                </li>
                                    
                        </ol>
                        </p>
                        
                        <h2>Host this website on Edgeone Pages from Tencent</h2>
                        <p>
                        When I was looking for a place to host this website, I found that <strong><a href="https://edgeone.ai/document/160427672992178176" target="_blank">Edgeone Pages</a></strong> from Tencent is a good choice.
                        It is a free service provided by Tencent, and it is very easy to use.
                        I have used Alibaba Cloud before, but it is paid and the interface is cumbersome to use, and it is very troublesome to access from outside of mainland China.
                        Edgeone Pages is very convenient, free, and accessible from anywhere.
                        The only thing I need to do is to upload the source code to GitHub and connect it to Edgeone Pages.
                        Then, I can access the website by visiting the URL provided by Edgeone Pages.
                        If you have your own domain name, you can also connect it to Edgeone Pages (I buy one on <strong>Namecheap</strong>, also very cheap).
                        Also, when you update your code on GitHub, Edgeone Pages will automatically deploy the latest version of the website.
                        </p>
                    </div>
                `
            },
            blog2: {
                title: "Deep Learning in Medical Imaging",
                date: "March 8, 2023",
                author: "Maya Johnson",
                readTime: "10 min read",
                content: `
                    <div class="content-body">
                        <p>Deep learning models, particularly Convolutional Neural Networks (CNNs), are revolutionizing the field of medical imaging. This article explores how these algorithms are assisting radiologists in diagnosing diseases earlier and more accurately.</p>
                        
                        <h2>The Rise of AI in Radiology</h2>
                        <p>Medical imaging data—X-rays, CT scans, MRIs—is growing exponentially. Radiologists are under immense pressure to interpret these images quickly and accurately. Artificial intelligence, specifically deep learning, offers a powerful solution to augment human capabilities.</p>
                        
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
            }
        }
    };
    
    // Content Manager to handle showing and hiding content
    window.contentManager = {
        scrollPosition: 0,
        
        showContent: function(type, id) {
            this.scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
            
            // No need to manually stop sidebar music anymore, SharedAudioManager handles state
            // But we do want to hide the sidebar player UI? The UI structure hides the mainContent which contains sidebar.
            // The floating player will sync with current audio state automatically.

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
                
                // Reset scroll position instantly before showing content
                window.scrollTo({
                    top: 0,
                    behavior: 'instant'
                });
                
                setTimeout(() => {
                    contentView.style.opacity = '1';
                }, 50);
            }, 300);
            
            // Create content HTML
            let contentHTML = `
                <div class="content-header">
                    <button onclick="window.contentManager.hideContent()" class="mb-4 flex items-center text-gray-400 hover:text-accent transition-colors">
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
            
            // 初始化代码高亮
            hljs.highlightAll();
            
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
            
            // Initialize Floating Music Player
            if (window.FloatingMusicPlayer) {
                // Check if already exists to avoid duplicates
                if (!this.floatingPlayer) {
                     this.floatingPlayer = new window.FloatingMusicPlayer(document.body);
                }
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
                
                // 隐藏回到顶部按钮
                const contentTopButton = document.getElementById('contentTopButton');
                if (contentTopButton) {
                    contentTopButton.classList.add('hidden');
                    contentTopButton.classList.remove('flex');
                }
                
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
            
            // Destroy floating player when leaving content view
            if (this.floatingPlayer) {
                this.floatingPlayer.destroy();
                this.floatingPlayer = null;
            }

            // Hide content top button immediately
            const contentTopButton = document.getElementById('contentTopButton');
            if (contentTopButton) {
                contentTopButton.classList.add('hidden');
                contentTopButton.classList.remove('flex');
            }

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
