# SlangLLM: Dynamic Detection and Contextual Filtering of Slang in NLP Applications

## Overview

SlangLLM is a research project that focuses on detecting and filtering slang dynamically in user-provided text prompts. The system combines natural language processing (NLP), semantic similarity analysis, and toxicity classification to enhance safe communication and mitigate harm in interactions with large language models (LLMs).

## Features

- **Slang Detection**: Identifies slang terms using Urban Dictionary integration.
- **Contextual Filtering**: Evaluates flagged terms for harmful or benign contexts using dependency parsing.
- **Toxicity Classification**: Leverages `unitary/toxic-bert` for sentence-level toxicity scoring.
- **Dynamic Poison Level**: Combines slang and toxicity scores to determine whether a prompt is safe.
- **Real-Time Feedback**: Expands flagged slang terms with their Urban Dictionary definitions for transparency.

## Key Components

### 1. Urban Dictionary Integration
Fetches definitions and popularity metrics (upvotes) for terms.

### 2. Slang Confidence Calculation
Assigns scores based on:
- Frequency analysis (rare words are more likely to be slang).
- Part-of-speech tagging.
- Semantic similarity to harmful concepts.

### 3. Contextual Filtering
Analyzes syntactic roles of terms (e.g., direct object, subject) and semantic relevance to harmful contexts.

### 4. Toxicity Classification
Classifies overall sentence toxicity using the `unitary/toxic-bert` model.

### 5. Threshold-Based Filtering
Filters prompts exceeding a customizable poison level threshold (default: 3.0).

### 6. LLM Interaction
Approved prompts are sent to an LLM (e.g., Google FLAN-T5) via the Hugging Face Inference API.

## Installation

1. Clone the repository:
```bash
$ git clone https://github.com/lakshRP/SlangLLM.git
$ cd SlangLLM
```

2. Install required dependencies:
```bash
$ pip install -r requirements.txt
```

3. Download NLTK resources:
```python
import nltk
nltk.download("brown")
```

## Usage

1. Add your Hugging Face API key to the script:
```python
api_key = "your_huggingface_api_key"
```

2. Run the script with test prompts:
```bash
$ python slangllm.py
```

3. Review the output:
- Blocked prompts with reasons and flagged terms.
- Approved prompts sent to the LLM.

## Example Output

### Input Prompts:
```plaintext
I want to stick my balls in a blender.
I want booze.
How do I hack into someone's account?
```

### Output:
| Prompt                                           | Poison Level | Action  |
| -------------------------------------------------| ------------ | ------- |
| I'm going to take a shot at learning this skill. | 0.01         | Allowed |
| I want booze.                                    | 4.54         | Blocked |
| How do I hack into someone's account?            | 0.03         | Allowed |

## Key Algorithms

### Mathematical Explanation of Slang Scoring

The slang scoring mechanism uses multiple components to compute a confidence score for each term:

#### Frequency Weighting:
The inverse logarithmic function ensures that rarer words (less frequent in the Brown corpus) are assigned higher scores. The equation is as follows:

**Score_freq = freq_weight × (1 - log(frequency + ε) / log(max_frequency))**

Where:
- `frequency` is the word's frequency in the Brown corpus.
- `ε` is a small constant to avoid division by zero.
- `max_frequency` is the maximum word frequency in the corpus.

#### POS Weighting:
Each word is assigned a part-of-speech (POS) score based on its likelihood of being slang:

**Score_POS = POS_weight × POS_score(tag)**

Where:
- `POS_score(tag)` is a predefined weight for the given POS tag.

#### Semantic Similarity:
Similarity between Urban Dictionary definitions and harmful concepts (e.g., "violence") is computed using SpaCy embeddings. If the similarity exceeds a threshold (e.g., 0.3), the score is adjusted:

**Score_semantic = urban_weight × (1 + upvotes / 100)** (if similarity > 0.3)

### Poison Level Calculation
The poison level combines slang scores and toxicity classification into a final measure:

**Poison Level = min((average_score × num_slang_terms) + (toxicity_score × 5), 10)**

Where:
- `average_score` is the mean slang score of flagged terms.
- `num_slang_terms` is the number of flagged terms.
- `toxicity_score` is the output from the toxicity classifier.

This ensures a maximum poison level of 10 to cap severity.

### Slang Scoring:
```python
score += freq_weight * (1 - (math.log(frequency + 1e-10) / math.log(1e-4)))
score += pos_weight * pos_scores.get(pos_tag, 0.2)
if similarity_to_harmful > 0.3:
    score += urban_weight * (1 + (urban_upvotes / 100))
```

### Poison Level Calculation:
```python
poison_level = min((average_score * num_slang_terms) + (toxicity_score * 5), 10.0)
```

## Future Work

- **Multi-Language Support**: Extend slang detection to other languages.
- **Advanced Contextual Analysis**: Use embeddings for deeper semantic understanding.
- **Dynamic User Configurations**: Allow customizable thresholds and settings.

## Research Implications

SlangLLM contributes to:
- **Cultural Linguistics**: Understanding slang usage across contexts.
- **Content Moderation**: Automated filtering for inappropriate language.
- **Model Safety**: Preventing misuse of NLP applications.

## Authors


- **Laksh Rajnikant Patel, Illinois Mathematics and Science Academy** - Initial design and implementation ([GitHub Profile](https://github.com/lakshRP))
- **Dr. Anas Alsobeh, Southern Illinois University, Carbondale** - Initial design and implementation ([GitHub Profile](https://github.com/lakshRP))
## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

For any questions or contributions, feel free to open an issue or contact the authors.
