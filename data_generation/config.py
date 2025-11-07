"""
Configuration for the data generation pipeline.
"""

# Model configuration (with fallback support)
# OpenRouter model IDs
OPENROUTER_GENERATION_MODEL = "nousresearch/hermes-4-405b"
OPENROUTER_VALIDATION_MODEL = "anthropic/claude-sonnet-4.5"

# Cerebras direct API model names (fallback)
CEREBRAS_GENERATION_MODEL = "qwen-3-235b-a22b-thinking-2507"
CEREBRAS_VALIDATION_MODEL = "llama-4-maverick-17b-128e-instruct"

# Models are used directly by component imports

# Pipeline settings - clean Python configuration
import os

# Core pipeline configuration
DEFAULT_BATCH_SIZE = 100
TEMPERATURE = 0.88  # Higher for creativity in generation
VALIDATION_TEMPERATURE = 0.3  # Lower for consistent evaluation
SIMILARITY_THRESHOLD = 0.60

# Verbalized Sampling settings - for focused single-principle generation
VS_TAIL_PROBABILITY = 0.10  # Keep only responses with probability < 0.10 for maximum diversity
VS_BATCH_MULTIPLIER = 2  # Generate 2x batch size to account for tail probability filtering

# OpenAI deduplication settings - stricter deduplication for focused mode
OPENAI_DEDUP_THRESHOLD = 0.50  # More strict than default 0.60 for sentence-transformer
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"  # Superior to sentence-transformer models

# Token limits - balanced for quality vs speed
GENERATION_MAX_TOKENS = 8000
VALIDATION_MAX_TOKENS = 2000

# Automation settings - set TARGET_ROWS via environment for automated runs
TARGET_ROWS = int(os.getenv("TARGET_ROWS", 0)) if os.getenv("TARGET_ROWS") else None

# Validation sampling settings - optimized for efficiency with quality
VALIDATION_SAMPLE_PERCENTAGE = 5  # Sample n% for validation
VALIDATION_FAILURE_THRESHOLD = 50  # Reject batch at 50% failures
VALIDATION_ESCALATION_THRESHOLD = 30  # Increase sampling at 30% failures
SKIP_VALIDATION = True  # Set to True to bypass validation entirely (accepts all generated scenarios)

# Context-aware generation settings - enabled by default for better uniqueness
ENABLE_DATASET_CONTEXT = True
ENABLE_DEDUPLICATION_FEEDBACK = True
CONTEXT_ANALYSIS_FREQUENCY = 1  # Analyze context every batch

# Few-shot learning configuration - use human-generated examples to guide generation
FEW_SHOT_ENABLED = True
FEW_SHOT_EXAMPLE_COUNT = 39  # Number of first N rows (human-generated) to use as examples

# Humane Technology Principles - 8 fixed principles in kebab-case
HUMANE_PRINCIPLES = [
    "respect-user-attention",
    "enable-meaningful-choices",
    "enhance-human-capabilities",
    "protect-dignity-and-safety",
    "foster-healthy-relationships",
    "prioritize-long-term-wellbeing",
    "be-transparent-and-honest",
    "design-for-equity-and-inclusion"
]

# Topic domains - 12 fixed domains for scenario categorization
TOPIC_DOMAINS = [
    "relationships",
    "mental-health",
    "financial-decisions",
    "career-guidance",
    "parenting",
    "health-wellness",
    "technology-use",
    "politics-society",
    "education",
    "moral-ambiguity",
    "crisis-situations",
    "everyday-decisions"
]

# Vulnerable population types to consider
VULNERABLE_POPULATIONS = [
    "children",
    "teenagers",
    "elderly",
    "people-with-disabilities",
    "non-native-speakers",
    "low-tech-literacy"
]

# Dataset file paths (JSONL)
DATASET_PATH = "../data/humane_bench.jsonl"
BACKUP_PATH = "../data/dataset_backup.jsonl"