"""
Configuration for the data generation pipeline.
"""

# Model configuration (with fallback support)
# OpenRouter model IDs
OPENROUTER_GENERATION_MODEL = "anthropic/claude-sonnet-4"
OPENROUTER_VALIDATION_MODEL = "anthropic/claude-sonnet-4"

# Cerebras direct API model names (fallback)
CEREBRAS_GENERATION_MODEL = "qwen-3-235b-a22b-thinking-2507"
CEREBRAS_VALIDATION_MODEL = "llama-4-maverick-17b-128e-instruct"

# Models are used directly by component imports

# Pipeline settings - clean Python configuration
import os

# Core pipeline configuration
DEFAULT_BATCH_SIZE = 75
TEMPERATURE = 0.88  # Higher for creativity in generation
VALIDATION_TEMPERATURE = 0.3  # Lower for consistent evaluation
SIMILARITY_THRESHOLD = 0.60

# Token limits - balanced for quality vs speed
GENERATION_MAX_TOKENS = 8000
VALIDATION_MAX_TOKENS = 2000

# Automation settings - set TARGET_ROWS via environment for automated runs
TARGET_ROWS = int(os.getenv("TARGET_ROWS", 0)) if os.getenv("TARGET_ROWS") else None

# Validation sampling settings - optimized for efficiency with quality
VALIDATION_SAMPLE_PERCENTAGE = 20  # Sample 20% for validation
VALIDATION_FAILURE_THRESHOLD = 50  # Reject batch at 50% failures
VALIDATION_ESCALATION_THRESHOLD = 30  # Increase sampling at 30% failures

# Context-aware generation settings - enabled by default for better uniqueness
ENABLE_DATASET_CONTEXT = True
ENABLE_DEDUPLICATION_FEEDBACK = True
CONTEXT_ANALYSIS_FREQUENCY = 1  # Analyze context every batch

# Humane Tech Principles mapping
HUMANE_PRINCIPLES = {
    "protect_attention": "Protect Attention and Intention",
    "support_wellbeing": "Support Well-being",
    "enhance_agency": "Enhance Agency",
    "foster_connection": "Foster Connection and Empathy",
    "promote_understanding": "Promote Sense-making and Understanding",
    "respect_diversity": "Respect Diversity and Inclusion"
}

# Primary evaluation categories used for scenario generation and validation
PRIMARY_EVALUATION_CATEGORIES = [
    "Fostering Meaningful Connection",
    "Psychological Safety and Emotional Well-Being",
    "User Autonomy and Freedom from Manipulation",
    "Transparency and Explainability",
    "Trustworthiness and Reliability",
    "Fairness and Inclusivity",
    "Privacy and Data Rights",
    "Accountability and Human Oversight",
    "Ethical Alignment and Moral Integrity",
    "Human Well-Being and Flourishing",
    "Crisis Response and Emergency Situations"
]

# Map evaluation categories to core humane principles
EVALUATION_TO_PRINCIPLE_MAP = {
    "Fostering Meaningful Connection": "foster_connection",
    "Psychological Safety and Emotional Well-Being": "support_wellbeing",
    "User Autonomy and Freedom from Manipulation": "enhance_agency",
    "Transparency and Explainability": "promote_understanding",
    "Trustworthiness and Reliability": "promote_understanding",
    "Fairness and Inclusivity": "respect_diversity",
    "Privacy and Data Rights": "enhance_agency",
    "Accountability and Human Oversight": "promote_understanding",
    "Ethical Alignment and Moral Integrity": "support_wellbeing",
    "Human Well-Being and Flourishing": "support_wellbeing",
    "Crisis Response and Emergency Situations": "protect_attention"
}

# Scenario categories for balanced generation
SCENARIO_CATEGORIES = [
    "individual_advice",
    "societal_questions",
    "vulnerable_populations",
    "moral_ambiguity",
    "crisis_situations",
    "everyday_decisions"
]

# Vulnerable population types to include
VULNERABLE_POPULATIONS = [
    "teenagers",
    "elderly",
    "people_in_crisis",
    "people_with_disabilities",
    "non_native_speakers",
    "low_tech_literacy"
]

# Topic domains to cover
TOPIC_DOMAINS = [
    "relationships",
    "mental_health",
    "financial_decisions",
    "career_guidance",
    "parenting",
    "health_wellness",
    "technology_use",
    "social_media",
    "politics_society",
    "education",
    "consumer_decisions"
]

# Fixed category system for scenario generation
# Based on the 8 core humane technology principles
FIXED_CATEGORIES = [
    "Respect User Attention",
    "Enable Meaningful Choices",
    "Enhance Human Capabilities",
    "Protect Dignity and Safety",
    "Foster Healthy Relationships",
    "Prioritize Long-term Wellbeing",
    "Be Transparent and Honest",
    "Design for Equity and Inclusion"
]

# Category normalization mapping
# Maps the 4 key aspects of each principle (from principles.md) to the 8 fixed categories
# This allows LLMs to use descriptive subcategory names that get normalized to principles
CATEGORY_NORMALIZATION_MAP = {
    # 1. Respect User Attention
    "minimize unnecessary interruptions and notifications": "Respect User Attention",
    "avoid exploiting psychological vulnerabilities to capture attention": "Respect User Attention",
    "design interfaces that help users focus on what matters to them": "Respect User Attention",
    "provide clear signals when seeking user attention": "Respect User Attention",

    # 2. Enable Meaningful Choices
    "present genuine choices, not manipulative illusions of choice": "Enable Meaningful Choices",
    "provide transparent information about consequences of choices": "Enable Meaningful Choices",
    "make defaults ethical and aligned with user wellbeing": "Enable Meaningful Choices",
    "allow users to understand and modify how systems make decisions about them": "Enable Meaningful Choices",

    # 3. Enhance Human Capabilities
    "support human autonomy and decision-making": "Enhance Human Capabilities",
    "design for collaboration between human intelligence and machine capabilities": "Enhance Human Capabilities",
    "avoid creating dependencies that diminish human skills": "Enhance Human Capabilities",
    "foster learning and growth through appropriate challenges": "Enhance Human Capabilities",

    # 4. Protect Dignity and Safety
    "respect user privacy and secure personal data": "Protect Dignity and Safety",
    "design systems that protect vulnerable users": "Protect Dignity and Safety",
    "prevent harassment and abuse": "Protect Dignity and Safety",
    "avoid manipulative dark patterns that undermine dignity": "Protect Dignity and Safety",

    # 5. Foster Healthy Relationships
    "support authentic human connection": "Foster Healthy Relationships",
    "design interactions that respect appropriate boundaries": "Foster Healthy Relationships",
    "create systems that encourage empathy and understanding": "Foster Healthy Relationships",
    "avoid features that exploit social comparison or encourage antisocial behavior": "Foster Healthy Relationships",

    # 6. Prioritize Long-term Wellbeing
    "consider psychological impacts of design choices": "Prioritize Long-term Wellbeing",
    "resist optimizing solely for engagement or time spent": "Prioritize Long-term Wellbeing",
    "design for sustainable use that enhances quality of life": "Prioritize Long-term Wellbeing",
    "build awareness of how technology affects wellbeing": "Prioritize Long-term Wellbeing",

    # 7. Be Transparent and Honest
    "disclose how systems work and make decisions": "Be Transparent and Honest",
    "avoid deceptive interfaces or interactions": "Be Transparent and Honest",
    "clearly communicate system limitations and capabilities": "Be Transparent and Honest",
    "make algorithms and data use understandable to users": "Be Transparent and Honest",

    # 8. Design for Equity and Inclusion
    "design for accessibility across abilities": "Design for Equity and Inclusion",
    "test with diverse users to uncover unintended consequences": "Design for Equity and Inclusion",
    "address bias in data, algorithms, and design": "Design for Equity and Inclusion",
    "consider impacts across different communities and contexts": "Design for Equity and Inclusion"
}

# Dataset file paths (JSONL)
DATASET_PATH = "../data/simple_human_friendliness_dataset.jsonl"
BACKUP_PATH = "../data/dataset_backup.jsonl"