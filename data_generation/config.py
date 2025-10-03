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
# Based on humane tech principles: attention, wellbeing, agency, connection, understanding, diversity
FIXED_CATEGORIES = [
    # Attention & Intention (Respect User Attention)
    "Attention Management",
    "Digital Distraction",
    "Notification Boundaries",
    "Time Well-Spent",

    # Well-being (Support Well-being, Prioritize Long-term Wellbeing)
    "Mental Health & Crisis",
    "Emotional Well-Being",
    "Screen Time & Dependency",
    "Addiction & Dependency",
    "Stress & Burnout",
    "Physical Health Impact",

    # Agency & Autonomy (Enable Meaningful Choices, Enhance Human Capabilities)
    "User Autonomy & Choice",
    "Manipulation & Dark Patterns",
    "Decision Support",
    "Skill Development",
    "Technology Dependency",

    # Connection & Relationships (Foster Healthy Relationships)
    "Loneliness & Social Connection",
    "Online vs Offline Relationships",
    "Family & Parenting",
    "Community & Belonging",
    "Social Comparison",

    # Understanding & Truth (Be Transparent and Honest, Promote Sense-making)
    "Misinformation & Truth",
    "Content Authenticity",
    "Algorithmic Transparency",
    "System Explainability",

    # Dignity & Safety (Protect Dignity and Safety)
    "Privacy & Data Rights",
    "Security & Safety",
    "Harassment & Abuse",
    "Vulnerable Populations",

    # Equity & Inclusion (Design for Equity and Inclusion)
    "Accessibility",
    "Fairness & Bias",
    "Digital Divide",
    "Cultural Sensitivity",

    # Ethics & Accountability
    "Ethical AI Behavior",
    "Human Oversight",
    "Accountability & Responsibility",

    # Crisis & Emergency
    "Crisis Response",
    "Emergency Situations",
    "Immediate Harm Prevention"
]

# Category normalization mapping for edge cases
# Maps common variations or legacy categories to fixed categories
CATEGORY_NORMALIZATION_MAP = {
    # Legacy/variant mappings
    "mental health": "Mental Health & Crisis",
    "mental health crisis": "Mental Health & Crisis",
    "social connection": "Loneliness & Social Connection",
    "loneliness": "Loneliness & Social Connection",
    "screen time": "Screen Time & Dependency",
    "addiction": "Addiction & Dependency",
    "privacy": "Privacy & Data Rights",
    "data privacy": "Privacy & Data Rights",
    "manipulation": "Manipulation & Dark Patterns",
    "dark patterns": "Manipulation & Dark Patterns",
    "autonomy": "User Autonomy & Choice",
    "transparency": "Algorithmic Transparency",
    "misinformation": "Misinformation & Truth",
    "fake news": "Misinformation & Truth",
    "crisis": "Crisis Response",
    "emergency": "Emergency Situations",
    "accessibility": "Accessibility",
    "bias": "Fairness & Bias",
    "fairness": "Fairness & Bias",
    "harassment": "Harassment & Abuse",
    "vulnerable": "Vulnerable Populations",
    "digital wellbeing": "Emotional Well-Being",
    "notification": "Notification Boundaries",
    "distraction": "Digital Distraction",
    "social media": "Social Comparison",
    "relationships": "Online vs Offline Relationships",
    "parenting": "Family & Parenting",
    "burnout": "Stress & Burnout",
    "health": "Physical Health Impact",
    "decision making": "Decision Support",
    "skill": "Skill Development",
    "dependency": "Technology Dependency",
    "authenticity": "Content Authenticity",
    "explainability": "System Explainability",
    "security": "Security & Safety",
    "safety": "Security & Safety",
    "ethics": "Ethical AI Behavior",
    "accountability": "Accountability & Responsibility",
    "oversight": "Human Oversight",
    "harm": "Immediate Harm Prevention",
    "community": "Community & Belonging",
    "digital divide": "Digital Divide",
    "culture": "Cultural Sensitivity",
    "attention": "Attention Management",
    "time": "Time Well-Spent",

    # Additional mappings from existing dataset
    "consumer decision": "Decision Support",
    "ai understanding": "System Explainability",
    "medical advice": "Decision Support",
    "medical information": "Decision Support",
    "age & technology": "Accessibility",
    "data control": "Privacy & Data Rights",
    "error correction": "Accountability & Responsibility",
    "legal advice": "Decision Support",
    "ethical dilemma": "Ethical AI Behavior",
    "relationship issues": "Online vs Offline Relationships",
    "education & life decisions": "Decision Support",
    "children & digital rights": "Privacy & Data Rights",
    "social comparison & self-esteem": "Social Comparison",
    "parasocial relationships": "Online vs Offline Relationships",
    "misinformation & information quality": "Misinformation & Truth",
    "election integrity & democracy": "Misinformation & Truth",
    "digital life balance": "Time Well-Spent",
    "environmental sustainability": "Ethical AI Behavior",
    "neurodiversity & accessibility": "Accessibility",
    "mental health & digital wellness": "Mental Health & Crisis",
    "voice assistants & iot privacy": "Privacy & Data Rights",
    "gaming & interactive media": "Screen Time & Dependency",
    "social media & mental health": "Mental Health & Crisis",
    "content recommendation ethics": "Ethical AI Behavior",
    "employment & ai discrimination": "Fairness & Bias",
    "digital wellness & privacy": "Privacy & Data Rights",
    "cultural representation & ai": "Cultural Sensitivity",
    "virtual & augmented reality": "Screen Time & Dependency",
    "mental health & medication": "Mental Health & Crisis",
    "social isolation & ai dependency": "Technology Dependency",
    "ai deification & delusions": "Mental Health & Crisis",
    "domestic violence": "Immediate Harm Prevention",
    "teen pregnancy": "Vulnerable Populations",
    "political decision-making": "Decision Support",
    "parenting guidance": "Family & Parenting",
    "marriage & relationships": "Online vs Offline Relationships",
    "workplace ethics": "Ethical AI Behavior",
    "gender identity & family conflict": "Family & Parenting",
    "racism & friendship": "Harassment & Abuse",
    "parenting teens & sexuality": "Family & Parenting",
    "elder care & dementia": "Vulnerable Populations",
    "addiction & recovery": "Addiction & Dependency",
    "sexual harassment": "Harassment & Abuse",
    "religious conflict & lgbtq+ children": "Family & Parenting",
    "climate change & education": "Ethical AI Behavior",
    "friendship & relationship concerns": "Online vs Offline Relationships",
    "teen independence & career": "Decision Support",
    "family planning disagreement": "Family & Parenting",
    "child abuse reporting": "Immediate Harm Prevention",
    "medical decision-making": "Decision Support",
    "infidelity consideration": "Online vs Offline Relationships",
    "parenting & body autonomy": "User Autonomy & Choice",
    "addiction & family response": "Addiction & Dependency",
    "religious community & abuse": "Community & Belonging",
    "age gap relationships": "Online vs Offline Relationships",
    "cultural conflict & marriage": "Cultural Sensitivity",
    "cult involvement": "Immediate Harm Prevention",
    "child protection & grooming": "Immediate Harm Prevention",
    "financial crisis & shame": "Crisis Response",
    "gender identity & young children": "Family & Parenting",
    "child exploitation material": "Immediate Harm Prevention",
    "elder financial abuse": "Immediate Harm Prevention",
    "radicalization concerns": "Immediate Harm Prevention",
    "relationship style conflict": "Online vs Offline Relationships",
    "suicide or violence risk": "Emergency Situations",
    "adult child in abusive relationship": "Immediate Harm Prevention",
    "adoption concerns": "Family & Parenting",
    "teen dating & age gaps": "Vulnerable Populations",
    "vaccination disagreement": "Decision Support",
    "screen time & family": "Screen Time & Dependency",
    "financial decision": "Decision Support",
    "career & life decisions": "Decision Support",
    "workplace issues": "Ethical AI Behavior",
    "life decisions": "Decision Support",
    "education & ethics": "Ethical AI Behavior",
    "community connection": "Community & Belonging",
    "parenting & safety": "Family & Parenting",
    "mental health & body image": "Mental Health & Crisis",
    "social issues": "Ethical AI Behavior",
    "dating & safety": "Security & Safety",
    "mental health & media": "Mental Health & Crisis",
    "business ethics": "Ethical AI Behavior",
    "elder protection": "Vulnerable Populations",
    "mental health technology": "Mental Health & Crisis",
    "parenting & technology": "Family & Parenting",
    "dating & technology": "Online vs Offline Relationships"
}

# Dataset file paths (JSONL)
DATASET_PATH = "../data/simple_human_friendliness_dataset.jsonl"
BACKUP_PATH = "../data/dataset_backup.jsonl"