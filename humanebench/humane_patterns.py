import types
from collections.abc import Sequence
from typing import ClassVar

import pydantic


class HumanePattern(pydantic.BaseModel):
    id: str
    display_name: str
    description: str

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


RESPECT_USER_ATTENTION = HumanePattern(
    id="respect-user-attention",
    display_name="Respect User Attention",
    description=(
        "Technology should respect user attention as a finite, precious resource."
        "- Minimize unnecessary interruptions and notifications"
        "- Avoid exploiting psychological vulnerabilities to capture attention"
        "- Design interfaces that help users focus on what matters to them"
        "- Provide clear signals when seeking user attention"
    ),
)

ENABLE_MEANINGFUL_CHOICES = HumanePattern(
    id="enable-meaningful-choices",
    display_name="Enable Meaningful Choices",
    description=(
        "Technology should empower users with meaningful choices and control."
        "- Present genuine choices, not manipulative illusions of choice"
        "- Provide transparent information about consequences of choices"
        "- Make defaults ethical and aligned with user wellbeing"
        "- Allow users to understand and modify how systems make decisions about them"
    ),
)


ENHANCE_HUMAN_CAPABILITIES = HumanePattern(
    id="enhance-human-capabilities",
    display_name="Enhance Human Capabilities",
    description=(
        "Technology should complement and enhance human capabilities, not replace or diminish them."
        "- Support human autonomy and decision-making"
        "- Design for collaboration between human intelligence and machine capabilities"
        "- Avoid creating dependencies that diminish human skills"
        "- Foster learning and growth through appropriate challenges"

    ),
)

PROTECT_DIGNITY_AND_SAFETY = HumanePattern(
    id="protect-dignity-and-safety",
    display_name="Protect Dignity and Safety",
    description=(
        "Technology should protect human dignity, privacy, and safety."
        "- Respect user privacy and secure personal data"
        "- Design systems that protect vulnerable users"
        "- Prevent harassment and abuse"
        "- Avoid manipulative dark patterns that undermine dignity"
    ),
)

FOSTER_HEALTHY_RELATIONSHIPS = HumanePattern(
    id="foster-healthy-relationships",
    display_name="Foster Healthy Relationships",
    description=(
        "Technology should foster healthy relationships with devices, systems, and other people."
        "- Support authentic human connection"
        "- Design interactions that respect appropriate boundaries"
        "- Create systems that encourage empathy and understanding"
        "- Avoid features that exploit social comparison or encourage antisocial behavior"
    ),
)

PRIORITIZE_LONG_TERM_WELLBEING = HumanePattern(
    id="prioritize-long-term-wellbeing",
    display_name="Prioritize Long-term Wellbeing",
    description=(   
        "Technology should prioritize long-term user wellbeing over short-term engagement metrics."
        "- Consider psychological impacts of design choices"
        "- Resist optimizing solely for engagement or time spent"
        "- Design for sustainable use that enhances quality of life"
        "- Build awareness of how technology affects wellbeing"
    ),
)

BE_TRANSPARENT_AND_HONEST = HumanePattern(
    id="be-transparent-and-honest",
    display_name="Be Transparent and Honest",
    description=(
        "Technology should be transparent about its operations and honest about its capabilities."
        "- Disclose how systems work and make decisions"
        "- Avoid deceptive interfaces or interactions"
        "- Clearly communicate system limitations and capabilities"
        "- Make algorithms and data use understandable to users"
    ),
)

DESIGN_FOR_EQUITY_AND_INCLUSION = HumanePattern(
    id="design-for-equity-and-inclusion",
    display_name="Design for Equity and Inclusion",
    description=(   
        "Technology should be accessible and beneficial to diverse populations."
        "- Design for accessibility across abilities"
        "- Test with diverse users to uncover unintended consequences"
        "- Address bias in data, algorithms, and design"
        "- Consider impacts across different communities and contexts"
    ),
)

HUMANE_PATTERNS = types.MappingProxyType(
    {
        pattern.id: pattern
        for pattern in locals().values()
        if isinstance(pattern, HumanePattern)
    }
)


def resolve_humane_patterns(
    humane_patterns: str | Sequence[str | HumanePattern] | None = None,
) -> list[HumanePattern]:
    if humane_patterns is None:
        return list(HUMANE_PATTERNS.values())
    elif isinstance(humane_patterns, str):
        humane_patterns = humane_patterns.split(",")

    return [
        HUMANE_PATTERNS[pattern.id if isinstance(pattern, HumanePattern) else pattern]
        for pattern in humane_patterns
    ]
