"""Type definitions and helpers for inference predictions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TagPredictionResult:
    """Structured result produced by tag prediction inference."""

    caption: str
    taglist: str
    rating_labels: dict[str, float]
    character_labels: dict[str, float]
    general_labels: dict[str, float]

    def format_summary(self, show_probabilities: bool = True) -> str:
        """Format prediction results as a human-readable multi-line string."""
        lines = [
            "=" * 60,
            f"Caption: {self.caption}",
            "=" * 60,
            f"Tags: {self.taglist}",
            "",
            "Ratings:",
        ]

        for tag, probability in self.rating_labels.items():
            if show_probabilities:
                lines.append(f"  {tag}: {probability:.3f}")
            else:
                lines.append(f"  {tag}")

        lines.extend(
            [
                "",
                f"Character tags ({len(self.character_labels)}):",
            ],
        )
        for tag, probability in self.character_labels.items():
            if show_probabilities:
                lines.append(f"  {tag}: {probability:.3f}")
            else:
                lines.append(f"  {tag}")

        lines.extend(
            [
                "",
                f"General tags ({len(self.general_labels)}):",
            ],
        )
        for tag, probability in self.general_labels.items():
            if show_probabilities:
                lines.append(f"  {tag}: {probability:.3f}")
            else:
                lines.append(f"  {tag}")

        return "\n".join(lines)
