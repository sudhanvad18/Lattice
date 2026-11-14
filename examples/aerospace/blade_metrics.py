"""Illustrative erosion trend calculations, not maintenance acceptance limits."""

from dataclasses import dataclass


@dataclass
class BladeMeasurement:
    """One measurement from a repeatable radial station."""

    serial: str
    cycles: int
    loss_mm: float

    def normalized_loss(self, reference_mm: float) -> float:
        if reference_mm <= 0:
            raise ValueError("Reference thickness must be positive")
        return self.loss_mm / reference_mm


def erosion_rate(before: BladeMeasurement, after: BladeMeasurement) -> float:
    """Calculate observed loss change per cycle for the same blade."""
    if before.serial != after.serial:
        raise ValueError("Measurements must refer to the same blade")
    elapsed = after.cycles - before.cycles
    if elapsed <= 0:
        raise ValueError("Cycle counter must increase")
    return (after.loss_mm - before.loss_mm) / elapsed


def summarize(before: BladeMeasurement, after: BladeMeasurement) -> dict:
    """Return trend metrics without assigning a serviceability decision."""
    return {
        "serial": after.serial,
        "loss_fraction": after.normalized_loss(4.0),
        "mm_per_cycle": erosion_rate(before, after),
    }


if __name__ == "__main__":
    baseline = BladeMeasurement("DEMO-001", 100, 0.02)
    repeat = BladeMeasurement("DEMO-001", 600, 0.08)
    print(summarize(baseline, repeat))
