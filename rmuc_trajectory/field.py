"""Field geometry and rule-manual canvas calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FIELD_WIDTH_M = 28.0
FIELD_HEIGHT_M = 15.0

# The source image is 1683 x 938 px.  Its strong outer border at roughly
# (73, 41)-(1602, 883) includes the half-metre-wide perimeter baffles.  The
# tracking coordinate system starts at the inner field/apron corners instead,
# represented by the rectangle below.  Keeping this calibration in source
# pixels makes the chosen physical boundary auditable and easy to re-measure.
SOURCE_IMAGE_SIZE = (1683, 938)
INNER_FIELD_RECT_PX = (100, 69, 1576, 856)

# The dataset records buildings at (0, 0). These approximate centers are
# calibrated from the rule-manual top-view canvas and are only for visualization.
OBJECTIVE_POSITIONS = {
    ("红", "基地"): (2.46, 7.44),
    ("红", "前哨站"): (10.87, 3.58),
    ("蓝", "前哨站"): (17.12, 11.32),
    ("蓝", "基地"): (25.73, 7.44),
}

# Approximate centers of the two fortress buff zones, calibrated from the same
# rule-manual top view as the static objectives. They support tactical
# geofencing only; the dataset has no native fortress-occupancy flag.
FORTRESS_POSITIONS = {
    "红": (6.22, 7.45),
    "蓝": (21.78, 7.45),
}
FORTRESS_RADIUS_M = 1.3


@dataclass(frozen=True)
class FieldCanvas:
    """Mapping between tracking coordinates and the top-view rule canvas.

    The rule manual states a 28 m by 15 m field. The extracted top-view render
    is illustrative rather than a survey drawing, so it is cropped to the inner
    fence and stretched to the official dimensions. This is appropriate for
    trajectory interpretation, not millimetre-level obstacle clearance.
    """

    image_path: Path
    width_m: float = FIELD_WIDTH_M
    height_m: float = FIELD_HEIGHT_M
    crop_left: float = INNER_FIELD_RECT_PX[0] / SOURCE_IMAGE_SIZE[0]
    crop_right: float = INNER_FIELD_RECT_PX[2] / SOURCE_IMAGE_SIZE[0]
    crop_top: float = INNER_FIELD_RECT_PX[1] / SOURCE_IMAGE_SIZE[1]
    crop_bottom: float = INNER_FIELD_RECT_PX[3] / SOURCE_IMAGE_SIZE[1]

    def crop_pixels(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        return (
            round(image_width * self.crop_left),
            round(image_height * self.crop_top),
            round(image_width * self.crop_right),
            round(image_height * self.crop_bottom),
        )


def default_canvas(project_root: Path) -> FieldCanvas:
    return FieldCanvas(project_root / "assets" / "rmuc_2026_field_top_view.jpeg")
