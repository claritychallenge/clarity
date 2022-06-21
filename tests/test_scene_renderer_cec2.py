import logging

import clarity.data.scene_renderer_cec2 as sr

logger = logging.getLogger(__name__)


def test_rotation():
    """Test head rotation code."""
    rotation = [
        {"sample": 100, "angle": -1.50},
        {"sample": 200, "angle": 1.50},
    ]
    origin = [1.0, 0.0, 0.0]
    duration = 300

    angles = sr.two_point_rotation(rotation, origin, duration)
    logger.info(f"{angles[0]}, {angles[299]}")
    logger.info(angles)
