# Tests for render_scenes module
#
import json
from unittest.mock import MagicMock, patch

import hydra

from recipes.cec2.data_preparation.render_scenes import run


@patch("recipes.cec2.data_preparation.render_scenes.SceneRenderer")
def test_render_scenes(mock_sr, tmp_path):
    """Test render_scenes function."""

    # Mock out the SceneRenderer as we don't want to actually render
    # any scenes - it's very slow and tested elsewhere in the unit tests
    scene_renderer_instance = MagicMock()
    mock_sr.return_value = scene_renderer_instance

    # Using the config file from the actual recipe, but overriding
    # just the paths to the scene data and the output directory
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../recipes/cec2/data_preparation/",
        job_name="test_cec2",
    )
    hydra_cfg = hydra.compose(
        config_name="config",
        overrides=[
            "path.root=tests/test_data",
            "path.metadata_dir=tests/test_data/metadata",
            (
                "scene_renderer.train.metadata.scene_definitions="
                "tests/test_data/metadata/scenes.test.json"
            ),
            (
                "scene_renderer.dev.metadata.scene_definitions="
                "tests/test_data/metadata/scenes.test.json"
            ),
            (
                "scene_renderer.demo.metadata.scene_definitions="
                "tests/test_data/metadata/scenes.test.json"
            ),
            f"scene_renderer.train.paths.scenes={tmp_path}",
        ],
    )

    run(hydra_cfg)

    # Check the the scene renderer was instantiated
    assert mock_sr.call_count == 3
    assert scene_renderer_instance.render_scenes.call_count == 3

    # Read the scene data directly
    with open("tests/test_data/metadata/scenes.test.json", encoding="utf-8") as fp:
        expected_scene_data = json.load(fp)

    # Check that the scene data was delivered to the SceneRenderer
    assert (
        scene_renderer_instance.render_scenes.call_args.args[0] == expected_scene_data
    )
