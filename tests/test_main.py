import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from PIL import Image

from two_point_five_d_info.main import main, luminance, parse_layer


def make_image(path, color, width=4, height=4, alpha=255):
    img = Image.new("RGBA", (width, height), (*color, alpha))
    img.save(path)
    return path


class TestParseLayer:
    def test_path_only(self):
        result = parse_layer("image.png")
        assert result == ("image.png", None)

    def test_path_with_crop(self):
        result = parse_layer("image.png:10,20,30,40")
        assert result == ("image.png", (10, 20, 30, 40))

    def test_path_with_spaces_in_crop(self):
        result = parse_layer("image.png:10, 20, 30, 40")
        assert result == ("image.png", (10, 20, 30, 40))

    def test_invalid_crop_count(self):
        with pytest.raises(click.BadParameter):
            parse_layer("image.png:10,20,30")


class TestLuminance:
    def test_black(self):
        assert luminance("#000000") == 0.0

    def test_white(self):
        assert luminance("#ffffff") == 255.0

    def test_red(self):
        assert luminance("#ff0000") == 0.299 * 255

    def test_green(self):
        assert luminance("#00ff00") == 0.587 * 255

    def test_blue(self):
        assert luminance("#0000ff") == 0.114 * 255

    def test_ascending(self):
        assert luminance("#0000ff") < luminance("#ff0000") < luminance("#00ff00")


class TestCLIIntegration:
    def test_single_layer(self, tmp_path):
        runner = CliRunner()
        img_path = str(tmp_path / "layer.png")
        make_image(img_path, (255, 0, 0))
        out = str(tmp_path / "out.json")

        result = runner.invoke(main, ["-l", img_path, "-o", out])
        assert result.exit_code == 0
        data = json.loads(Path(out).read_text())
        assert len(data["image"]) == 4
        assert len(data["image"][0]) == 4
        assert set(c for row in data["image"] for c in row) == {"#ff0000"}
        assert data["height_map"] == {"#ff0000": 3.0}

    def test_two_layers_alpha_composite(self, tmp_path):
        runner = CliRunner()
        bottom = str(tmp_path / "red.png")
        top = str(tmp_path / "blue_semi.png")
        out = str(tmp_path / "out.json")

        make_image(bottom, (255, 0, 0))
        make_image(top, (0, 0, 255), alpha=128)

        result = runner.invoke(main, ["-l", bottom, "-l", top, "-o", out])
        assert result.exit_code == 0
        data = json.loads(Path(out).read_text())
        assert "#0000ff" not in data["height_map"]
        assert "#7f0080" in data["height_map"]

    def test_crop(self, tmp_path):
        runner = CliRunner()
        img_path = str(tmp_path / "big.png")
        out = str(tmp_path / "out.json")
        make_image(img_path, (0, 255, 0), width=10, height=10)

        result = runner.invoke(main, ["-l", f"{img_path}:2,3,7,8", "-o", out])
        assert result.exit_code == 0
        data = json.loads(Path(out).read_text())
        assert len(data["image"]) == 5
        assert len(data["image"][0]) == 5

    def test_dimension_mismatch(self, tmp_path):
        runner = CliRunner()
        a = str(tmp_path / "a.png")
        b = str(tmp_path / "b.png")
        out = str(tmp_path / "out.json")
        make_image(a, (255, 0, 0), width=4, height=4)
        make_image(b, (0, 255, 0), width=6, height=6)

        result = runner.invoke(main, ["-l", a, "-l", b, "-o", out])
        assert result.exit_code != 0
        assert "dimensions mismatch" in result.output

    def test_missing_file(self, tmp_path):
        runner = CliRunner()
        out = str(tmp_path / "out.json")
        result = runner.invoke(main, ["-l", "/nonexistent/path.png", "-o", out])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_pixel_height_step(self, tmp_path):
        runner = CliRunner()
        a = str(tmp_path / "a.png")
        b = str(tmp_path / "b.png")
        out = str(tmp_path / "out.json")
        make_image(a, (255, 0, 0), alpha=128)
        make_image(b, (0, 255, 0), alpha=128)

        result = runner.invoke(
            main, ["-l", a, "-l", b, "-o", out, "-p", "10", "-s", "2"]
        )
        assert result.exit_code == 0
        data = json.loads(Path(out).read_text())
        heights = list(data["height_map"].values())
        assert heights[0] == pytest.approx(10.0)
        assert heights[-1] == pytest.approx(10.0 + (len(heights) - 1) * 2)

    def test_height_map_sorted_by_brightness(self, tmp_path):
        runner = CliRunner()
        bottom = str(tmp_path / "bottom.png")
        top = str(tmp_path / "top.png")
        out = str(tmp_path / "out.json")
        make_image(bottom, (0, 0, 255))
        make_image(top, (255, 0, 0), alpha=128)

        result = runner.invoke(main, ["-l", bottom, "-l", top, "-o", out])
        assert result.exit_code == 0
        data = json.loads(Path(out).read_text())
        colors = list(data["height_map"].keys())
        heights = list(data["height_map"].values())
        lum_values = [luminance(c) for c in colors]
        assert lum_values == sorted(lum_values)
        assert heights == sorted(heights)

    def test_missing_output(self):
        runner = CliRunner()
        result = runner.invoke(main, ["-l", "/tmp/whatever.png"])
        assert result.exit_code != 0

    def test_no_layers(self):
        runner = CliRunner()
        result = runner.invoke(main, ["-o", "/tmp/out.json"])
        assert result.exit_code != 0
