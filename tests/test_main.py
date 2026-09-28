import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from PIL import Image

from two_point_five_d_info.main import luminance, main, pad, parse_layer, shift_layer


def make_image(path, color, width=4, height=4, alpha=255):
    img = Image.new("RGBA", (width, height), (*color, alpha))
    img.save(path)
    return path


class TestParseLayer:
    def test_path_only(self):
        result = parse_layer("image.png")
        assert result == ("image.png", None, (0, 0), (0, 0))

    def test_path_with_crop(self):
        result = parse_layer("image.png:10,20,30,40")
        assert result == ("image.png", (10, 20, 30, 40), (0, 0), (0, 0))

    def test_path_with_spaces_in_crop(self):
        result = parse_layer("image.png:10, 20, 30, 40")
        assert result == ("image.png", (10, 20, 30, 40), (0, 0), (0, 0))

    def test_invalid_crop_count(self):
        with pytest.raises(click.BadParameter):
            parse_layer("image.png:10,20,30")

    def test_path_with_shift(self):
        result = parse_layer("image.png@3")
        assert result == ("image.png", None, (3, 0), (0, 0))

    def test_path_with_negative_shift(self):
        result = parse_layer("image.png@-2")
        assert result == ("image.png", None, (-2, 0), (0, 0))

    def test_path_with_vertical_shift(self):
        result = parse_layer("image.png@0,-2")
        assert result == ("image.png", None, (0, -2), (0, 0))

    def test_path_with_both_shifts(self):
        result = parse_layer("image.png@3,-2")
        assert result == ("image.png", None, (3, -2), (0, 0))

    def test_path_with_crop_and_shift(self):
        result = parse_layer("image.png:10,20,30,40@-2")
        assert result == ("image.png", (10, 20, 30, 40), (-2, 0), (0, 0))

    def test_path_with_pad(self):
        result = parse_layer("image.png#32")
        assert result == ("image.png", None, (0, 0), (32, 0))

    def test_path_with_both_pads(self):
        result = parse_layer("image.png#32,32")
        assert result == ("image.png", None, (0, 0), (32, 32))

    def test_path_with_negative_pad_is_positive(self):
        result = parse_layer("image.png#-32,-32")
        assert result == ("image.png", None, (0, 0), (32, 32))

    def test_path_with_crop_pad_shift(self):
        result = parse_layer("image.png:10,20,30,40#32,0@1,-2")
        assert result == ("image.png", (10, 20, 30, 40), (1, -2), (32, 0))

    def test_invalid_shift(self):
        with pytest.raises(click.BadParameter):
            parse_layer("image.png@abc")

    def test_invalid_shift_count(self):
        with pytest.raises(click.BadParameter):
            parse_layer("image.png@1,2,3")

    def test_invalid_pad(self):
        with pytest.raises(click.BadParameter):
            parse_layer("image.png#abc")

    def test_invalid_pad_count(self):
        with pytest.raises(click.BadParameter):
            parse_layer("image.png#1,2,3")


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


class TestShiftLayer:
    def test_zero_is_identity(self):
        img = make_image_obj((255, 0, 0), width=4, height=3)
        assert shift_layer(img, 0, 0) is img

    def test_shift_right(self):
        img = make_image_obj((255, 0, 0), width=4, height=1)
        out = shift_layer(img, 1, 0)
        pixels = [out.getpixel((x, 0)) for x in range(4)]
        assert pixels[0] == (0, 0, 0, 0)
        assert pixels[1:] == [(255, 0, 0, 255)] * 3

    def test_shift_left(self):
        img = make_image_obj((0, 255, 0), width=4, height=1)
        out = shift_layer(img, -1, 0)
        pixels = [out.getpixel((x, 0)) for x in range(4)]
        assert pixels[:3] == [(0, 255, 0, 255)] * 3
        assert pixels[3] == (0, 0, 0, 0)

    def test_shift_down(self):
        img = make_image_obj((255, 0, 0), width=1, height=4)
        out = shift_layer(img, 0, 1)
        pixels = [out.getpixel((0, y)) for y in range(4)]
        assert pixels[0] == (0, 0, 0, 0)
        assert pixels[1:] == [(255, 0, 0, 255)] * 3

    def test_shift_up(self):
        img = make_image_obj((0, 255, 0), width=1, height=4)
        out = shift_layer(img, 0, -1)
        pixels = [out.getpixel((0, y)) for y in range(4)]
        assert pixels[:3] == [(0, 255, 0, 255)] * 3
        assert pixels[3] == (0, 0, 0, 0)

    def test_shift_beyond_width(self):
        img = make_image_obj((0, 0, 255), width=2, height=1)
        out = shift_layer(img, 5, 0)
        assert all(out.getpixel((x, 0)) == (0, 0, 0, 0) for x in range(2))

    def test_shift_beyond_height(self):
        img = make_image_obj((0, 0, 255), width=1, height=2)
        out = shift_layer(img, 0, 5)
        assert all(out.getpixel((0, y)) == (0, 0, 0, 0) for y in range(2))


def make_image_obj(color, width=4, height=4, alpha=255):
    return Image.new("RGBA", (width, height), (*color, alpha))


class TestPad:
    def test_zero_is_identity(self):
        img = make_image_obj((255, 0, 0), width=4, height=4)
        assert pad(img, 0, 0) is img

    def test_pad_x_even(self):
        img = make_image_obj((255, 0, 0), width=1, height=1)
        out = pad(img, 2, 0)
        assert out.size == (3, 1)
        cols = [out.getpixel((x, 0)) for x in range(3)]
        assert cols == [(0, 0, 0, 0), (255, 0, 0, 255), (0, 0, 0, 0)]

    def test_pad_x_odd_extra_right(self):
        img = make_image_obj((255, 0, 0), width=1, height=1)
        out = pad(img, 3, 0)
        assert out.size == (4, 1)
        cols = [out.getpixel((x, 0)) for x in range(4)]
        assert cols == [(0, 0, 0, 0), (255, 0, 0, 255), (0, 0, 0, 0), (0, 0, 0, 0)]

    def test_pad_y(self):
        img = make_image_obj((0, 255, 0), width=1, height=1)
        out = pad(img, 0, 2)
        assert out.size == (1, 3)
        rows = [out.getpixel((0, y)) for y in range(3)]
        assert rows == [(0, 0, 0, 0), (0, 255, 0, 255), (0, 0, 0, 0)]

    def test_pad_both_axes(self):
        img = make_image_obj((0, 0, 255), width=2, height=2)
        out = pad(img, 2, 2)
        assert out.size == (4, 4)


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

    def test_layer_shift_right(self, tmp_path):
        runner = CliRunner()
        img_path = str(tmp_path / "layer.png")
        out = str(tmp_path / "out.json")
        make_image(img_path, (255, 0, 0), width=4, height=2)

        result = runner.invoke(main, ["-l", f"{img_path}@1", "-o", out, "-sa"])
        assert result.exit_code == 0
        data = json.loads(Path(out).read_text())
        assert data["image"][0][0] is None
        assert data["image"][0][1:] == ["#ff0000"] * 3

    def test_layer_shift_left(self, tmp_path):
        runner = CliRunner()
        img_path = str(tmp_path / "layer.png")
        out = str(tmp_path / "out.json")
        make_image(img_path, (0, 255, 0), width=4, height=2)

        result = runner.invoke(main, ["-l", f"{img_path}@-1", "-o", out, "-sa"])
        assert result.exit_code == 0
        data = json.loads(Path(out).read_text())
        assert data["image"][0][:3] == ["#00ff00"] * 3
        assert data["image"][0][3] is None

    def test_layer_shift_down(self, tmp_path):
        runner = CliRunner()
        img_path = str(tmp_path / "layer.png")
        out = str(tmp_path / "out.json")
        make_image(img_path, (255, 0, 0), width=2, height=4)

        result = runner.invoke(main, ["-l", f"{img_path}@0,1", "-o", out, "-sa"])
        assert result.exit_code == 0
        data = json.loads(Path(out).read_text())
        assert all(c is None for c in data["image"][0])
        assert all(c == "#ff0000" for c in data["image"][1])

    def test_layer_pad(self, tmp_path):
        runner = CliRunner()
        img_path = str(tmp_path / "layer.png")
        out = str(tmp_path / "out.json")
        make_image(img_path, (255, 0, 0), width=2, height=2)

        result = runner.invoke(main, ["-l", f"{img_path}#2,2", "-o", out, "-sa"])
        assert result.exit_code == 0
        data = json.loads(Path(out).read_text())
        assert len(data["image"]) == 4
        assert len(data["image"][0]) == 4
        assert data["image"][1][1] == "#ff0000"
        assert all(c is None for c in data["image"][0])
        assert data["image"][0][0] is None
        assert data["image"][3][3] is None

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
