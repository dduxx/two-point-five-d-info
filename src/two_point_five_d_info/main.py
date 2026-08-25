import json
from typing import Any, Dict, List, Optional, Set, Tuple

import click
from PIL import Image


def luminance(hex_color: str) -> float:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return 0.299 * r + 0.587 * g + 0.114 * b


def parse_layer(value: str) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
    if ":" in value:
        path, crop_str = value.rsplit(":", 1)
        parts = crop_str.split(",")
        if len(parts) != 4:
            raise click.BadParameter(
                f"crop must be 'left,upper,right,lower', got '{crop_str}'"
            )
        try:
            crop: Tuple[int, int, int, int] = tuple(int(p.strip()) for p in parts)
        except ValueError:
            raise click.BadParameter(
                f"crop coordinates must be integers, got '{crop_str}'"
            )
        return (path, crop)
    return (value, None)


@click.command()
@click.option(
    "-l",
    "--layer",
    required=True,
    multiple=True,
    help="Image path, optionally with crop: 'path[:left,upper,right,lower]'",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(writable=True),
    help="Path for the output JSON file",
)
@click.option(
    "-p",
    "--pixel-height",
    type=float,
    default=3.0,
    show_default=True,
    help="Starting height for the darkest color",
)
@click.option(
    "-s",
    "--pixel-step",
    type=float,
    default=0.5,
    show_default=True,
    help="Height increment between each brightness rank",
)
@click.option(
    "-sa",
    "--skip-zero-alpha",
    is_flag=True,
    default=False,
    help="Replace fully transparent pixels with null in the output",
)
@click.option(
    "-ro",
    "--reverse-order",
    is_flag=True,
    default=False,
    help="Sort height_map by descending brightness instead of ascending",
)
@click.option(
    "-mc",
    "--max-colors",
    type=int,
    default=None,
    help="Reduce the composite image to at most this many unique colors",
)
def main(
    layer: Tuple[str, ...],
    output: str,
    pixel_height: float,
    pixel_step: float,
    skip_zero_alpha: bool,
    reverse_order: bool,
    max_colors: Optional[int],
) -> None:
    """Combine image layers and produce a JSON height map.

    Layers are alpha-composited bottom-to-top in the order given.
    Each -l flag accepts a path with an optional PIL-style crop region
    after a colon: 'image.png:left,upper,right,lower'.
    """
    layers: List[Tuple[str, Optional[Tuple[int, int, int, int]]]] = [
        parse_layer(val) for val in layer
    ]

    images: List[Image.Image] = []
    for path, crop in layers:
        try:
            img: Image.Image = Image.open(path)
        except FileNotFoundError:
            raise click.BadParameter(f"file not found: '{path}'")
        except Exception as exc:
            raise click.BadParameter(f"cannot open '{path}': {exc}")

        if img.mode != "RGBA":
            img = img.convert("RGBA")

        if crop:
            img = img.crop(crop)

        images.append(img)

    if not images:
        raise click.BadParameter("at least one layer is required")

    width, height = images[0].size
    for i, img in enumerate(images):
        if img.size != (width, height):
            raise click.BadParameter(
                f"layer dimensions mismatch: layer 1 is {width}x{height}, "
                f"layer {i + 1} ('{layers[i][0]}') is {img.size[0]}x{img.size[1]}"
            )

    composite: Image.Image = images[0].copy()
    for img in images[1:]:
        composite = Image.alpha_composite(composite, img)

    if max_colors is not None:
        quantized = composite.convert("RGB").quantize(colors=max_colors).convert("RGBA")
        quantized.putalpha(composite.split()[-1])
        composite = quantized

    pixels: Any = composite.load()
    image_grid: List[List[Optional[str]]] = []
    unique_hex: Set[str] = set()

    for y in range(height):
        row: List[Optional[str]] = []
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if skip_zero_alpha and a == 0:
                row.append(None)
            else:
                hex_code: str = f"#{r:02x}{g:02x}{b:02x}"
                row.append(hex_code)
                unique_hex.add(hex_code)
        image_grid.append(row)

    sorted_colors: List[str] = sorted(unique_hex, key=luminance, reverse=reverse_order)
    height_map: Dict[str, float] = {}
    for i, color in enumerate(sorted_colors):
        height_map[color] = pixel_height + i * pixel_step

    result: Dict[str, object] = {
        "image": image_grid,
        "height_map": height_map,
    }

    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    click.echo(f"Output written to {output}")
    click.echo(f"  Image: {width}x{height} pixels")
    click.echo(f"  Unique colors in height_map: {len(height_map)}")


if __name__ == "__main__":
    main()
