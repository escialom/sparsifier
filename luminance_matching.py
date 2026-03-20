import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from config import model_config


def luminance_matching(input_dir, output_dir, threshold=0.01):

    """
    Match the luminance statistics of phosphene regions across images from the same density.

    This function performs mask-aware luminance matching on grayscale PNG
    images. Only pixels above `threshold` are considered part of the phosphene
    foreground; pixels below or equal to this threshold are treated as
    background and excluded from the luminance normalization step.

    The image with the highest overall luminance in `input_dir` is selected as
    the reference image. For every other image, the mean and standard deviation
    of the foreground pixels are adjusted to match those of the reference
    foreground. Background pixels remain zero.

    Luminance matching should be performed separately for each phosphene density condition.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing the input PNG images.
    output_dir : str or Path
        Directory where the luminance-matched images will be saved. The matched
        images are stored in the subfolder `luminance_matched`.
    threshold : float, optional
        Pixel-intensity threshold used to distinguish foreground phosphene
        regions from background. Pixels with values above this threshold are
        included in the luminance matching. Default is 0.01.

    Returns
    -------
    None
    """

    transform = transforms.ToTensor()
    input_dir, output_dir = Path(input_dir), Path(output_dir)

    # Select the brightest image as reference
    path_reference_img, _ = find_highest_luminance_img(input_dir)
    ref_pil = Image.open(path_reference_img).convert("L")
    ref = transform(ref_pil)

    # Select the brightest image as reference
    ref_mask = ref > threshold
    ref_mean = ref[ref_mask].mean()
    ref_std = ref[ref_mask].std()
    print(f"Reference: {path_reference_img.name}, mean={ref_mean:.6f}, std={ref_std:.6f}")

    out_dir = output_dir / "luminance_matched"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the reference image unchanged into the output folder
    ref_img_output_path = out_dir / path_reference_img.name
    save_image(ref, ref_img_output_path)
    print(f"Saved reference image to: {ref_img_output_path}")

    # Process all remaining images
    for path in input_dir.rglob("*.png"):
        if path == path_reference_img:
            continue  # skip reprocessing the reference

        img = transform(Image.open(path).convert("L"))
        mask = img > threshold
        # skip blank images with no foreground pixels
        if mask.sum() == 0:
            continue
        # Compute luminance statistics on foreground pixels only
        img_mean = img[mask].mean()
        img_std = img[mask].std()

        # Normalize foreground pixels and rescale them to match the reference
        matched = torch.zeros_like(img)
        matched[mask] = (img[mask] - img_mean) / (img_std + 1e-8)
        matched[mask] = matched[mask] * ref_std + ref_mean

        # Keep background black and clip the result to a valid image range
        matched = matched.clamp(0.0, 1.0)

        save_image(matched, out_dir / path.name)
        print(f"{path.name}: mean before={img_mean:.6f}, after={matched[mask].mean():.6f}")

def find_highest_luminance_img(input_dir):

    """
    Find the image with the highest total luminance in a directory tree.

    This function searches recursively through `input_dir` for PNG images,
    computes the sum of pixel intensities for each image, and returns the path
    of the brightest image together with its luminance value.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing PNG images.

    Returns
    -------
    path_highest_luminance_img : Path
        Path to the image with the highest total luminance.
    highest_luminance : float
        Sum of pixel intensities of the brightest image.
    """

    transform = transforms.ToTensor()
    all_images = list(input_dir.rglob('*.png'))
    highest_luminance = 1e-8
    for image_path in all_images:
        img = Image.open(image_path)
        img = transform(img)
        curr_img_luminance = img.sum()
        if curr_img_luminance > highest_luminance:
            highest_luminance = curr_img_luminance.item()
            path_highest_luminance_img = image_path

    return path_highest_luminance_img, highest_luminance


if __name__ == "__main__":
    args = model_config.parse_arguments()

    base_path = Path(args.output_path) / "stimuli"

    densities = [12, 16, 21, 27, 35, 46, 59, 77, 100]

    for d in densities:
        input_dir = base_path / f"All_stim_before_luminance{d}"
        output_dir = base_path / f"All_stim_after_luminance{d}"

        print(f"\nProcessing phosphene density: {d}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")

        luminance_matching(input_dir, output_dir)