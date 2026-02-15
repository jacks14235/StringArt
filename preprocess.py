from PIL import Image

# crop image to a circle with white background
# padding is the number of pixels to add beyond the circle
def circle_crop(img, size, padding=0):
    """
    Crop the input image to a circle of given size, pad with white beyond the circle.
    Args:
        img: Input PIL.Image.
        size: Final square output size (pixels).
        padding: # of extra pixels of white beyond the circle.
    Returns:
        PIL.Image object, size=(size, size), with a circular crop.
    """
    # Convert to RGBA to allow transparency
    img = img.convert("RGBA")

    # Resize and center to fit inside a circle with optional padding
    circle_diam = size - 2 * padding
    min_side = min(img.width, img.height)
    scale = circle_diam / min_side
    new_w, new_h = int(round(img.width * scale)), int(round(img.height * scale))
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    # Create white square background
    bg = Image.new("RGBA", (size, size), (255, 255, 255, 255))

    # Compute position to paste
    offset_x = (size - new_w) // 2
    offset_y = (size - new_h) // 2
    bg.paste(img, (offset_x, offset_y), img)

    # Create circular alpha mask
    mask = Image.new("L", (size, size), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    ellipse_bounds = (padding, padding, size - padding, size - padding)
    draw.ellipse(ellipse_bounds, fill=255)

    # Composite with mask to make transparent corners
    bg.putalpha(mask)

    # Paste on white background to remove any transparency, get back to 'RGB'
    white_bg = Image.new("RGB", (size, size), (255, 255, 255))
    white_bg.paste(bg, mask=bg.split()[-1])

    return white_bg

def main():
    img = Image.open('lion.jpg')
    img = circle_crop(img, 512, 64)
    img.save('lion_circle.jpg')

if __name__ == '__main__':
    main()