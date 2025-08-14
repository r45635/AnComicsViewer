#!/usr/bin/env python3
"""Generate logo and icon files for AnComicsViewer.

Creates a simple, clean logo featuring:
- Comic panel grid (3x2 layout)
- Modern flat design
- High contrast for readability
- Multiple sizes for different uses

Outputs:
- logo.png (256x256) - main logo
- icon.ico (multi-size) - Windows icon
- icon.png (128x128) - general purpose icon
"""

import os
from PIL import Image, ImageDraw, ImageFont
import io

def create_logo(size=256, bg_color=(255, 255, 255), panel_color=(45, 45, 45), 
                accent_color=(0, 120, 215), text_color=(45, 45, 45)):
    """Create the AnComicsViewer logo."""
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate panel grid dimensions (3x2 comic panels)
    margin = size // 8
    grid_width = size - 2 * margin
    grid_height = int(grid_width * 0.65)  # rectangular aspect ratio like comics
    start_x = margin
    start_y = (size - grid_height) // 2
    
    # Panel dimensions
    panel_width = grid_width // 3 - 6  # 3 columns with gaps
    panel_height = grid_height // 2 - 6  # 2 rows with gaps
    gap = 6
    
    # Draw 6 comic panels in 3x2 grid
    for row in range(2):
        for col in range(3):
            x = start_x + col * (panel_width + gap)
            y = start_y + row * (panel_height + gap)
            
            # Highlight one panel to show "detection"
            color = accent_color if (row == 0 and col == 1) else panel_color
            border_width = 3 if color == accent_color else 1
            
            # Draw panel with rounded corners
            draw.rounded_rectangle([x, y, x + panel_width, y + panel_height], 
                                 radius=4, fill=color, outline=bg_color, width=border_width)
            
            # Add simple content lines to suggest text/artwork
            if color != accent_color:
                line_y = y + 8
                for _ in range(3):
                    draw.rectangle([x + 6, line_y, x + panel_width - 6, line_y + 2], 
                                 fill=(200, 200, 200))
                    line_y += 8
    
    return img

def create_simple_icon(size=128):
    """Create a simplified icon version (just the panel grid)."""
    return create_logo(size, bg_color=(240, 240, 240))

def save_ico(img_256, filename):
    """Save as Windows ICO with multiple sizes."""
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = []
    
    for size in sizes:
        resized = img_256.resize((size, size), Image.Resampling.LANCZOS)
        images.append(resized)
    
    # Save as ICO
    images[0].save(filename, format='ICO', sizes=[(img.width, img.height) for img in images])

def main():
    """Generate all logo files."""
    print("Generating AnComicsViewer logo and icons...")
    
    # Create main logo (256x256)
    logo = create_logo(256)
    logo.save('logo.png')
    print("✓ logo.png (256x256)")
    
    # Create general icon (128x128)
    icon = create_simple_icon(128)
    icon.save('icon.png')
    print("✓ icon.png (128x128)")
    
    # Create Windows ICO
    try:
        save_ico(logo, 'icon.ico')
        print("✓ icon.ico (multi-size)")
    except Exception as e:
        print(f"! icon.ico failed: {e}")
    
    # Create favicon (32x32)
    favicon = logo.resize((32, 32), Image.Resampling.LANCZOS)
    favicon.save('favicon.png')
    print("✓ favicon.png (32x32)")
    
    print("\nLogo concept:")
    print("- Comic panel grid (3x2 layout)")
    print("- One highlighted panel (detection focus)")
    print("- Clean, modern flat design")
    print("- High contrast for visibility")

if __name__ == "__main__":
    main()
