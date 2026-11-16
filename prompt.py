"""Base system prompt and local (zero-cost) image color extraction."""

import io
from PIL import Image

BASE_SYSTEM_PROMPT = """You are given two input images and video context:
- IMAGE 1 (Subject Frame): The presenter, product, or scene from the video.
- IMAGE 2 (Reference Thumbnail): A design & layout blueprint ONLY (spatial grid, depth layering, badge/pill geometry, lighting contrast, and composition hierarchy). IMAGE 2 may come from a completely different game, brand, or topic.

GOAL:
Generate a 1080x1920 vertical 9:16 social/reel thumbnail cover that extracts IMAGE 2's abstract graphic design template and layout structure, features the exact subject/product from IMAGE 1, and uses ONLY the game/brand identity, logos, world theme, and copy from the VIDEO CONTEXT below.

STRICT RULES (100% FIDELITY):
1. SUBJECT IDENTITY (FROM IMAGE 1):
   - Preserve the exact facial features, expression, hair, outfit, logos on clothing, pose, and any held controller/product from IMAGE 1 with 100% fidelity. Do NOT replace or genericize the presenter or product.
   - Cleanly segment the subject from IMAGE 1 and place them at IMAGE 2's foreground subject position, scale, and camera framing (matching any rim light, drop shadow, or depth separation style).

2. TEMPLATE EXTRACTION vs. ZERO REFERENCE IP CARRYOVER (FROM IMAGE 2):
   - EXTRACT ONLY THE DESIGN SYSTEM: Replicate IMAGE 2's spatial grid, element placement, depth layering (e.g., top split-pill badge, large hero title/logo slot behind the subject's head/torso, foreground subject cutout, angled side hook text), badge/pill container geometry, outline/shadow weight, and studio lighting contrast.
   - EXCLUDE ALL REFERENCE IP, LOGOS & SYMBOLS: NEVER copy any game/brand logo, Roman numerals (e.g., "VI"), crests, icons, mascots, or watermark shapes from IMAGE 2. Where IMAGE 2 has a hero emblem/logo behind the subject, replace that entire graphic with the authentic emblem/logo and visual identity of the target game/topic specified in VIDEO CONTEXT.
   - EXCLUDE REFERENCE FRANCHISE FONTS ON HERO TITLES: Do NOT write the target game/topic title in IMAGE 2's franchise-specific typeface. Render the hero title using the target franchise's own authentic logo/typography style, while keeping IMAGE 2's generic UI typography role for secondary CTA pills and side callouts.
   - EXCLUDE REFERENCE WORLD / SCENERY / PROPS: NEVER copy literal background scenes, animals, locations, vehicles, or characters from IMAGE 2's franchise. Build the background environment around the target game/topic from VIDEO CONTEXT (or an enhanced, studio-lit version of IMAGE 1's setting) with IMAGE 2's depth-of-field and color-contrast mood.

3. TEXT & HOOK (FROM VIDEO CONTEXT):
   - Replace all text slots with copy derived strictly from the VIDEO CONTEXT (spelled 100% accurately, crisp, high-contrast, no gibberish, no extra UI icons).
   - Keep the top/bottom safe zones clean; NEVER place any text, pill, badge, icon, or graphic over the subject's face or eyes.

4. OUTPUT QUALITY:
   - Photorealistic subject integration, razor-sharp typography, clean edges with zero cutout artifacts or floating background fragments, studio-grade contrast."""


def extract_rgb_colors(image_bytes: bytes, max_colors: int = 4) -> list[list[int]]:
    """Extract dominant RGB colors locally via Pillow quantization ($0.00 cost)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
    quantized = img.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
    counts = sorted(quantized.getcolors(maxcolors=max_colors) or [], key=lambda x: -x[0])
    return [list(rgb) for _, rgb in counts[:max_colors]]


def build_prompt(
    video_context: str,
    user_message: str = "",
    *,
    has_frame: bool = False,
    has_ref: bool = False,
    ref_colors: list[list[int]] | None = None,
) -> str:
    """Combine the base system prompt with video context and follow-up instructions."""
    parts = [BASE_SYSTEM_PROMPT]
    if ref_colors:
        hexes = ", ".join(f"#{r:02X}{g:02X}{b:02X}" for r, g, b in ref_colors)
        parts.append(
            f"REFERENCE PALETTE HINT (from IMAGE 2 for contrast/mood only; override with target brand/game colors when appropriate): {hexes}"
        )
    parts.append(f"INPUTS PROVIDED: frame_image={has_frame}, reference_image={has_ref}")
    if video_context.strip():
        parts.append(f"VIDEO CONTEXT:\n{video_context.strip()}")
    if user_message.strip():
        parts.append(f"USER INSTRUCTION:\n{user_message.strip()}")
    return "\n\n".join(parts)
