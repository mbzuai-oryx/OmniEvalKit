# TFO project page

Static GitHub Pages website for:

**Training-Free Speech-Centric Omni Understanding with Frozen VLMs**

## Preview locally

From this directory:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

## Structure

- `index.html` — page content and semantic structure
- `styles.css` — responsive visual design
- `script.js` — navigation, reveal effects, and interactive result charts
- `assets/teaser_TFO.mp4` — teaser video
- `assets/teaser_poster.jpg` — generated video poster frame
- `assets/TFO_logo.png` — paper logo
- `assets/mbzuai_logo.png` — institute logo

The site has no build step and can be served directly from a GitHub Pages branch or `/docs`-style folder.

## Add the paper link after publication

In `index.html`, find the comment `Replace # with the paper's arXiv URL after publication`.
Replace the `href="#"` value with the final arXiv URL, remove `aria-disabled="true"`,
and remove the `arXiv soon` status text.
