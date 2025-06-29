# 4D Tesseract Visualizer

Ever wanted to rotate a hypercube like a god? This interactive Python project lets you explore the 4th dimension without collapsing into a singularity.

## 🔭 What It Does

* Projects a 4D hypercube (a tesseract) into 3D space, and then into 2D
* Lets you rotate it along all six orthogonal planes (XY, XZ, YZ, XW, YW, ZW)
* Interactive sliders built with `ipywidgets`
* Plots update live, so you can finally *see* the invisible dimensions people keep tweeting about

## 🚀 Try It Online (Binder)

Don’t feel like installing anything? Just click the badge below and run the notebook in your browser:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mbagalman/Tesseract/main?filepath=tesseract_visualizer.ipynb)

## 🧹 Installation (for mortals)

If you want to run this locally:

```bash
git clone https://github.com/mbagalman/tesseract-visualizer.git
cd tesseract-visualizer
pip install -r requirements.txt
jupyter notebook
```

Then open `tesseract_visualizer.ipynb` and rotate to your heart’s content.

## 🧪 Dependencies

* `numpy`
* `matplotlib`
* `ipywidgets`

All safely specified in `requirements.txt`.

## 💡 Why?

Because visualizing 4D geometry helps build intuition. And also because cubes are boring.

## 👨‍💻 Author

Made with math, matplotlib, and mild sarcasm by Michael Bagalman.

## 🪪 License

MIT License — use it, fork it, teach with it, just don’t sell NFTs of the tesseract and blame me when it unravels spacetime.
