# 🚀 21 Days, 21 Projects — Machine Learning · Deep Learning · Generative AI

[![Project: 21 in 21](https://img.shields.io/badge/21%20Projects-21%20Days-brightgreen?style=for-the-badge&logo=python)](https://github.com/Ritam-910/21_DAYS_21_PROJECTS)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Open In Colab](https://img.shields.io/badge/Open%20in-Colab-orange?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/Ritam-910/21_DAYS_21_PROJECTS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](./LICENSE)

✨ A curated, hands-on journey: 21 practical, self-contained Python projects delivered over 21 days — covering classic machine learning, deep learning, and modern generative AI workflows. All notebooks are written to run in Google Colab for maximum accessibility.

---

## 📌 What this repo is
This repository contains 21 Jupyter/Colab notebooks (one per day) that progressively build your skills in:
- Fundamental machine learning (regression, classification, model selection)
- Deep learning (CNNs, RNNs, Transformers)
- Generative AI (VAEs, GANs, diffusion models / Stable Diffusion)

All notebooks are authored in Python and designed for Google Colab (GPU/TPU-ready where applicable). Each notebook is meant to be executable start-to-finish, with explanations, visualizations, and reproducible code.

---

## 📚 Highlights & Goals
- Learn by doing: short, focused projects with clear learning objectives.
- Reproducible Colab notebooks so anyone can run them instantly.
- Emphasis on practical skills: datasets, preprocessing, training, evaluation, and deployment.
- Modern content: Hugging Face Transformers, PyTorch/TensorFlow examples, and generative pipelines.

---

## ▶️ How to run a notebook (Colab)
1. Click the "Open In Colab" badge at the top or open a notebook directly:
   - Pattern: https://colab.research.google.com/github/Ritam-910/21_DAYS_21_PROJECTS/blob/main/DayXX/DayXX_Project-Title.ipynb
2. If the notebook needs a GPU: Runtime → Change runtime type → Hardware accelerator → GPU.
3. Run cells top-to-bottom. Add `!pip install -r requirements.txt` if a requirements file is provided or run per-notebook install cells.

Tip: Add the query parameter `?authuser=1` if you have multiple Google accounts and want Colab to use a specific one.

---

## 🧩 Requirements & environment
Colab includes many libraries by default. If needed, each notebook contains cells to install required packages. Example requirements snippet:

```bash
# Example (run in a Colab cell)
!pip install -q torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
!pip install -q tensorflow==2.12
!pip install -q transformers datasets accelerate diffusers
!pip install -q shap lime streamlit scikit-learn matplotlib seaborn
```

If you run locally, create a virtualenv and use `pip install -r requirements.txt`.

---

## ✅ Contributing
Contributions are welcome!
- Open an issue if you find bugs or want to discuss a project idea.
- Prefer small, focused PRs: add a notebook, fix docs, add assets.
- Follow the existing notebook naming convention and keep notebooks reproducible in Colab.
- Add references / dataset sources where applicable.

Suggested workflow:
1. Fork the repo
2. Add your DayXX notebook or improvements
3. Open a PR describing changes and include sample outputs/screenshots

---

## 📝 License
This project is distributed under the MIT License. See [LICENSE](./LICENSE) for details.

---

## 🙏 Acknowledgements & Resources
- TensorFlow, PyTorch, Hugging Face, Colab, scikit-learn, and the open datasets community.
- Tutorials, official docs, and many great community notebooks that inspired content.

---

## 📬 Contact & Follow
Maintainer: [Ritam-910](https://github.com/Ritam-910)  
Questions / collabs: open an issue or reach out via GitHub.

---
