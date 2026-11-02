# Camera View Adjustment Prediction for Improving Image Composition

Repository containing the research paper, one-page architecture summary, and public benchmark datasets for:

> **Camera View Adjustment Prediction for Improving Image Composition**  
> Yu-Chuan Su, Raviteja Vemulapalli, Ben Weiss, Chun-Te Chu, Philip Andrew Mansfield, Lior Shapira, Colvin Pitts *(Google Research, 2021)*  
> **Paper**: [arXiv:2104.07608](https://arxiv.org/pdf/2104.07608) (`2104.07608_Camera_View_Adjustment.pdf`)

---

## 📁 Repository Contents

- [`2104.07608_Camera_View_Adjustment.pdf`](./2104.07608_Camera_View_Adjustment.pdf) — Full paper PDF (including supplementary appendices A–F).
- [`Architecture_Summary.md`](./Architecture_Summary.md) — **One-Page Architecture Summary** covering the two-stage teacher-student semi-supervised pipeline, MobileNet + Spatial Pyramid Pooling (SPP) backbones, pairwise ranking composition scoring model ($M_c$), zero-pixel border augmentations, 72-view simulation pseudo-labeling, and three-head gated view adjustment network ($M_v$).
- [`datasets/`](./datasets/) — Public datasets and official annotation/code repositories used in the paper:
  1. **FCDB (Flickr Cropping Database)** — *Chen et al., WACV 2017*:
     - `datasets/FCDB_images.zip` (321 MB) & `datasets/FCDB_repo/` (1,265 train + 348 test images with ground-truth best crop annotations and pairwise rankings). Used for Stage 1 & Stage 2 training and constructing the **3,076-sample View Adjustment Evaluation Benchmark**.
  2. **GAICD (Grid Anchor Based Image Cropping Dataset)** — *Zeng et al., CVPR 2019 / TPAMI 2020*:
     - `datasets/GAICD_dataset.zip` (117 MB), `datasets/GAICD_repo/`, & `datasets/GAICD_Pytorch_repo/` (1,036 train + 200 test images, each with 90 grid-anchored candidate crops and MOS composition scores). Used for Stage 1 & Stage 2 training and constructing the **View Adjustment Evaluation Benchmark**.
  3. **CPC (Comparative Photo Composition Dataset)** — *Wei et al., CVPR 2018*:
     - `datasets/CPC_dataset.tar.gz` (1.37 GB) & `datasets/CPC_VPN_repo/` (10,797 images with 24 candidate crops and dense comparative composition scores). Used in Stage 1 to train the Composition Scoring Model ($M_c$).
  4. **VAPNet-PyTorch (View Adjustment Benchmark & Perturbation Generator)**:
     - `datasets/VAPNet_repo/` — Community PyTorch implementation of *Camera View Adjustment Prediction* (`generate_dataset.py`, `vapnet.py`, `CSNet/`) that generates the exact 8-way inverse-perturbation View Adjustment dataset from FCDB and GAICD.
  5. **Pano2Vid ($360^\circ$ Video Viewport Dataset)** — *Su et al., ACCV 2016*:
     - `datasets/Pano2Vid_repo/` — Official Pano2Vid code and human viewport trajectory annotations (`panoramic_videos.json`, `humanCam.json`, `human_edit.json`) used for the 200-frame multi-step $360^\circ$ subjective evaluation (Section 4.3).
  6. **Unsplash & Open Images (Unlabeled Web Photo Repositories)**:
     - `datasets/Unsplash_repo/` — Official Unsplash Dataset (`unsplash.com/data`) specification and scripts (used for Stage 1 unlabeled well-composed image perturbations).
     - `datasets/OpenImages_repo/` — Official Open Images Dataset (`github.com/cvdfoundation/open-images-dataset`) specification and download manifest (5.5M human-verified subset used for Stage 2 pseudo-label generation; full raw archive is ~561 GB on AWS S3 `s3://open-images-dataset`).

