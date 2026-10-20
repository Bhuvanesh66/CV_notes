# Module 5 — 3D Vision & Motion: Exam Notes

**Course:** SST Computer Vision (Scaler) · **Sessions:** 11 & 12 — "From Pixels to Points" (3D Vision & Motion)

**Built from:**
- the Session 11–12 slides (70 pages)
- the Session 12 class transcript (auto-generated, so some terms were garbled; see Part 15)
- `question-pattern.pdf` (the exam format)

**Syllabus:**
1. Pinhole camera & perspective projection
2. Intrinsic & extrinsic parameters
3. Camera calibration & distortion
4. Homography revisited
5. Stereo vision & depth from disparity
6. Optical flow & tracking
7. Structure from Motion (SfM)
8. SLAM (conceptual)

> **Coverage note:** there is **no Session 11 transcript**, so the slides were the source for it. Lucas–Kanade, KLT tracking and loop closure were **not on the slides**; they are filled in from standard material and marked **[syllabus add-on]**.
>
> **Preview tip (VS Code):** press `Ctrl+Shift+V` to render the tables and `$…$` maths.

---

## How this module maps to your exam

| Section | Likely from Module 5 | Practise in |
|---|---|---|
| **A** Single MCQ | What K / R / t hold; homography DOF and min points; Z = fB/d; F vs E; epipole; what VO drift is | Part 14-A |
| **B** Multi-select | Intrinsics vs extrinsics; homography assumptions; stereo failure cases; VO assumptions; F/E facts; SfM ambiguities | Part 14-B |
| **C** Descriptive (≤ 100 words) | Why calibrate; why 4 points; baseline trade-off; epipolar constraint; aperture problem; scale ambiguity; VO vs SfM vs SLAM; loop closure | Part 14-C |
| **D** Coding | **Project a 3D point**, **apply / estimate a homography**, **depth from disparity**, **stereo → 3D point**, reprojection RMSE, Lucas–Kanade, chaining poses | Part 12 (tested) |
| **E** Case study | Parking BEV, ADAS depth, warehouse robot SLAM, drone mapping, document scanner, AR | Part 13 |
| **F** Numericals ★ | u = f·X/Z + c; Z = fH/h; Z = fB/d and its error; DOF counts; 2D vs 1D search; distortion; LK solve | Part 11 + Part 10 sheet |

**★ = very likely to be examined.** "Teacher:" = a point stressed in class.

**Contents:**

| Part | Topic |
|---|---|
| 1 | Why 3D vision |
| 2 | Pinhole camera, projection, intrinsics & extrinsics |
| 3 | Calibration & lens distortion |
| 4 | Homography revisited |
| 5 | Stereo vision & depth from disparity |
| 6 | Epipolar geometry & triangulation (+ the full stereo reconstruction pipeline) |
| 7 | Optical flow & tracking |
| 8 | Structure from Motion |
| 9 | Visual odometry & SLAM |
| 10 | Formula sheet |
| 11 | Numericals |
| 12 | Coding |
| 13 | Case studies |
| 14 | Question bank |
| 15 | Traps & checklist |

---

# PART 1 — Why 3D Vision?

- **3D vision recovers the geometry of the world** (where things are, how far, what shape) **from images.**
- **A photo flattens 3D onto a 2D sensor. Depth is the dimension that gets lost.** 3D vision is the art of getting it back.
- Humans do it effortlessly: **two eyes** (stereo), **motion**, and a lifetime of **experience** (priors).
- **Where it's used:**
  - **Autonomous driving:** the distance to every car, cyclist and curb.
  - **Robotics:** grasping, navigation, avoiding obstacles.
  - **AR/VR:** anchoring virtual objects to real surfaces.
  - **Mapping:** city-scale 3D maps.
  - **Drones:** inspection, landing, terrain following.
  - *Common thread: each needs to know **where** things are in 3D, not just **what** they are.*

**2D vs 3D perception:**

| Task | Question | Output |
|---|---|---|
| Classification | What is in the image? | "car" |
| 2D detection | What, and where **in the image**? | a pixel box |
| **3D localisation** | What, and where **in the world**? | **position, size, heading** |

Each step adds geometry. The last one **needs depth, which a single image does not give directly.**

**The complete pipeline — the map of this module:**

```
FORWARD (how an image forms):   World ──[R|t]──► Camera ──K──► Image                     (Parts 2–3)
INVERSE (recovering 3D):        Image(s) → feature matching → stereo / epipolar → triangulation → 3D point cloud   (Parts 4–9)
```

*"Walk left to right to understand the camera, then right to left to undo it."*

---

# PART 2 — Camera Projection: Pinhole, Intrinsics, Extrinsics ★

## 2.1 The pinhole camera

- Eye ↔ camera:
  - **lens ↔ lens** (bends light to a focus)
  - **retina ↔ image sensor** (records where light lands)
  - **seeing ↔ projection** (3D scene becomes a 2D image)
- **Pinhole:** block every ray except those through **one tiny hole**.
  - **Each scene point maps to exactly one image point.**
  - The image comes out **upside-down**.
  - **Image size depends on f**, the hole-to-sensor distance (the focal length).
- **Virtual image plane trick:** imagine the image plane **in front of** the hole at distance f. The image is upright and the maths is the same.
- **"What if the hole gets bigger or smaller?"**
  - **Bigger:** more light, but **blurry** (each scene point spreads into a disc).
  - **Smaller:** sharper but **darker**. If it is far too small, diffraction blurs it again.
  - Real cameras use a **lens** to get both brightness and sharpness.

## 2.2 Perspective projection = similar triangles ★

Two triangles share the angle at the pinhole O, so their sides are in proportion:

$$\frac{y}{f} = \frac{Y}{Z}\quad\Rightarrow\quad x = f\frac{X}{Z},\qquad y = f\frac{Y}{Z}$$

**Perspective is just one division by depth Z.** No matrices are needed yet.

**The depth effect: near looks big, far looks small:**

$$\text{image size} = \frac{f\cdot\text{real size}}{Z}\qquad(\text{size} \propto 1/Z)$$

| Distance | Width of the same car |
|---|---|
| Z = 10 m | 240 px |
| Z = 20 m | 120 px |
| Z = 40 m | 60 px |

**Double the distance, half the size.**
- **Parallel lines** (lane lines) meet at a **vanishing point**.
- A **toy car up close and a real car far away can look identical**: that is the **scale/depth ambiguity** of a single image.

## 2.3 Three coordinate systems

```
World frame (metres, fixed to scene) ──[R | t] (move: rigid motion)──► Camera frame (metres, moves with camera) ──K (project)──► Image frame (pixels, origin top-left)
```

**The camera frame (OpenCV convention):**
- **Origin C** = the camera (pinhole) centre.
- **Z axis** = where the camera looks, the **optical axis**.
- **X right, Y down.**
- The image plane sits at **Z = f**. The optical axis pierces it at the **principal point**.

## 2.4 From metres to pixels: the intrinsics ★

$$u = f_x\frac{X}{Z} + c_x,\qquad v = f_y\frac{Y}{Z} + c_y$$

- $f_x X/Z$ is the similar-triangles result, now in **pixels** instead of metres.
- **+ c_x, c_y** moves the origin from the image centre to the **top-left corner**.

**Intrinsic parameters, "the camera's personality":**
- **f_x, f_y: the focal length in pixels.** How strongly the lens magnifies.
  - **Longer f → narrower view, bigger objects. Shorter f → wide view.**
  - There are **two values because pixels may not be square.**
  - Conversion: $f_{px} = f_{mm}\times\frac{\text{image width (px)}}{\text{sensor width (mm)}}$.
- **c_x, c_y: the principal point.** Where the optical axis hits the sensor; **usually near, but not exactly at, the image centre.**
- **Skew γ:** a 5th intrinsic, almost always 0.
- ★ **Intrinsics belong to the camera. Move the camera and they don't change.** (A zoom lens changing f is the exception.)

$$K = \begin{bmatrix} f_x & \gamma & c_x\\ 0 & f_y & c_y\\ 0 & 0 & 1\end{bmatrix}$$

**Field of view:** $\text{FOV} = 2\arctan\!\left(\frac{W}{2f_x}\right)$. For W = 1280 and f = 640: 2·atan(1) = **90°**.

## 2.5 Extrinsics: where the camera is ★

- **R, the rotation (3×3):** which way the camera points. It has **3 DOF** (roll, pitch, yaw).
- **t, the translation (3×1):** where the camera is. It has **3 DOF**.

$$\mathbf{X}_{cam} = R\,\mathbf{X}_{world} + \mathbf{t}$$

★ **Extrinsics change every time the camera moves.** For example, in a multi-camera car, every camera has its own extrinsics relative to the car body.

| | Intrinsics (K) | Extrinsics (R, t) |
|---|---|---|
| Describes | the camera's **internal optics and sensor** | the camera's **pose in the world** |
| Parameters | f_x, f_y, c_x, c_y (+ skew): **4–5** | R (3) + t (3): **6** |
| Changes when the camera moves? | **No** | **Yes** |
| Maps | camera coordinates (m) → pixels | world (m) → camera (m) |

## 2.6 The complete camera model ★★

```
World point (X,Y,Z) ──Pose [R|t]: rotate + move──► Camera point (Xc,Yc,Zc) ──Intrinsics K: scale, shift, ÷Z──► Pixel (u,v)
```

$$\mathbf{x} \simeq K\,[R\,|\,\mathbf{t}]\,\mathbf{X},\qquad s\begin{bmatrix}u\\v\\1\end{bmatrix} = K\,[R\,|\,\mathbf t]\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}$$

- **"≃" means equal up to scale.** You **divide by the third coordinate**, and that is the ÷Z.
- **P = K[R | t] is the 3×4 camera (projection) matrix.** It has **11 DOF** (12 entries − 1 scale) = 5 intrinsic + 6 extrinsic.
- **Homogeneous coordinates** (an extra 1) turn the division into a matrix multiply. That's why we add the "1".

**Worked example:** K with f = 800 and (c_x, c_y) = (320, 240); R = I, t = 0; point (0.5, −0.25, 4) m.
- u = 800 × 0.5/4 + 320 = **420**
- v = 800 × (−0.25)/4 + 240 = **190**

**With translation:** t = (0, 0, 5) moves the world 5 m in front of the camera. The world point (1, 1, 0) becomes (1, 1, 5), so u = 800·0.2 + 320 = **480** and v = 160 + 240 = **400**.

**A point with Z ≤ 0 is behind the camera** and does not project. (Exam code must handle this.)


---

# PART 3 — Camera Calibration & Lens Distortion ★

**Goal:** find **K, R, t** (and the distortion coefficients) that explain how the camera projects 3D points to pixels.

## 3.1 Why calibrate? A wrong focal length is dangerous

A car is **1.5 m tall** and appears **60 px tall**. How far away is it?

$$Z = \frac{f\cdot H}{h}$$

| Focal length used | Distance |
|---|---|
| **Calibrated** f = 1000 px | 1000 × 1.5/60 = **25 m** |
| **Guessed** f = 800 px | 800 × 1.5/60 = **20 m** |

- That is a **5 m error**. At 50 km/h (13.9 m/s), 5 m = **0.36 s**: *"the difference between braking in time and not."*
- ★ **Every 3D measurement inherits the calibration error.**

## 3.2 The problem

$$\mathbf{x} \simeq K\,[R\,|\,\mathbf t]\,\mathbf X$$

Estimate K, R and t from **known 3D points** (known relative to each other) and their **2D image positions**.

- **4 intrinsic unknowns:** f_x, f_y, c_x, c_y (5 with skew).
- **6 extrinsic unknowns per image:** R (3 DOF) + t (3 DOF).
- Plus the **distortion coefficients**: k₁, k₂, k₃, p₁, p₂.

## 3.3 Why everyone uses a checkerboard

| Property | Why it helps |
|---|---|
| **Known geometry** | Measure the square size once (e.g., 25 mm) and **every corner's 3D position is known** |
| **Precise corners** | Black–white X-junctions can be located to a **fraction of a pixel** |
| **Easy to detect automatically** | High contrast, regular pattern, cheap to print |
| **Flat** | All corners share **Z = 0**, which simplifies the maths (it becomes a **homography**) |

## 3.4 The calibration recipe (the slide's 9 steps)

1. **Checkerboard and 3D points.** Put the **world origin on the board**, so every corner has **Z = 0**. Corner (i, j) sits at **(i·s, j·s, 0)**.
   - Example: 25 mm squares and 9×6 inner corners give (0,0,0), (25,0,0), (50,0,0), …
2. **Capture 10–20 images.** Vary the position, orientation and distance; cover different parts of the image; keep the board fully visible.
3. **Detect and refine the corners:**
   - **Detect:** `cv2.findChessboardCorners(img, (8,5))`
   - **Order:** consistently, row by row, to match the known 3D points
   - **Refine** to sub-pixel accuracy: `cv2.cornerSubPix(...)`
4. **2D–3D correspondences.** Each photo gives about **40 pairs** of (known 3D corner ↔ detected pixel), e.g., (0,0,0) ↔ (152.4, 210.7).
5. **A planar homography per image.** Z = 0, so $s[u,v,1]^\top = H[X,Y,1]^\top$ with **H = K[r₁ r₂ t]**.
6. **Estimate K from several homographies** (Zhang's method, §3.5). K has 5 unknowns; you need **≥ 3 views**.
7. **Recover R and t for each image:**
   - $\mathbf r_1 = \lambda K^{-1}\mathbf h_1$, $\mathbf r_2 = \lambda K^{-1}\mathbf h_2$
   - $\mathbf r_3 = \mathbf r_1\times\mathbf r_2$ (orthogonal to both)
   - $\mathbf t = \lambda K^{-1}\mathbf h_3$
8. **Estimate the lens distortion.** Refine K, R, t and the distortion jointly by **minimising the reprojection error** (non-linear least squares, **Levenberg–Marquardt**).
9. **Result:** K, the distortion coefficients, R and t per image, and the **reprojection error** (good is ~0.1–0.5 px).

`ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)`

## 3.5 Zhang's method: the optimisation idea (concept level)

1. For a planar board, **H = [h₁ h₂ h₃] = λK[r₁ r₂ t]**, so $\mathbf r_1 = \frac1\lambda K^{-1}\mathbf h_1$ and $\mathbf r_2 = \frac1\lambda K^{-1}\mathbf h_2$.
2. **Rotation columns are orthonormal.** This holds for *any* rotation, so it eliminates R:
   - **Orthogonal:** $\mathbf r_1^\top\mathbf r_2 = 0 \Rightarrow \mathbf h_1^\top K^{-\top}K^{-1}\mathbf h_2 = 0$
   - **Equal length:** $\mathbf r_1^\top\mathbf r_1 = \mathbf r_2^\top\mathbf r_2 \Rightarrow \mathbf h_1^\top B\mathbf h_1 = \mathbf h_2^\top B\mathbf h_2$
   - with $B = K^{-\top}K^{-1}$ (symmetric, 6 unknowns up to scale)
3. **Each image gives 2 linear equations** in b. Stack N images into **Vb = 0** and solve with **SVD** (the singular vector of the smallest singular value). **≥ 3 views are needed** (6 unknowns, 2 per view).
4. **Recover K from B** with a **Cholesky decomposition**.
5. Then get R and t per image, and refine everything (plus distortion) with LM on the reprojection error.

**Why ≥ 3 views?** B has 6 entries up to scale, so 5 unknowns. Each view gives 2 constraints, so 3 views give 6 ≥ 5. With skew assumed 0, **2 views** can suffice. In practice, use 10–20 views for robustness.

## 3.6 Reprojection error — the key idea ★

1. Take a known 3D corner **X**.
2. Project it with the **current** K, R, t to get the **predicted pixel x̂**.
3. Compare it with the **detected pixel x**.

$$E = \sum \lVert \mathbf x - \hat{\mathbf x}\rVert^2\quad\text{minimised over K, distortion, and every view's R, t}$$

- **Rule of thumb: an RMS error well under ~0.5 px is good.**
- **Much larger** means blurry photos, wrongly detected corners, or a bad distortion model.
- The **same idea** reappears in triangulation, SfM bundle adjustment and SLAM: *"optimise until the reprojections land on the observations."*

**Worked example:** observed (100, 100) and (200, 200); predicted (100.3, 99.6) and (200, 200.5).
- The errors are 0.5 and 0.5.
- **RMSE = √((0.25 + 0.25)/2) = 0.5 px**, which is on the borderline.

## 3.7 Lens distortion (on the calibration slide; syllabus item) ★

Real lenses bend straight lines. The pinhole model assumes they don't, so we model the bending on **normalised coordinates** (x, y), with $r^2 = x^2 + y^2$:

$$x_d = x(1 + k_1r^2 + k_2r^4 + k_3r^6) + 2p_1xy + p_2(r^2 + 2x^2)$$
$$y_d = y(1 + k_1r^2 + k_2r^4 + k_3r^6) + p_1(r^2 + 2y^2) + 2p_2xy$$

| Type | Coefficients | Cause | Looks like |
|---|---|---|---|
| **Radial** | k₁, k₂, k₃ | lens curvature; worse **far from the centre** (grows with r) | **Barrel** (k₁ < 0: lines bow **outward**, typical of wide-angle/fisheye) or **pincushion** (k₁ > 0: lines bow **inward**, telephoto) |
| **Tangential** | p₁, p₂ | lens **not parallel** to the sensor (assembly misalignment) | slight skewed or tilted distortion |

- **Worked example:** x = 0.2, y = 0.1, k₁ = −0.2 → r² = 0.05, factor = 1 − 0.01 = 0.99 → **x_d = 0.198, y_d = 0.099**. The point moves **toward the centre**, which is barrel distortion.
- **Undistort before geometry:** `cv2.undistort(img, K, dist)`. Homography, stereo and epipolar maths all assume a **pinhole (straight-line) model**.
- Distortion coefficients are **intrinsic** properties: they belong to the lens and don't change when the camera moves.
- **Fisheye lenses** (~180° FOV) need a different model (`cv2.fisheye`).

---

# PART 4 — Homography Revisited ★

## 4.1 When every point lies on a plane

Start from the full model, then set **Z = 0** for every point on the plane:

$$s\begin{bmatrix}u\\v\\1\end{bmatrix} = K[\mathbf r_1\ \mathbf r_2\ \mathbf r_3\ \mathbf t]\begin{bmatrix}X\\Y\\0\\1\end{bmatrix} = K(\mathbf r_1X + \mathbf r_2Y + \mathbf r_3\cdot 0 + \mathbf t) = \underbrace{K[\mathbf r_1\ \mathbf r_2\ \mathbf t]}_{H\ (3\times3)}\begin{bmatrix}X\\Y\\1\end{bmatrix}$$

**r₃·0 = 0, so the third rotation column contributes nothing, and the 3×4 camera matrix collapses to a 3×3 homography.**

- **A homography is a 3×3 matrix H that maps points between two views of the same plane:** $\mathbf x' \simeq H\mathbf x$.
- It also relates two images taken by a **purely rotating camera** (no translation), whatever the scene. That's how **panoramas** are stitched: $H = K R K^{-1}$.

## 4.2 Why exactly 4 points? ★★

$$H=\begin{bmatrix}h_{11}&h_{12}&h_{13}\\h_{21}&h_{22}&h_{23}\\h_{31}&h_{32}&h_{33}\end{bmatrix},\qquad x'=\frac{h_{11}x+h_{12}y+h_{13}}{h_{31}x+h_{32}y+h_{33}},\quad y'=\frac{h_{21}x+h_{22}y+h_{23}}{h_{31}x+h_{32}y+h_{33}}$$

| Count | Value |
|---|---|
| Unknowns in H | 9 entries − 1 (scale) = **8 DOF** |
| Equations per point pair | **2** (one for u, one for v) |
| Point pairs needed | 8 ÷ 2 = **4** |

- **Condition: no three of the four points can lie on one line** (that would make the system degenerate).
- **Why "−1 for scale"?** x' ≃ Hx is equal **up to scale**, so H and 2H give the same mapping. We usually fix h₃₃ = 1.
- **In practice:** use **many matches + RANSAC** to ignore bad ones: `cv2.findHomography(src, dst, cv2.RANSAC)`. Warp with `cv2.warpPerspective(img, H, (w, h))`.

**DLT (Direct Linear Transform):** each correspondence gives 2 rows of **A h = 0**. Stack ≥ 4 pairs and take **h = the last right-singular vector of A** (SVD). See the code in Part 12.

**RANSAC, briefly:**
1. Repeatedly pick **4 random matches**, fit H, and count the **inliers** (matches with reprojection error < a threshold).
2. Keep the H with the most inliers, then refit on all its inliers.

It is robust to wrong matches (outliers).

**Applying H, numerical:** H = [[1, 0, 0], [0, 1, 0], [0.001, 0, 1]] and the point (100, 50).
- Homogeneous result: (100, 50, 0.1 + 1 = 1.1).
- **(90.9, 45.45)**. The bottom row creates the **perspective effect**: points get scaled depending on their position.

**The transformation hierarchy** (with DOF and the minimum number of points):

| Transform | DOF | Preserves | Min. points |
|---|---|---|---|
| Translation | 2 | everything except position | 1 |
| Euclidean (rigid) | 3 | lengths, angles | 2 |
| Similarity | 4 | angles, length *ratios* | 2 |
| Affine | 6 | **parallel lines**, area ratios | 3 |
| **Projective (homography)** | **8** | **straight lines** only (parallel lines can meet) | **4** |

## 4.3 Applications (all assume the content lies on one plane)

1. **Document scanning:** tap the 4 corners → compute H → resample every pixel to get a flat, fronto-parallel page.
2. **Inverse perspective mapping (IPM) / bird's-eye view:** converging lane lines become **parallel and measurable**. Parking lines are parallel and evenly spaced in reality, and one H undoes the perspective because **the ground is flat**.
3. **Surround-view BEV:** four fisheye cameras are warped to the ground plane and stitched around the car (parking assist).
4. Panorama stitching (rotating camera), AR overlay on a planar marker, sports-field overlays, and **camera calibration itself** (one H per checkerboard view).

## 4.4 Limitation: the world is not flat ★

- **H is exact only for points ON the plane** (or for a purely rotating camera).
- **Anything with height gets smeared.** A box on the ground warped with the ground-plane H shows its true **footprint** correctly, but its **height is smeared along the ground** in the BEV.
- That's why BEV images show cars and poles "stretched" away from the camera.

---

# PART 5 — Stereo Vision & Depth from Disparity ★★

## 5.1 Why a single camera fails

- **One pixel = one ray. Depth along the ray is lost.** Every point on the ray lands on the same pixel.
- **Small-and-near looks exactly like big-and-far.** No single-image geometry can undo this.
- **"How do monocular depth networks still predict depth?"** **Learned priors**: typical object sizes, perspective cues, texture gradients, scene layout. They estimate; they don't measure.

## 5.2 Two eyes: the original stereo system

- **The finger experiment** (teacher did it in class): hold up a finger and close one eye, then the other.
  - **The finger jumps a lot; a far tree barely moves.**
  - That jump is **disparity**. **Near objects shift more; far objects shift less.**
- **Disparity = the horizontal shift of the same point between the left and right images**, after **rectification**, which puts the match on the **same row**:

$$d = u_L - u_R\qquad\text{e.g. } 400 - 330 = 70\text{ px}$$

## 5.3 The depth equation ★★

```
          P (point at depth Z)
         / \
        /   \           similar triangles:
       /     \          P–O_L–O_R  ∼  P–x_L–x_R
  x_L •───────• x_R     ───────────────────────
     /  f      \        B / Z = (B − d) / (Z − f)
  O_L ─── B ─── O_R     ⇒  Z = f·B / d
```

$$\boxed{Z = \frac{f\cdot B}{d}}\qquad\text{f in px, B in m, d in px} \Rightarrow Z\text{ in m}$$

- **Depth is inversely proportional to disparity.** f and B are fixed by your hardware: *"you bought the camera (f) and you mounted it (B)"*.
- **d → 0 means Z → ∞** (very far, or the sky). **d = 0 is invalid**, so guard against dividing by zero.

**Worked examples (slides):**
- f = 720 px, B = 0.54 m (KITTI):
  - **d = 70 px → Z = 5.6 m**
  - **d = 8 px → Z = 48.6 m**
  - **d = 24 px → Z = 720 × 0.54/24 = 16.2 m**
- **Off by 1 px** (d = 23) → **16.9 m**. A 1-px error costs 0.7 m at 16 m.

**Depth error grows with Z²** ★:

$$\Delta Z \approx \frac{Z^2}{f\cdot B}\,\Delta d$$

- Example: f = 800, B = 0.10 m, so fB = 80. A **1-px error** costs **4/80 = 5 cm at 2 m**, but **16/80 = 20 cm at 4 m**.
- **Doubling the distance quadruples the error.** Fix: a **wider baseline or a longer focal length**.

**Full 3D point from a rectified pair** (slide example):
- f = 800 px, B = 0.10 m, c_x = 640, c_y = 360, u_L = 520, u_R = 480, v = 300.
- **d = 40 px → Z = 800 × 0.10/40 = 2.00 m**
- **X = (u_L − c_x)·Z/f = (520 − 640)·2/800 = −0.30 m**
- **Y = (v − c_y)·Z/f = (300 − 360)·2/800 = −0.15 m**

## 5.4 Baseline: the trade-off ★ (the teacher asked this)

- **Baseline B** = the distance between the two optical centres.
  - **Human eyes ≈ 6.5 cm.**
  - **A car stereo rig: tens of cm** (**KITTI ≈ 54 cm**).

| Wider baseline | Narrower baseline |
|---|---|
| ✅ **Better depth accuracy far away** (more disparity per metre) | ✅ **Easier matching** (the views look almost the same) |
| ❌ **Bigger blind zone** near the cameras (things close by are seen by only one camera) | ✅ Smaller blind zone, good for **near** objects |
| ❌ **Harder matching** (the views look more different; more occlusion) | ❌ Poor depth far away (tiny disparities) |

- **Teacher:** *"if the cameras are almost at the same place, the images are nearly identical and matching is easy, but you can't estimate depth. You want it a little wider: matching gets harder, but depth gets better."*
- **Your eyes' blind zone:** an object closer than about 6 cm is blurry and can't be fused.
- Tilting (toeing in) the cameras can shift the overlap region.

## 5.5 Depth map and where stereo struggles

- **Depth map:** compute d for **every pixel**, apply Z = fB/d, and colour by distance. Near ≈ 3 m, far ≈ 80 m; **the sky has no valid depth**.
- **Failure cases (★ multi-select):**

| Problem | Why |
|---|---|
| **Textureless areas** (white walls, sky, road) | Every patch looks the same, so any match is equally good |
| **Reflections & glass** | The mirror image seems to be *behind* the glass, giving a wrong, too-far depth |
| **Occlusions** | Some points are seen by **only one camera**, so they have **no match at all** |
| Repetitive patterns (fences, tiles) | Many look-alike matches |

  *"Stereo is only as good as the matching."*
- **Left–right consistency check (teacher):** match left→right and right→left along the epipolar line. **If the two don't point to the same match, discard it** (likely occluded or wrong).

## 5.6 Stereo as an optimisation problem (Session 12 transcript ★)

**The old way:** match a keypoint, shoot two rays, intersect them. Bad matches put the intersection in the wrong place.

**Reformulate as optimisation:**
1. For a pixel q in image 1, **try a set of quantised depths** along its ray (e.g., 1.0, 1.1, …, 10 m). This is called **plane sweep / cost volume**.
2. For each candidate depth, **reproject into image 2** and compare colours.
   - **Photometric (data) loss:** does the colour at x₂ match the colour at x₁?
   - **Teacher's example:** a wrong depth lands on a "plant" pixel when it should be on a "toy" pixel, so the loss is high.
3. Add a **smoothness loss**:
   - **Neighbouring pixels have similar depth**, so the **depth gradient can't be too big** (pixel q at 1.1 m means q+1 is ~1.2 m, not 10 m).
   - **Edge-aware:** at object edges depth *does* jump, so first **segment the image into regions** (similarity-based merging, like Selective Search) and apply smoothness **only within a region**. Each region is assumed to have roughly linear disparity.
4. Quantised disparities form a **"vocabulary"**. Optimisation picks the best label per pixel:
   - **unary term:** does the colour data agree?
   - **pairwise term:** are neighbours in the same region similar?
5. Because it is now an **optimisation problem**, a **neural network** can learn it (learned stereo, self-supervised depth).

- **Benchmark:** the **Middlebury stereo** datasets and evaluation.
- **OpenCV:** **StereoSGBM** (Semi-Global Block Matching) gives dense disparity; `cv2.reprojectImageTo3D(disp, Q)` gives (X, Y, Z) per pixel.


---

# PART 6 — Epipolar Geometry & Triangulation ★

## 6.1 The matching problem

- To triangulate, you need **correspondences**: the same physical point in both images.
- **Without geometry, every pixel must be compared with the whole other image.**
  - That is **slow**: for 1280×720 that is **921,600 candidates per pixel**.
  - It also gives **many look-alike mistakes**.
- **Epipolar geometry tells you where a corresponding point can appear: on one line.**

## 6.2 The epipolar plane, line and epipole ★★

```
                  P (somewhere along the ray through x: P1? P2? P3?)
                 /\
                /  \           epipolar plane = plane through O, O′, P
     x •       /    \      • x′ lies on l′ (epipolar line)
   left img   /      \     right img
             O ─────── O′
               baseline
```

| Term | Meaning |
|---|---|
| **Baseline** | the line joining the two camera centres O and O′ |
| **Epipolar plane** | **three points define a plane**: O, O′ and the 3D point P. Both x and x′ lie in it |
| **Epipolar line l′** | where that plane **slices** the right image. **The match of x must lie on l′** |
| **Epipole e′** | **where the other camera's centre appears in this image.** **Every epipolar line passes through it** |

**Why a line?** We don't know how far along its ray P is (P₁, P₂ or P₃…). Each candidate projects somewhere in the right image, and **all those projections fall on one line, l′**. **So we search along l′, not the whole image.**

**Where is the epipole?**

| Setup | Epipole | Epipolar lines |
|---|---|---|
| General | some point, often outside the image | all meet at e′ |
| **Rectified stereo** | **at infinity** | **parallel, horizontal rows** |
| **Driving forward** | **at the image centre**, the **"focus of expansion"** | radiate outward from the centre |

**Search reduction (★ numerical):**
- **2D search: 1280 × 720 = 921,600 candidates.**
- **1D search: ≤ 1,280 candidates** (one row).
- **That is 720× fewer**, so it is faster and makes far fewer mistakes.

## 6.3 Fundamental matrix F vs essential matrix E ★★

**The epipolar constraint (for corresponding points in homogeneous coordinates):**

$$\mathbf x_2^\top F\,\mathbf x_1 = 0,\qquad \mathbf l_2 = F\mathbf x_1\ \text{(the epipolar line in image 2)}$$

$$\hat{\mathbf x}_2^\top E\,\hat{\mathbf x}_1 = 0\ \ (\hat{\mathbf x} = K^{-1}\mathbf x\text{, normalised}),\qquad E = [\mathbf t]_\times R,\qquad E = K_2^\top F K_1$$

$$[\mathbf t]_\times = \begin{bmatrix}0&-t_z&t_y\\t_z&0&-t_x\\-t_y&t_x&0\end{bmatrix}\ (\text{skew-symmetric: } [\mathbf t]_\times\mathbf a = \mathbf t\times\mathbf a)$$

| | **Fundamental F** | **Essential E** |
|---|---|---|
| Works on | **raw pixels** | **normalised coordinates K⁻¹x** |
| Needs K? | **No** (works with **uncalibrated** cameras) | **Yes** (calibrated) |
| DOF | **7** (9 − 1 scale − 1 because det F = 0, rank 2) | **5** (R: 3 + t: 3 − 1 scale) |
| Estimate from | **≥ 8 matches (the 8-point algorithm)**; 7 is the minimum | ≥ 5 matches (the 5-point algorithm) |
| Gives you | **epipolar lines** (where to search) | **relative pose: R and t (t only up to scale)** |
| OpenCV | `cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC)` | `cv2.findEssentialMat` + `cv2.recoverPose` |

★ *"The fundamental matrix only tells you how pixels correspond on a screen. The essential matrix tells you how the physical camera moved in the 3D world."* So: **calibrate first → E → recover how the second camera moved.**

**Extra facts for multi-select questions:**
- Decomposing E (SVD) gives **4 possible (R, t)** solutions. The right one is picked by the **cheirality check**: triangulated points must be **in front of both cameras**.
- **t from E has unknown scale**, so you get the direction of translation but not how far.

**Summary: geometry makes matching tractable:**

```
Two-view geometry (F or E) → epipolar line for each pixel → 1D search along the line → correspondences x ↔ x′
```

- **Geometry says WHERE to look** (the line). **Appearance says WHAT matches** (compare patches or learned features along it).
- **Rectified special case:** the lines become rows, so the match is just a horizontal shift, the **disparity**.
- *"Epipolar geometry → correspondence. Correspondence → depth."*

## 6.4 Triangulation: recovering the 3D point ★

**Idea:**
- **Each pixel defines a ray** holding every 3D point that could have made it.
- Two cameras give two rays, and **where they cross is the 3D point P**.
- **You need:** both cameras' **K, R, t** (calibration) **and** the **match x ↔ x′** (from the epipolar search).
- Back-projection: $\hat{\mathbf x} = K^{-1}[u, v, 1]^\top$ is the ray direction, and the ray is $\mathbf r = \mathbf C + \lambda\hat{\mathbf x}$.

**The ideal case:** $O + \lambda\mathbf d = O' + \mu\mathbf d'$.
- Solve for λ and μ, then P = O + λd.
- That is **3 equations (x, y, z) in 2 unknowns, so normally there is no exact solution.**

**The real case — the rays don't meet:**
- Seen from the top they seem to cross; from the side they **pass each other** with a gap.
- **Causes:**
  - **pixel noise** (corners found only to a fraction of a pixel)
  - **calibration error** (K, R, t are estimates)
  - **quantisation** (pixels are discrete)

**Solutions:**
1. **Midpoint method:** find the **shortest segment Q₁Q₂ joining the rays** (perpendicular to both). Its **midpoint P̂** is the estimate.
2. **Better and standard:** choose the P̂ whose reprojections land **closest to the observed pixels**, i.e., **minimise the reprojection error**, as in calibration.
3. **Linear triangulation (DLT):** each view gives 2 equations, where p₁ᵀ, p₂ᵀ, p₃ᵀ are the rows of the camera matrix P and X = (X, Y, Z, 1):

$$u\,(\mathbf p_3^\top\mathbf X) - \mathbf p_1^\top\mathbf X = 0,\qquad v\,(\mathbf p_3^\top\mathbf X) - \mathbf p_2^\top\mathbf X = 0$$

   Stack both views to get **A X = 0 (A is 4×4)**, so **X = the last singular vector of A** (SVD). Then divide by the 4th coordinate. `cv2.triangulatePoints(P1, P2, x1, x2)`.
4. **Rectified-stereo shortcut (no SVD):** $Z = fB/d$, $X = (u - c_x)Z/f$, $Y = (v - c_y)Z/f$.

**Check (tested in code):** X = (0.3, −0.2, 2.0), f = 800, c = (320, 240), and the second camera 0.1 m to the right.
- The point projects to x₁ = (440, 160) and x₂ = (400, 160).
- DLT recovers **(0.3, −0.2, 2.0)**, and Z = 800·0.1/40 = 2 m ✓.

## 6.5 The full stereo 3D-reconstruction pipeline (slides, "putting it together")

**In:** two photos from a **calibrated stereo rig** (known K, baseline B, and R, t between the cameras). **Out:** a **3D point cloud in metres**, cleaned and turned into a **surface mesh**.

1. **Find the same point in both images:**
   - **Detect** keypoints (SIFT, ORB).
   - **Describe** each patch as a vector (**SIFT: 128 numbers**).
   - **Match** to the nearest vector in the other image, but **keep it only if it clearly beats the 2nd-best: the Lowe ratio < 0.75**.
   - `sift = cv2.SIFT_create(); kp1, d1 = sift.detectAndCompute(L, None); m = cv2.BFMatcher().knnMatch(d1, d2, k=2)`
   - You get hundreds of matches, but look-alike textures still give a few wrong ones.
2. **Keep only matches that fit the epipolar geometry:**
   - $\mathbf x'^\top F\mathbf x = 0$, so the match must lie on its line **l′ = Fx**.
   - **RANSAC:** fit F to **8 random matches**, count the matches within ~1 px of their line, repeat, and keep the best. Drop the matches far from their line.
   - **Rectify:** every epipolar line becomes a pixel row, so dense matching becomes a 1D search.
   - **Rig vs moving camera:**
     - **Rig:** R and t come from stereo calibration.
     - **Moving camera:** estimate **E** and recover R and t with `cv2.recoverPose` (**t only up to scale**).
3. **Triangulate:** d = u_L − u_R, Z = fB/d, X and Y as above. If the pair isn't rectified, intersect the rays with DLT.
4. **From depth to a 3D object:**
   - **Go dense:** run **StereoSGBM** for a disparity at nearly every pixel, then `cv2.reprojectImageTo3D(disp, Q)`.
   - **Clean:** drop invalid disparities, **remove statistical outliers**, and **cut the floor with a RANSAC plane fit**.
   - **Mesh:** e.g., **Poisson reconstruction** in Open3D.
   - **All sides:** one stereo pair sees one side. Capture several viewpoints and **align the clouds (ICP)** or **solve all poses with SfM**, then fuse them.

---

# PART 7 — Optical Flow & Tracking ★

## 7.1 What optical flow is

- **Optical flow = the apparent motion of pixels between consecutive frames**: for each pixel, a displacement **(u, v) = (dx, dy)**.
  - Teacher: *"a relative motion between the two camera images."* Where is this keypoint in the next image?
- **Camera-motion patterns (teacher's examples):**

| Motion | Flow pattern |
|---|---|
| Camera pans or turns | All pixels move the opposite way |
| **Moving forward** (driving) | Pixels **expand outward from a centre**, the **focus of expansion (FOE)**, which is the epipole. The road seems to flow backward |
| Moving down (skydiving) | Flow points outward/upward from below |
| Sitting in a train | The background streams past |

- **Depth matters:** when the camera moves, **near pixels move a lot and far pixels move little** (the same idea as disparity and the finger experiment). Flow is related to **depth and camera motion**.
- Two stereo cameras also give a "flow" between the left and right images (that is the disparity).

## 7.2 The maths [syllabus add-on: standard material]

**Brightness constancy:** a point keeps its brightness as it moves: $I(x, y, t) = I(x+u, y+v, t+1)$.

Take a first-order Taylor expansion and you get the **optical-flow constraint equation**:

$$\boxed{I_x\,u + I_y\,v + I_t = 0}$$

where I_x and I_y are the spatial gradients and I_t is the temporal difference between frames.

**The aperture problem ★:**
- That is **1 equation with 2 unknowns per pixel**.
- You can only recover the motion **perpendicular to an edge**. Motion *along* an edge is invisible (think of a barber-pole illusion, or a line seen through a small hole).
- You need extra assumptions:

| Method | Assumption | Type |
|---|---|---|
| **Lucas–Kanade (LK)** | **Flow is constant in a small window** (e.g., 5×5), giving 25 equations for 2 unknowns, solved by least squares | **Sparse/local** (good at corners) |
| **Horn–Schunck** | **Flow is smooth over the whole image** (a global smoothness term) | **Dense/global** |
| Farnebäck | polynomial expansion | dense (OpenCV) |
| **RAFT** (deep) | learned correlation volume + iterative updates | dense, state of the art |

**Lucas–Kanade solve:**

$$A=\begin{bmatrix}I_{x1}&I_{y1}\\ \vdots&\vdots\end{bmatrix},\ \mathbf b=-\begin{bmatrix}I_{t1}\\ \vdots\end{bmatrix}\ \Rightarrow\ \begin{bmatrix}u\\v\end{bmatrix} = (A^\top A)^{-1}A^\top\mathbf b$$

- It **works only if AᵀA is invertible and well-conditioned**: both eigenvalues must be large. That is true at **corners** (the **Shi–Tomasi / Harris** idea).
- At an **edge** one eigenvalue ≈ 0 (the aperture problem). In a **flat region** both are ≈ 0.
- **Large motions** break the Taylor approximation. Fix: a **coarse-to-fine image pyramid**.

**Numerical:** pixel 1 has (I_x, I_y, I_t) = (2, 1, −4) and pixel 2 has (1, 3, −7).
- The equations are 2u + v = 4 and u + 3v = 7.
- **u = 1, v = 2.** (Tested in code. With only pixel 1 you'd have the aperture problem: infinitely many solutions.)

## 7.3 Using flow: dynamic objects and segmentation (teacher's industry tip ★)

- **To tell static scene from moving objects:**
  - When the camera moves, the **static background has one consistent flow pattern** (following the epipolar geometry and F).
  - **A moving object breaks that pattern**: it shows up as an **anomaly** with its own trajectory.
- **In ADAS work, teams use optical flow to remove dynamic objects first.** Only the static parts are then used for calibration, reconstruction and odometry.
- **Segment-then-flow:** a person walking moves differently from the background. Segmenting regions first and computing flow per region gives cleaner flow (the tennis-player example).

## 7.4 Tracking [syllabus add-on, plus links to Modules 4 and 5]

- **KLT tracker (Kanade–Lucas–Tomasi):**
  1. Detect good corners (**Shi–Tomasi**: both eigenvalues of AᵀA large).
  2. **Track them frame to frame with pyramidal LK**: `cv2.calcOpticalFlowPyrLK`.
  3. Drop tracks with large error and re-detect when too few remain.
- **Tracking-by-detection** (Module 4's ByteTrack/SORT):
  - Detect objects in each frame.
  - A **Kalman filter** predicts each track's next position (**predict**), then corrects it with the matched detection (**update**).
  - Match detections to tracks by IoU (**Hungarian** assignment).
  - It keeps **consistent IDs** and bridges missed detections.
  - **ID switches** are the typical failure.
- **Feature tracking is the front-end of visual odometry and SLAM** (Part 9).

---

# PART 8 — Structure from Motion (SfM) ★

## 8.1 The idea

- **Recover the 3D structure (points) AND the camera poses from many images** taken from different viewpoints, often an **unordered** photo collection. *"Create structure from the movements."*
- **Teacher's cube example:**
  - Take a known 1 m cube and fix the origin at one corner, so all the corner coordinates are known.
  - Photograph it from several viewpoints with **unknown K, R, t**.
  - Optimise the depths and poses until **projecting into one image and reprojecting into the others is consistent**.
  - *"Very similar to camera calibration with the checkerboard."*
- For **m images and n points**:

$$\mathbf x_{ij} \simeq P_i\,\mathbf X_j,\qquad P_i = K_i[R_i|\mathbf t_i];\ \text{solve for all } P_i\ (i=1..m) \text{ and } \mathbf X_j\ (j=1..n)$$

## 8.2 The standard SfM pipeline (incremental, as in COLMAP) [the steps are standard material]

1. **Features + matching** (SIFT + ratio test) across image pairs.
2. **Geometric verification** with **RANSAC on F/E** (or H), which removes wrong matches.
3. **Initialise** from a good image pair: **E → R, t → triangulate** the first points.
4. **Register a new image** by estimating its pose from known 3D points ↔ its 2D features with **PnP** (Perspective-n-Point).
5. **Triangulate** the new points.
6. **Bundle adjustment** (below), then repeat from step 4.
7. Output: a **sparse point cloud + camera poses**. Dense MVS reconstruction can follow.

## 8.3 Bundle adjustment ★

**Jointly refine ALL camera poses and ALL 3D points** to minimise the **total reprojection error**:

$$\min_{\{P_i\},\{\mathbf X_j\}}\ \sum_{i}\sum_{j} \big\lVert \mathbf x_{ij} - \pi(P_i, \mathbf X_j)\big\rVert^2$$

- **"Bundle"** = the bundles of light rays from each 3D point to each camera, adjusted together.
- Teacher: instead of optimising image 1 against image 2, then 1 against 3, and so on, **optimise all at the same time**. That is bundle adjustment.
- It is solved with **Levenberg–Marquardt**, exploiting sparsity.
- **Size:** with m cameras (6 pose parameters each) and n points (3 each), there are **6m + 3n unknowns**. 10 cameras and 1,000 points → **3,060 unknowns**.
- It gives high quality, but is **expensive** and done **offline** in SfM. VO and SLAM use a cheaper **windowed** version (Part 9).

## 8.4 Ambiguities: is SfM uniquely solvable? ★★ (the teacher asked the class)

**No.** From images alone there are many solutions:

$$\mathbf x = P\mathbf X = (PQ^{-1})(Q\mathbf X)\quad\text{for any invertible } 4\times4\ Q$$

Transform the scene by Q and the cameras by Q⁻¹, and **the image observations don't change.**

| Reconstruction "up to…" | Constraint added on Q | DOF left in Q | What's still wrong |
|---|---|---|---|
| **Projective** | none | **15** (4×4 − 1) | shapes can look very weird in 3D (only straight lines are kept) |
| **Affine** | **parallel lines stay parallel** | **12** | angles and lengths are distorted |
| **Similarity (metric)** | **orthogonality / calibrated cameras** (right angles kept) | **7** = 3 rotation + 3 translation + **1 scale** | **the absolute scale is unknown** |

**Scale ambiguity ★:**
- $\mathbf x = K[R\,|\,\mathbf t]\mathbf X$ is unchanged if you **scale the scene by k and scale the translation by k too**.
- A **bigger scene with the camera further away looks identical** to a smaller scene with the camera closer (the toy-car problem again).
- ***"Without a reference measurement, it is impossible to recover the absolute scale of the scene."***
- **Fix:** know **one real dimension**. Teacher: *"place my phone in the scene; I know its size, so I get the scale k."* Other options: a stereo baseline, an IMU, GPS, or the wheel distance.
- Rotation is **not** affected by scale; **translation is**.

---

# PART 9 — Visual Odometry & SLAM (conceptual) ★

## 9.1 Visual odometry (VO)

- **Odometry** = estimating how far and in which direction you moved. **Visual odometry = estimating the camera's (robot's) motion from images.**
- An agent moves and captures images with a **rigidly attached camera** at discrete times k:

| Setup | What you get at each time k |
|---|---|
| **Monocular** | one image I_k. **Scale is unknown** |
| **Stereo** | a left/right pair. The origin is usually the left camera. Depth comes from the stereo, so **scale is known** |
| **RGB-D** | image + a **depth channel** (e.g., iPhone Pro LiDAR, structured light) |

- Poses at times k−1 and k are related by a **rigid-body transformation $T_k = \begin{bmatrix}R_k & \mathbf t_k\\ 0 & 1\end{bmatrix}$**.
- **Chain them:** $C_n = C_{n-1}\,T_n$. The starting pose C₀ can be anything.
- **The main task of VO: compute the relative transformations T_k from the images, then concatenate them to recover the full trajectory**, pose after pose (incrementally).

**Pose-chaining numerical (tested):**
1. Move 1 m forward (+x): position (1, 0, 0).
2. Turn 90° left and move 1 m: position (2, 0, 0), now facing +y.
3. Move 1 m "forward" again, which is now world +y: **C = (2, 1, 0)**.

### VO assumptions, and WHY each is needed ★ (the teacher asked the class)

| Assumption | Why |
|---|---|
| **Sufficient illumination** | Keypoints must be **detectable and repeatable**. Dark images give no features |
| **Static scene dominates moving objects** | Moving objects violate the static-world maths. *"If you walk alongside me as I move, the scene looks unchanged and I'm confused whether I moved."* Use **optical flow** to find and remove dynamic objects |
| **Enough texture** | To **uniquely identify and match keypoints**. Textureless walls give ambiguous matches |
| **Sufficient scene overlap** between frames | You need **common features to match and triangulate**. With no overlap there is nothing to relate |

### Other odometry sources (and their failure modes)

| Source | How | Fails when |
|---|---|---|
| **Wheel odometry** | wheel circumference × revolutions (motor RPM × time) | **wheel slip** (on a slippery floor the wheel turns but the robot doesn't move) |
| **GPS** | satellite global coordinates over time | **tunnels, indoors, urban canyons** (no signal); metres of error |
| **IMU** (inertial measurement unit) | accelerometer + gyroscope integrate the motion | **drifts quickly** (integration error) |
| **LiDAR / laser odometry** | match 3D scans | featureless corridors, rain/fog, cost |
| **Visual** | camera images | low light, no texture, motion blur, dynamic scenes |

**VO's advantage over wheel odometry: it is not affected by wheel slip.** In practice you **fuse all of them** (sensor fusion), so when one fails another works.

### The VO pipeline

```
Image sequence → feature detection → feature matching/tracking → motion estimation (T_k) → local optimisation (windowed BA) → trajectory
```

**Three ways to estimate T_k (★ know which data each uses):**

| Method | Input | How | Scale |
|---|---|---|---|
| **2D → 2D** | features in I_{k−1} and I_k (monocular) | compute the **essential matrix E** and decompose it (SVD) into R_k, t_k | **t only up to scale**; you must compute the relative scale and rescale |
| **3D → 3D** | **stereo** pairs at k−1 and k | **triangulate** the features in each stereo pair to get 3D points, then find the T_k that **aligns the two 3D point sets** | known |
| **3D → 2D** | 3D points (from previous frames) and 2D features in the new frame | **PnP (Perspective-n-Point)**: the camera pose from 3D↔2D matches, minimising the reprojection error. **P3P needs ≥ 3 points** (+1 to disambiguate) | known if the 3D is metric |

**Local optimisation (windowed bundle adjustment):**
- SfM does BA on **all images offline**. VO runs **in real time** as frames arrive, so it does BA over **only the last m frames (keyframes)**, optionally.
- It minimises the **reprojection (or photometric) error** over the poses and 3D points in that window.

### VO vs SfM

| | **SfM** | **VO** |
|---|---|---|
| Input | **unordered** image set | **sequential** video, as frames arrive |
| Goal | **structure + camera poses** | mainly the **camera motion (trajectory)**, in **real time** |
| Optimisation | **full bundle adjustment, offline** (cost grows with the number of images) | fast; **windowed BA optional** |
| Priority | **reconstruction quality** | **speed** ("quick answers while the robot moves") |
| Relation | VO is a **particular case of SfM**: sequential, real-time | — |

## 9.2 The drift problem ★

- **Each relative estimate T_k has a small error, and chaining them accumulates the error: drift.**
- **Teacher's example:**
  1. You command "move 5 m ahead", but the robot goes **4 m** and slightly to the side (actuation error).
  2. You *believe* you're at the commanded spot, so the next move ("turn right, go 3 m") starts from a wrong place.
  3. **The errors compound**, and you end up somewhere else entirely.
- **VO alone can't fix this. It needs landmarks, a map and loop closure, which is what SLAM adds.**

## 9.3 SLAM: Simultaneous Localisation And Mapping ★

**Build a map of the environment AND localise yourself in it, at the same time.** It is a chicken-and-egg problem: you need the map to localise, and the pose to build the map.

**The landmark loop (teacher's stereo SLAM walkthrough):**
1. **Create 3D landmarks:**
   - From the stereo disparity, get the depth Z of distinct features (e.g., "the watch").
   - Then $\mathbf X = Z\cdot K^{-1}[u, v, 1]^\top$: the unit ray scaled by depth gives the 3D landmark.
2. **The robot moves** from t to t+1 with an estimated rotation R and translation t.
3. **Predict:** transform the landmarks with the estimated motion and **project them into the new view**. *"Where should the watch appear now?"*
4. **Observe:** detect and match the same features in the new stereo pair, and compute their current depth.
5. **Correct:** compare the predicted and observed pixel locations, which is the **reprojection error** (or a 3D error if you also measure depth). **Optimise the pose (and landmarks) to minimise it.**
6. **Update the map** with the new and refined landmarks, and repeat.

**Loop closure ★ [syllabus add-on]:**
- When the robot **returns to a place it has seen before** and **recognises** it (e.g., bag-of-words place recognition), the mismatch between "where I think I am" and "where I recognise I am" exposes the **accumulated drift**.
- A **global optimisation** (**pose-graph optimisation / global BA**) spreads the correction over the whole trajectory. **This is what removes drift.**

**SLAM families (concept level):**
- **Filter-based:** EKF-SLAM, a Kalman filter over the pose + landmarks.
- **Graph/optimisation-based:** pose graph + BA (the modern standard). Examples: **ORB-SLAM** (feature-based), LSD/DSO (direct, photometric).
- **Sensors:** monocular / stereo / RGB-D / visual-inertial (camera + IMU) / LiDAR SLAM.

| | **VO** | **SLAM** | **SfM** |
|---|---|---|---|
| Real time? | yes | yes | no (offline) |
| Keeps a map? | at most a local window | **yes, a global map** | yes (the reconstruction) |
| Loop closure? | **no, so it drifts** | **yes, which corrects drift** | global BA over all images |
| Main output | trajectory | trajectory + map | 3D model + poses |

**One-liners:**
- **VO** = "where did I move?"
- **SLAM** = "where am I, *and* what does the world look like, consistently?"
- **SfM** = "rebuild the scene from a pile of photos".


---

# PART 10 — Formula Sheet

| # | Topic | Formula / fact |
|---|---|---|
| 1 | Perspective | $x = fX/Z,\ y = fY/Z$; image size = f·size/Z (∝ 1/Z) |
| 2 | Pixels | $u = f_xX/Z + c_x,\ v = f_yY/Z + c_y$ |
| 3 | Distance from known height | $Z = fH/h$ |
| 4 | Focal length mm → px | $f_{px} = f_{mm}\cdot W_{px}/W_{sensor,mm}$ |
| 5 | Field of view | $2\arctan(W/2f_x)$ |
| 6 | K | [[f_x, γ, c_x], [0, f_y, c_y], [0, 0, 1]]; 4 unknowns (5 with skew) |
| 7 | Extrinsics | $\mathbf X_c = R\mathbf X_w + \mathbf t$; 6 DOF (3 + 3) |
| 8 | Camera matrix | $\mathbf x \simeq K[R\,|\,\mathbf t]\mathbf X$; P is 3×4 with **11 DOF** |
| 9 | Reprojection error | $E = \sum\lVert\mathbf x - \hat{\mathbf x}\rVert^2$; good RMS < 0.5 px |
| 10 | Distortion | radial $(1 + k_1r^2 + k_2r^4 + k_3r^6)$; tangential p₁, p₂; k₁ < 0 is barrel |
| 11 | Zhang | H = K[r₁ r₂ t]; $\mathbf h_1^\top B\mathbf h_2 = 0$, $\mathbf h_1^\top B\mathbf h_1 = \mathbf h_2^\top B\mathbf h_2$; B = K⁻ᵀK⁻¹; ≥ 3 views |
| 12 | Homography | $\mathbf x' \simeq H\mathbf x$; **8 DOF**; **4 points** (no 3 collinear); planar: H = K[r₁ r₂ t]; rotating camera: H = KRK⁻¹ |
| 13 | Apply H | $x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}$ (similar for y′) |
| 14 | DOF hierarchy | translation 2 / rigid 3 / similarity 4 / affine 6 / projective 8 |
| 15 | Disparity | $d = u_L - u_R$ (rectified, same row) |
| 16 | **Stereo depth** | $Z = fB/d$; $X = (u - c_x)Z/f$; $Y = (v - c_y)Z/f$ |
| 17 | Depth error | $\Delta Z \approx \frac{Z^2}{fB}\Delta d$ (grows with Z²) |
| 18 | Epipolar (F) | $\mathbf x_2^\top F\mathbf x_1 = 0$; $\mathbf l_2 = F\mathbf x_1$; 7 DOF, rank 2, 8-point |
| 19 | Essential | $\hat{\mathbf x}_2^\top E\hat{\mathbf x}_1 = 0$; $E = [\mathbf t]_\times R = K_2^\top FK_1$; 5 DOF; gives R and t (up to scale) |
| 20 | Search reduction | 2D: W·H candidates → 1D: ≤ W (1280×720: 921,600 vs 1,280, i.e. 720×) |
| 21 | Triangulation (DLT) | $u\,\mathbf p_3^\top\mathbf X - \mathbf p_1^\top\mathbf X = 0$, $v\,\mathbf p_3^\top\mathbf X - \mathbf p_2^\top\mathbf X = 0$; AX = 0 (4×4), SVD |
| 22 | Ray | $\mathbf X = Z\cdot K^{-1}[u, v, 1]^\top$ (ray scaled by depth) |
| 23 | Optical flow | $I_xu + I_yv + I_t = 0$ (1 equation, 2 unknowns: the aperture problem) |
| 24 | Lucas–Kanade | $[u, v]^\top = (A^\top A)^{-1}A^\top\mathbf b$, $\mathbf b = -I_t$; needs AᵀA well-conditioned (a corner) |
| 25 | Bundle adjustment | $\min\sum_{ij}\lVert\mathbf x_{ij} - \pi(P_i, \mathbf X_j)\rVert^2$; 6m + 3n unknowns |
| 26 | SfM ambiguity | projective 15 → affine 12 → similarity 7 (scale is always unknown without a reference) |
| 27 | VO chaining | $C_n = C_{n-1}T_n$, $T = [R\ \mathbf t; 0\ 1]$ |
| 28 | Minimum points | homography 4; F 8 (7); E 5; PnP 3 (P3P); camera matrix P by DLT 6 |

---

# PART 11 — Numericals Bank (Section F) — formula → substitute → answer

**N1. Project a point.** f_x = f_y = 800, (c_x, c_y) = (320, 240), R = I, t = 0, X = (0.5, −0.25, 4).
- u = 800·0.5/4 + 320 = **420**; v = 800·(−0.25)/4 + 240 = **190**.

**N2. Project with extrinsics.** Same K; t = (0, 0, 5); X_w = (1, 1, 0).
- X_c = (1, 1, 5), so u = 800·0.2 + 320 = **480** and v = **400**.

**N3. Image size vs distance.** A 1.8 m person, f = 1000 px, at Z = 20 m.
- h = 1000·1.8/20 = **90 px**. At 40 m it is **45 px** (double the distance, half the size).

**N4. Distance from a known height.** A car 1.5 m tall appears 60 px tall.
- f = 1000 → **Z = 25 m**; f = 800 → **20 m**.
- 5 m error ÷ 13.9 m/s (50 km/h) = **0.36 s**.

**N5. Focal length conversion.** A 4 mm lens, 6.4 mm sensor width, 1280 px image.
- f_px = 4·1280/6.4 = **800 px**.
- FOV = 2·atan(1280/1600) = 2·38.66° ≈ **77.3°**.
- With f = 640 px: **90°**.

**N6. DOF counting.**
- **K: 4** (5 with skew). **R, t: 6.** **P = K[R|t]: 11.**
- Calibration with 15 views: 5 + 6·15 = **95** unknowns.
- One view of a 9×6 board gives 54 corners × 2 = **108** equations.

**N7. Homography points.**
- H has 9 − 1 = **8 DOF**, and each pair gives 2 equations, so **4 pairs**.
- An affine transform (6 DOF) needs **3 pairs**.
- A fundamental matrix needs **8** (7 minimum).

**N8. Apply a homography.** H = [[1, 0, 0], [0, 1, 0], [0.001, 0, 1]], point (100, 50).
- w = 0.001·100 + 1 = 1.1, so the point maps to (100/1.1, 50/1.1) = **(90.9, 45.5)**.

**N9. Disparity → depth (KITTI-like).** f = 720 px, B = 0.54 m, fB = 388.8.

| d | Z |
|---|---|
| 70 px | **5.55 m** |
| 24 px | **16.2 m** |
| 23 px | **16.9 m** (a 1-px error = 0.7 m) |
| 8 px | **48.6 m** |
| 0 px | **invalid / ∞** |

**N10. Stereo 3D point.** f = 800, B = 0.10, c = (640, 360), u_L = 520, u_R = 480, v = 300.
- d = 40, so **Z = 2.0 m**, **X = −0.30 m**, **Y = −0.15 m**.

**N11. Depth error.** f·B = 80. With Δd = 1 px:
- at Z = 2 m: ΔZ = 4/80 = **5 cm**
- at Z = 4 m: **20 cm**
- at Z = 8 m: **80 cm**

  **Doubling B halves the error.**

**N12. Required baseline.** You want 10 cm accuracy at 10 m with f = 1000 px and Δd = 0.5 px.
- B = Z²Δd/(f·ΔZ) = 100·0.5/(1000·0.1) = **0.5 m**.

**N13. Disparity at a given depth.** f = 700, B = 0.12 m, object at 6 m.
- d = fB/Z = 84/6 = **14 px**.

**N14. Epipolar search reduction.** 1920×1080 image.
- 2D: **2,073,600** candidates; 1D: **≤ 1,920**; that is **1,080×** fewer.
- (The slide used 1280×720: 921,600 vs 1,280, which is **720×**.)

**N15. Lens distortion.** Normalised (x, y) = (0.2, 0.1), k₁ = −0.2.
- r² = 0.05, factor = 0.99, so **(0.198, 0.099)**: pulled toward the centre (**barrel**).
- With k₁ = +0.2: factor 1.01 → **(0.202, 0.101)** (**pincushion**).

**N16. Reprojection RMSE.** The errors are (0.3, −0.4) and (0, 0.5), with magnitudes 0.5 and 0.5.
- **RMSE = 0.5 px.**

**N17. Lucas–Kanade.** (I_x, I_y, I_t) = (2, 1, −4) and (1, 3, −7).
- 2u + v = 4 and u + 3v = 7, so **u = 1, v = 2** px/frame.

**N18. Aperture case.** Gradients (1, 0, −1) and (2, 0, −2): only the x-gradient exists (a vertical edge).
- AᵀA = [[5, 0], [0, 0]] is singular, so **v cannot be found**. Only u = 1 is known: the aperture problem.

**N19. Bundle adjustment size.** 20 cameras and 5,000 points.
- Unknowns = 6·20 + 3·5000 = **15,120**.
- If every point is seen in 4 images: 5,000·4·2 = **40,000** residuals.

**N20. Pose chaining.** +1 m forward; turn 90° left and +1 m; then +1 m.
- Final position **(2, 1, 0)** (tested with 4×4 matrices).

**N21. Scale from a reference.** A monocular SfM reconstruction measures a phone as 0.5 units long. The real phone is 15 cm.
- Scale = 0.15/0.5 = **0.3 m per unit**. A wall measured at 12 units is **3.6 m**.

**N22. SfM ambiguity DOF.** Projective Q is 4×4 − 1 = **15**; affine **12**; similarity **7** (3 R + 3 t + 1 scale).

---

# PART 12 — Coding Bank (Section D) — tested in Python/NumPy

> **Edge cases the hidden tests love:**
> - points **behind the camera** (Z ≤ 0)
> - **disparity ≤ 0**
> - **fewer than 4** homography pairs
> - **ill-conditioned** LK windows (edges or flat regions)
> - empty inputs
> - forgetting to **divide by the homogeneous coordinate**

```python
import numpy as np

def project_points(K, R, t, X):
    """X: (N,3) world points -> (N,2) pixels. Points with Z<=0 (behind camera) -> nan."""
    X = np.asarray(X, dtype=float).reshape(-1, 3)
    Xc = X @ np.asarray(R, float).T + np.asarray(t, float).reshape(1, 3)   # world -> camera
    uv = np.full((len(X), 2), np.nan)
    ok = Xc[:, 2] > 0                                                       # in front of camera
    x = Xc[ok] @ np.asarray(K, float).T                                     # apply K
    uv[ok] = x[:, :2] / x[:, 2:3]                                           # divide by depth
    return uv
# K=[[800,0,320],[0,800,240],[0,0,1]], R=I, t=0, X=(0.5,-0.25,4) -> (420,190);  X=(0,0,-1) -> nan

def apply_homography(H, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, float).T
    return ph[:, :2] / ph[:, 2:3]                       # divide by w!
# H=[[1,0,0],[0,1,0],[0.001,0,1]], (100,50) -> (90.91, 45.45)

def estimate_homography(src, dst):
    """DLT: needs >= 4 point pairs, no 3 collinear. Returns H with H[2,2] = 1."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("need >= 4 matching point pairs")
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(A))
    H = Vt[-1].reshape(3, 3)            # null-space vector = last right-singular vector
    return H / H[2, 2]
# unit square -> (10,20),(30,20),(32,45),(8,44): re-applying H reproduces dst exactly

def depth_from_disparity(disp, f, B):
    """Z = f*B/d; invalid (d <= 0) -> inf."""
    d = np.asarray(disp, float)
    Z = np.full(d.shape, np.inf)
    ok = d > 0
    Z[ok] = f * B / d[ok]
    return Z
# f=720, B=0.54: d=[24,23,70,8,0] -> [16.2, 16.904, 5.554, 48.6, inf]

def stereo_point(uL, uR, v, f, B, cx, cy):
    d = uL - uR
    if d <= 0:
        return None
    Z = f * B / d
    return ((uL - cx) * Z / f, (v - cy) * Z / f, Z)
# (520,480,300, f=800, B=0.1, cx=640, cy=360) -> (-0.3, -0.15, 2.0)

def triangulate_dlt(P1, P2, x1, x2):
    (u1, v1), (u2, v2) = x1, x2
    A = np.array([u1 * P1[2] - P1[0], v1 * P1[2] - P1[1],
                  u2 * P2[2] - P2[0], v2 * P2[2] - P2[1]])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]
# P1 = K[I|0], P2 = K[I|(-0.1,0,0)], x1=(440,160), x2=(400,160) -> (0.3, -0.2, 2.0)

def reprojection_rmse(observed, predicted):
    o = np.asarray(observed, float); p = np.asarray(predicted, float)
    if len(o) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum((o - p) ** 2, axis=1))))
# [(100,100),(200,200)] vs [(100.3,99.6),(200,200.5)] -> 0.5

def lucas_kanade(Ix, Iy, It):
    """Flow (u, v) for one window from lists of gradients. None if ill-conditioned (aperture problem)."""
    A = np.stack([np.ravel(Ix), np.ravel(Iy)], axis=1).astype(float)
    b = -np.ravel(It).astype(float)
    ATA = A.T @ A
    if np.linalg.matrix_rank(ATA) < 2 or np.linalg.cond(ATA) > 1e6:
        return None
    return np.linalg.solve(ATA, A.T @ b)
# Ix=[2,1], Iy=[1,3], It=[-4,-7] -> (1, 2);  Ix=[1,2], Iy=[0,0] -> None (edge)

def compose(T_list):
    """Chain 4x4 relative transforms: C_n = T_1 @ T_2 @ ... @ T_n."""
    C = np.eye(4)
    for T in T_list:
        C = C @ T
    return C
# [+1 x], [rotZ 90 + 1 x], [+1 x] -> position (2, 1, 0)

def distort(xn, yn, k1=0, k2=0, k3=0, p1=0, p2=0):
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return xd, yd
# distort(0.2, 0.1, k1=-0.2) -> (0.198, 0.099)
```

**OpenCV calls to recognise in "read the code" MCQs:**

| Call | What it does |
|---|---|
| `cv2.findChessboardCorners` → `cv2.cornerSubPix` → `cv2.calibrateCamera` | returns `ret` (RMS), K, dist, rvecs, tvecs |
| `cv2.undistort(img, K, dist)` | remove lens distortion |
| `cv2.findHomography(src, dst, cv2.RANSAC)`, `cv2.warpPerspective(img, H, (w, h))` | estimate and apply a homography |
| `cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC)`, `cv2.findEssentialMat`, `cv2.recoverPose` | epipolar geometry and relative pose |
| `cv2.triangulatePoints(P1, P2, x1, x2)` | returns **4×N homogeneous** points, so divide by the last row |
| `cv2.StereoSGBM_create(...).compute(L, R)` | disparity (OpenCV returns it **×16**, so divide by 16) |
| `cv2.reprojectImageTo3D(disp, Q)` | dense (X, Y, Z) per pixel |
| `cv2.calcOpticalFlowPyrLK` (sparse LK / KLT), `cv2.calcOpticalFlowFarneback` (dense) | optical flow |
| `cv2.solvePnP(obj3d, img2d, K, dist)` | camera pose from 3D↔2D matches |

---

# PART 13 — Case-Study Playbook (Section E)

Use the chain **Problem → Task → Data → Model / geometry → Metric → Failure mode → Mitigation.**

**Case 1 — Parking-assist bird's-eye view (surround view)**
- **Task:** warp 4 fisheye cameras to a top-down ground view.
- **Approach:**
  1. **Calibrate** the intrinsics plus fisheye **distortion** for each camera.
  2. Get the **extrinsics** relative to the car.
  3. **Undistort**, then compute a **ground-plane homography** per camera (4+ known ground markers, RANSAC).
  4. Warp and stitch.
- **Metric:** reprojection error < 0.5 px; ground-marker alignment error in cm.
- **Failure modes:**
  - **Objects with height smear** (H holds only on the plane).
  - Calibration shifts with car load or tyre pressure, so recalibrate online (teacher: ADAS does calibration on the go, using optical flow to drop dynamic objects).

**Case 2 — Stereo depth for ADAS braking**
- **Task:** per-pixel depth to obstacles.
- **Geometry:** a rectified stereo rig, Z = fB/d. **Choose B for the required range:** ΔZ ≈ Z²Δd/(fB). For example, 10 cm at 10 m needs B ≈ 0.5 m.
- **Data/model:** SGBM or a learned stereo network; a left–right consistency check.
- **Metric:** depth error vs range; % of bad pixels; latency.
- **Failure modes:**
  - **Textureless road or sky**, **reflections/glass**, **occlusions**.
  - Night and rain.
  - The far-range error grows with Z².
- **Mitigation:** fuse with radar or LiDAR, use temporal filtering, a wider baseline, or a longer f.

**Case 3 — Warehouse robot navigation (SLAM)**
- **Task:** localise and map in real time.
- **Sensors:** stereo or RGB-D (so the **scale is known**) + IMU + wheel odometry (fusion).
- **Method:**
  - Feature-based VO (ORB) front-end.
  - Landmarks with predict → reproject → correct.
  - Windowed BA.
  - **Loop closure** when revisiting aisles.
- **Metric:** trajectory error (ATE), drift per 100 m, map consistency.
- **Failure modes:**
  - **Repetitive aisles** cause false loop closures.
  - Moving forklifts and people (the static-scene assumption breaks), so **mask dynamic objects with optical flow or segmentation**.
  - Low light; white walls (textureless).
  - **Wheel slip** for wheel odometry.

**Case 4 — Drone survey: 3D model of a building (SfM)**
- **Task:** reconstruct a 3D model from hundreds of photos (offline is fine).
- **Method:**
  1. SIFT matching + RANSAC.
  2. Incremental SfM: E → PnP → triangulate → **bundle adjustment**.
  3. Dense MVS, then a mesh.
- **Scale:** use **GPS tags or a known reference length**, because monocular SfM has no absolute scale.
- **Failure modes:**
  - Low overlap between photos.
  - Glass façades and repetitive windows.
  - Moving shadows, motion blur.

**Case 5 — Mobile document scanner**
- **Approach:** detect the page's 4 corners → **homography** to a fronto-parallel rectangle → warpPerspective.
- **Why it works:** a page is a **plane**, so one H is exact.
- **Failure modes:**
  - A curled page is not planar.
  - Corners hidden by fingers.
  - Fewer than 4 corners or **3 collinear**, so H is degenerate.
- **Mitigation:** contour detection with a quadrilateral fit; let the user adjust the corners.

**Case 6 — AR furniture placement**
- **Needs:** camera pose every frame (**VO/SLAM**), plane detection (**homography/RANSAC** on the floor), and metric scale (IMU or depth sensor).
- **Failure modes:** textureless floors, fast motion blur, drift making objects "slide" (loop closure and re-localisation help).

---

# PART 14 — Question Bank

## A. Single-correct MCQs (answers below)

1. Which parameters change when the camera is moved to a new position? (a) f_x, f_y (b) c_x, c_y (c) R, t (d) distortion coefficients
2. Minimum point correspondences to estimate a homography: (a) 3 (b) 4 (c) 6 (d) 8
3. In rectified stereo, depth is: (a) proportional to disparity (b) inversely proportional to disparity (c) independent of baseline (d) proportional to d²
4. f = 700 px, B = 0.1 m, d = 35 px. Z = ? (a) 0.5 m (b) 2 m (c) 20 m (d) 245 m
5. The matrix relating corresponding **pixel** coordinates of two **uncalibrated** cameras: (a) K (b) H (c) F (d) E
6. Where is the epipole for a camera moving straight forward? (a) at infinity (b) the image centre (focus of expansion) (c) the top-left corner (d) undefined
7. Barrel distortion corresponds to: (a) k₁ > 0 (b) k₁ < 0 (c) p₁ ≠ 0 (d) f_x ≠ f_y
8. The reprojection error measures: (a) the pixel distance between detected and projected points (b) the depth error in metres (c) the colour difference (d) the rotation error
9. The optical-flow constraint gives per pixel: (a) 2 equations, 2 unknowns (b) 1 equation, 2 unknowns (c) 2 equations, 1 unknown (d) 0 equations
10. Accumulated error in visual odometry is called: (a) aliasing (b) drift (c) parallax (d) jitter
11. Which removes VO drift? (a) bigger images (b) loop closure (c) higher fps only (d) a smaller baseline
12. A ground-plane homography applied to a tall box: (a) maps it perfectly (b) smears its height along the ground (c) removes it (d) doubles it
13. Monocular SfM cannot recover: (a) rotation (b) relative structure (c) absolute scale (d) correspondences
14. The essential matrix has how many DOF? (a) 5 (b) 7 (c) 8 (d) 11

**Answers:**

| Q | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ans | c | b | b | **b** | c | b | b | a | b | b | b | b | c | a |

- Q4: 700 × 0.1/35 = 2 m.
- Q1 (closest distractor): distortion is **intrinsic**, so it stays the same when the camera moves.
- Q5: E needs calibrated (normalised) coordinates.

## B. Multi-select (all correct and no wrong options for full marks)

1. **Intrinsic parameters include:** (a) f_x (b) c_y (c) the rotation R (d) skew (e) the translation t → **a, b, d**
2. **A single homography correctly relates two images when:** (a) the scene is planar (b) the camera purely rotates (c) the scene has arbitrary depth and the camera translates (d) both images are of the same plane from different viewpoints → **a, b, d**
3. **Stereo matching struggles with:** (a) textureless walls (b) reflections/glass (c) occluded regions (d) richly textured brick walls (e) repetitive patterns → **a, b, c, e**
4. **True about F and E:** (a) F works on raw pixels (b) E needs K (c) E gives R and t up to scale (d) F has 5 DOF (e) the 8-point algorithm estimates F → **a, b, c, e** (F has 7 DOF)
5. **A wider stereo baseline:** (a) improves far-depth accuracy (b) increases the blind zone (c) makes matching easier (d) increases disparity for a given depth → **a, b, d**
6. **Visual-odometry assumptions:** (a) sufficient illumination (b) static scene dominates (c) enough texture (d) sufficient overlap between frames (e) GPS availability → **a, b, c, d**
7. **Ways to estimate VO motion:** (a) 2D–2D via E (b) 3D–3D alignment from stereo (c) 3D–2D via PnP (d) histogram equalisation → **a, b, c**
8. **Reduce the reconstruction ambiguity toward metric:** (a) parallelism (affine) (b) orthogonality / calibrated K (similarity) (c) one known real-world length (scale) (d) more blur → **a, b, c**
9. **Calibration using a checkerboard:** (a) uses Z = 0 for all corners (b) needs multiple views at different orientations (c) refines corners to sub-pixel accuracy (d) needs only one image for full K and distortion (e) minimises the reprojection error → **a, b, c, e**
10. **Lucas–Kanade fails or is unreliable when:** (a) the window lies on a straight edge (b) the region is flat (c) the motion is very large without a pyramid (d) the window lies on a corner with small motion → **a, b, c**

## C. Descriptive (2 marks, ≤ 100 words): model answers

**C1. Intrinsic vs extrinsic parameters.**
> Intrinsic parameters describe the camera itself: focal lengths f_x, f_y (in pixels), the principal point c_x, c_y and skew, packed into K, plus lens distortion. They map camera coordinates to pixels and do not change when the camera moves. Extrinsic parameters, rotation R (3 DOF) and translation t (3 DOF), describe the camera's pose in the world, X_cam = R·X_world + t, and change whenever the camera moves. Together, x ≃ K[R|t]X.

**C2. Why calibrate a camera?**
> Every 3D measurement depends on K, R, t and distortion. A wrong focal length scales all depths: a 1.5 m car appearing 60 px tall is 25 m away with the calibrated f = 1000 px, but 20 m with a guessed 800 px, a 0.36 s braking difference at 50 km/h. Calibration (checkerboard, Zhang's method) estimates these by minimising the reprojection error, and undistortion makes straight lines straight so that homography and stereo maths hold.

**C3. Why does a homography need at least 4 point correspondences?**
> A homography is a 3×3 matrix defined only up to scale, so it has 9 − 1 = 8 degrees of freedom. Each point correspondence gives two equations (one for x′, one for y′), so 8/2 = 4 pairs are needed, with no three collinear, or the system becomes degenerate. In practice many matches are used with RANSAC to reject outliers.

**C4. Limitation of homography-based bird's-eye view.**
> A homography is exact only for points on one plane (or a purely rotating camera). In a ground-plane bird's-eye view, the footprint of an object is placed correctly, but anything with height (cars, poles, people) gets stretched and smeared along the ground away from the camera, because those points violate the planar assumption.

**C5. Derive/explain Z = fB/d.**
> In a rectified stereo pair with baseline B and focal length f, a point at depth Z appears at x_L and x_R on the same row; disparity d = x_L − x_R. Triangles P–O_L–O_R and P–x_L–x_R are similar, giving B/Z = (B − d)/(Z − f), which simplifies to Z = fB/d. Depth is inversely proportional to disparity: near objects shift more. The error grows as ΔZ ≈ Z²Δd/(fB).

**C6. Baseline trade-off.**
> A wider baseline gives larger disparities, so depth far away is more accurate (ΔZ ∝ Z²/(fB)). But the two views differ more, so matching is harder and occlusions increase, and the near-field blind zone seen by only one camera grows. A narrow baseline makes matching easy and works close up, but gives poor far-range depth. Human eyes use ~6.5 cm; KITTI's car rig uses ~54 cm.

**C7. Epipolar constraint, and why it matters.**
> For a pixel x in image 1, the 3D point lies somewhere on its back-projected ray. All candidate points on that ray project onto a single line l′ = Fx in image 2, the epipolar line, which is the intersection of the epipolar plane (O, O′, P) with the image. So the correspondence satisfies x′ᵀFx = 0, and matching becomes a 1D search along a line instead of a 2D search (≈ 720× fewer candidates for 1280×720), making it faster with fewer false matches.

**C8. Fundamental vs essential matrix.**
> Both encode two-view epipolar geometry. The fundamental matrix F (7 DOF, rank 2) relates raw pixel coordinates, x₂ᵀFx₁ = 0, needs no calibration, and is estimated from ≥ 8 matches; it gives epipolar lines. The essential matrix E = [t]ₓR = K₂ᵀFK₁ (5 DOF) works on normalised coordinates of calibrated cameras and can be decomposed into the relative rotation and translation (translation only up to scale).

**C9. Why do triangulated rays not meet, and what do we do?**
> Pixel noise, calibration errors and pixel quantisation make the two back-projected rays skew, passing near each other rather than intersecting. We estimate the point as the midpoint of the shortest segment between the rays, or better, the point minimising the reprojection error in both images. Linear DLT stacks two equations per view into AX = 0 and solves it with SVD. A rectified pair simply uses Z = fB/d.

**C10. What is the aperture problem?**
> Brightness constancy gives one equation per pixel, I_x·u + I_y·v + I_t = 0, with two unknowns. Through a small window on a straight edge, only the motion component perpendicular to the edge is observable; motion along the edge produces no change. Lucas–Kanade adds the assumption that flow is constant in a window and solves by least squares, which works only where gradients vary in two directions (corners).

**C11. Scale ambiguity in monocular SfM/VO.**
> Image projections are unchanged if the whole scene and the camera translations are scaled by the same factor: a bigger scene viewed from further away looks identical. So monocular reconstruction gives structure and translation only up to an unknown scale; rotation is unaffected. Absolute scale needs a reference: a known object size, a stereo baseline, IMU, GPS or wheel odometry.

**C12. Bundle adjustment.**
> Bundle adjustment jointly refines all camera poses (and possibly intrinsics) and all 3D points by minimising the total reprojection error, Σ‖x_ij − π(P_i, X_j)‖², usually with Levenberg–Marquardt. It is the final, accuracy-critical step of SfM, run over all images offline. VO and SLAM use a windowed version over recent keyframes for real-time speed.

**C13. VO vs SLAM vs SfM.**
> SfM reconstructs structure and camera poses from an unordered image set offline, with full bundle adjustment, prioritising quality. Visual odometry estimates the camera trajectory sequentially in real time by chaining frame-to-frame motions, so errors accumulate (drift). SLAM adds a persistent map of landmarks and loop closure: recognising a revisited place lets a global optimisation correct the drift, giving both localisation and a consistent map.

**C14. Why must the scene be mostly static for VO, and how do you handle moving objects?**
> VO assumes apparent motion is caused only by the camera. Moving objects violate this: if someone walks alongside the camera, the view barely changes and the system may think it did not move. Dynamic regions are detected because their optical flow does not follow the global camera-induced flow pattern (an anomaly), and they are masked out (or segmented) before pose estimation, calibration or mapping.

**C15. Why is a checkerboard used for calibration?**
> Its geometry is known: measuring one square size gives every corner's 3D coordinates. Its black–white X-junctions can be localised to sub-pixel accuracy, it is high-contrast and easy to detect automatically, and it is flat (Z = 0), so each view relates to the image by a homography. That simplifies Zhang's method for estimating K, then R, t and distortion.

---

# PART 15 — Traps, Transcript Corrections & Checklist

## 15.1 Transcript / slide fixes

| Correct | Transcript or slide said |
|---|---|
| **Visual odometry**, **SLAM**, **bundle adjustment**, **IMU**, **Middlebury**, **intrinsics**, **essential matrix**, **PnP** | "visual photometry/automating/otometry", "slab", "bundle access", "INU", "middle-line stereo", "in GEN6", "ascension/sensing matrix", "PNP homework" |
| **Richard Szeliski**, *Computer Vision: Algorithms and Applications* (the course book) | "Richard Silsky's book" |
| Homography slides are numbered 28–35 but come **before** calibration in the PDF; the PDF module order is Projection → Homography → Calibration → Epipolar → Triangulation → Stereo | — |
| The slide says K has "4 intrinsic unknowns"; the calibration infographic counts **5 (with skew γ)**. Both are fine; state your convention | — |

## 15.2 Classic traps

1. **Intrinsics don't change when the camera moves; extrinsics do.** Distortion is intrinsic.
2. **Homography: 8 DOF, 4 points, no 3 collinear.** It is exact only for **planes** or **pure rotation**.
3. **Z = fB/d:** d in **pixels**, f in **pixels**, B in **metres**. **d = 0 is invalid.** The error grows with **Z²**.
4. **Wider baseline:** better far depth, but **harder matching and a bigger blind zone**.
5. **F:** uncalibrated, pixels, 7 DOF, rank 2, 8-point. **E:** calibrated, normalised, 5 DOF, gives R and t **up to scale**.
6. The **epipole** = the other camera's centre in this image. **Rectified: at infinity. Forward motion: the image centre (FOE).**
7. **Triangulation needs both the calibration AND the correspondence.**
8. **The optical-flow constraint is 1 equation in 2 unknowns** (the aperture problem). LK works at **corners**.
9. **Monocular SfM/VO: scale is unknown. Stereo/RGB-D: scale is known.**
10. **VO drifts. SLAM fixes it with landmarks + loop closure.** SfM is offline with full BA.
11. **Reprojection error appears everywhere:** calibration, triangulation, BA, PnP, SLAM.
12. A **ground-plane BEV smears objects with height**.

## 15.3 Memory hooks

- **Projection chain:** *"Move (R|t), then squash (K), then divide (÷Z)."*
- **H vs F:** *"H maps a point to a point (plane). F maps a point to a line (any scene)."*
- **Depth:** *"Near jumps more"* (the finger test): Z = fB/d.
- **VO vs SLAM:** *"VO remembers the path; SLAM remembers the place."*

## 15.4 Last-night checklist

- [ ] Write u = f_x·X/Z + c_x and x ≃ K[R|t]X, and say what each symbol is
- [ ] Intrinsics vs extrinsics table; P has 11 DOF
- [ ] Calibration steps + reprojection error + distortion formula (barrel vs pincushion)
- [ ] Derive H = K[r₁ r₂ t] from Z = 0; 8 DOF → 4 points; applications and limitation
- [ ] Derive Z = fB/d; compute the slide examples (16.2 m, 2 m with X = −0.3, Y = −0.15); ΔZ ∝ Z²
- [ ] Baseline trade-off; stereo failure cases; LR check; stereo as optimisation (photometric + smoothness)
- [ ] Epipolar plane, line, epipole; F vs E table; 921,600 → 1,280 (720×)
- [ ] Triangulation: midpoint, reprojection, DLT
- [ ] Optical flow: constraint equation, aperture problem, LK solve (u = 1, v = 2), FOE, dynamic-object masking
- [ ] SfM pipeline, BA formula, ambiguity table (15/12/7), scale ambiguity + fix
- [ ] VO: setups, T_k chaining, 4 assumptions with reasons, other odometry + failures, 2D–2D / 3D–3D / 3D–2D, drift
- [ ] SLAM loop (predict → reproject → correct) and loop closure; VO vs SLAM vs SfM table
